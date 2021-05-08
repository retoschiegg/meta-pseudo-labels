"""Start training procedure of classifier."""
import os
import sys
import logging
import shutil

import tensorflow as tf
import tensorflow_addons as tfa

import train_util
from dataset import (
    get_labeled_train_ds,
    get_unlabeled_train_ds,
    get_val_ds,
)
from models import (
    get_student_model,
    get_teacher_model,
    get_finetune_model,
)

LOGGER = logging.getLogger(__name__)


def get_student_file(mpl_dir):
    """Get path of student model."""
    return os.path.join(mpl_dir, "student")


def get_ema_file(mpl_dir):
    """Get path of ema model."""
    return os.path.join(mpl_dir, "ema")


def get_exported_student_file(mpl_dir):
    """Get path of student model."""
    return os.path.join(mpl_dir, "exported-student")


def get_exported_ema_file(mpl_dir):
    """Get path of ema model."""
    return os.path.join(mpl_dir, "exported-ema")


def get_teacher_file(mpl_dir):
    """Get path of teacher model."""
    return os.path.join(mpl_dir, "teacher")


def get_finetune_file(training_dir):
    """Get path of final model."""
    return os.path.join(training_dir, "model")


def train_finetune(
    finetune_dir,
    mpl_model_dir,
    data_dir,
    train_config,
    config_name,
):
    """Finetune mpl model."""
    # long method as it contains whole training loop
    # pylint: disable=too-many-locals,too-many-statements

    if os.path.exists(finetune_dir):
        shutil.rmtree(finetune_dir)
    if not os.path.exists(finetune_dir):
        os.makedirs(finetune_dir)

    _, labeled_dataset, num_steps, val_dataset = init_training(
        train_config,
        finetune_dir,
        data_dir,
        train_config.finetune_batch_size,
        config_name,
    )

    LOGGER.info(
        "Number of steps per epoch: %s", num_steps,
    )

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        finetune_model_file = get_finetune_file(finetune_dir)
        finetune_model = get_finetune_model(train_config, mpl_model_dir)

        optimizer = tf.keras.optimizers.SGD(
            momentum=train_config.finetune_optimizer_momentum,
            nesterov=train_config.finetune_optimizer_nesterov,
        )

        best_val_loss = sys.maxsize
        step = 0
        total_steps = train_config.finetune_epochs * num_steps

        finetune_loss = tf.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE,
        )
        train_loss_metric = tf.keras.metrics.Mean()
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
        val_loss_metric = tf.keras.metrics.Mean()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

        @tf.function
        def val_step(images, label_indices):
            predictions = tf.nn.softmax(finetune_model(images, training=False), -1)
            loss = finetune_loss(
                y_true=train_util.get_onehot(label_indices, train_config.labels),
                y_pred=predictions,
            )
            val_loss_metric(tf.reduce_sum(loss) / float(images.shape[0]))
            val_acc_metric.update_state(label_indices, predictions)

        @tf.function
        def train_step(images, label_indices):
            with tf.GradientTape() as tape:
                predictions = tf.nn.softmax(finetune_model(images, training=True), -1)
                loss = finetune_loss(
                    y_true=train_util.get_onehot(label_indices, train_config.labels),
                    y_pred=predictions,
                )
            gradient = tape.gradient(loss, finetune_model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, finetune_model.trainable_variables))
            train_loss_metric(tf.reduce_sum(loss) / float(images.shape[0]))
            train_acc_metric.update_state(label_indices, predictions)

        dist_labeled_dataset = strategy.experimental_distribute_dataset(labeled_dataset)
        dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)
        for epoch in range(train_config.finetune_epochs):

            for images, label_indices in dist_labeled_dataset:
                step += 1
                optimizer.learning_rate.assign(
                    train_util.get_learning_rate(
                        step,
                        train_config.finetune_learning_rate,
                        total_steps,
                        train_config.finetune_learning_rate_warmup,
                        0,
                    )
                )
                strategy.run(train_step, args=(images, label_indices))
                if step % 10 == 0:
                    tf.summary.scalar("finetune-lr", data=optimizer.learning_rate, step=step)

            train_acc = train_acc_metric.result()
            train_loss = train_loss_metric.result()
            train_acc_metric.reset_states()
            train_loss_metric.reset_states()
            tf.summary.scalar("finetune-train/acc", data=train_acc, step=step)
            tf.summary.scalar("finetune-train/loss", data=train_loss, step=step)

            for images, label_indices in dist_val_dataset:
                strategy.run(val_step, args=(images, label_indices))

            val_loss = val_loss_metric.result()
            val_acc = val_acc_metric.result()
            val_loss_metric.reset_states()
            val_acc_metric.reset_states()
            tf.summary.scalar("finetune-val/loss", data=val_loss, step=step)
            tf.summary.scalar("finetune-val/acc", data=val_acc, step=step)
            tf.summary.flush()

            LOGGER.info(
                "Epoch %s: train-loss=%s, train-acc=%s, val-loss=%s, val-acc=%s",
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                tf.keras.models.save_model(finetune_model, finetune_model_file)

    return finetune_model_file


def init_training(
    train_config,
    training_dir,
    data_dir,
    batch_size,
    config_name,
):
    """Initialize training."""
    # for tensorbaord
    tensorboard_dir = os.path.join(training_dir, "log")
    file_writer = tf.summary.create_file_writer(tensorboard_dir)
    file_writer.set_as_default()

    num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
    LOGGER.info("Number of gpus found: %s", num_gpus)

    global_batch_size = batch_size * num_gpus

    labeled_dataset, num_steps = get_labeled_train_ds(
        train_config, data_dir, global_batch_size, config_name
    )
    val_dataset = get_val_ds(
        train_config, data_dir, global_batch_size, config_name
    )

    return num_gpus, labeled_dataset, num_steps, val_dataset


def train_mpl(
    training_dir,
    data_dir,
    config_name,
    train_config,
    log_every_step,
    continue_mpl,
):
    """Start mpl training."""
    # long method as it contains whole training loop
    # pylint: disable=too-many-locals,too-many-statements,too-many-nested-blocks,too-many-branches

    if not continue_mpl and os.path.exists(training_dir):
        shutil.rmtree(training_dir)
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    teacher_checkpoint_dir = get_teacher_file(training_dir)
    student_checkpoint_dir = get_student_file(training_dir)
    ema_checkpoint_dir = get_ema_file(training_dir)
    exported_student_checkpoint_dir = get_exported_student_file(training_dir)
    exported_ema_checkpoint_dir = get_exported_ema_file(training_dir)

    num_gpus, labeled_dataset, num_steps, val_dataset = init_training(
        train_config,
        training_dir,
        data_dir,
        train_config.mpl_batch_size,
        config_name,
    )

    unlabeled_batch_size = (
        train_config.mpl_batch_size * train_config.mpl_unlabeled_batch_size_multiplier
    )
    unlabeled_dataset = get_unlabeled_train_ds(
        train_config, data_dir, unlabeled_batch_size * num_gpus, config_name
    )

    LOGGER.info("Number of steps per epoch: %s", num_steps)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        teacher_model = get_teacher_model(train_config)
        student_model = get_student_model(train_config)
        ema_model = get_student_model(train_config)

        teacher_optimizer = tfa.optimizers.SGDW(
            weight_decay=train_config.mpl_optimizer_weight_decay,
            momentum=train_config.mpl_optimizer_momentum,
            nesterov=train_config.mpl_optimizer_nesterov,
        )
        student_optimizer = tfa.optimizers.SGDW(
            weight_decay=train_config.mpl_optimizer_weight_decay,
            momentum=train_config.mpl_optimizer_momentum,
            nesterov=train_config.mpl_optimizer_nesterov,
        )

        teacher_checkpoint = tf.train.Checkpoint(
            optimizer=teacher_optimizer, model=teacher_model
        )
        teacher_checkpoint_manager = tf.train.CheckpointManager(
            teacher_checkpoint, teacher_checkpoint_dir, 3
        )
        student_checkpoint = tf.train.Checkpoint(
            optimizer=student_optimizer, model=student_model
        )
        student_checkpoint_manager = tf.train.CheckpointManager(
            student_checkpoint, student_checkpoint_dir, 3
        )
        ema_checkpoint = tf.train.Checkpoint(model=ema_model)
        ema_checkpoint_manager = tf.train.CheckpointManager(
            ema_checkpoint, ema_checkpoint_dir, 3
        )

        best_val_loss = sys.maxsize
        step = tf.Variable(0, dtype=tf.int64)
        continue_step = 0
        total_steps = train_config.mpl_epochs * num_steps

        if continue_mpl:
            teacher_checkpoint.restore(
                teacher_checkpoint_manager.latest_checkpoint
            ).expect_partial()
            student_checkpoint.restore(
                student_checkpoint_manager.latest_checkpoint
            ).expect_partial()
            ema_checkpoint.restore(
                ema_checkpoint_manager.latest_checkpoint
            ).expect_partial()
            continue_step = student_checkpoint.save_counter * num_steps

        training_mean_metrics = {
            "mpl-uda/u-ratio": tf.keras.metrics.Mean(),
            "mpl-uda/l-ratio": tf.keras.metrics.Mean(),
            "mpl/dot-product": tf.keras.metrics.Mean(),
            "mpl/moving-dot-product": tf.keras.metrics.Mean(),
            "mpl-cross-entropy/teacher-on-l": tf.keras.metrics.Mean(),
            "mpl-cross-entropy/teacher-on-u": tf.keras.metrics.Mean(),
            "mpl-cross-entropy/student-on-u": tf.keras.metrics.Mean(),
            "mpl-cross-entropy/student-on-l": tf.keras.metrics.Mean(),
        }
        val_student_loss_metric = tf.keras.metrics.Mean()
        val_student_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_acc_student"
        )
        val_ema_loss_metric = tf.keras.metrics.Mean()
        val_ema_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_acc_ema"
        )

        mpl_loss = tf.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE,
        )
        student_unlabeled_loss = tf.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
            label_smoothing=train_config.mpl_label_smoothing,
        )
        student_labeled_loss = tf.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE,
        )
        uda_cross_entropy = train_util.uda_cross_entropy_fn(train_config, total_steps)

        @tf.function
        def val_step(images, label_indices):
            ema_predictions = tf.nn.softmax(ema_model(images, training=False), -1)
            student_predictions = tf.nn.softmax(
                student_model(images, training=False), -1
            )

            def update_metrics(predictions, val_loss_metric, val_acc_metric):
                loss = student_labeled_loss(
                    y_true=train_util.get_onehot(label_indices, train_config.labels),
                    y_pred=predictions,
                )
                val_loss_metric(tf.reduce_sum(loss) / float(images.shape[0]))
                val_acc_metric.update_state(label_indices, predictions)

            update_metrics(ema_predictions, val_ema_loss_metric, val_ema_acc_metric)
            update_metrics(
                student_predictions, val_student_loss_metric, val_student_acc_metric
            )

        @tf.function
        def train_step(
            step_tensor,
            labeled_images,
            label_indices,
            org_images,
            aug_images,
            uda_weight,
        ):
            all_images = tf.concat([labeled_images, org_images, aug_images], 0)

            with tf.GradientTape() as ttape:
                all_logits = teacher_model(all_images, training=True)
                logits, labels, masks, cross_entropy = uda_cross_entropy(
                    all_logits, label_indices, step_tensor
                )

            with tf.GradientTape() as stape:
                u_aug_and_l_images = tf.concat([aug_images, labeled_images], 0)
                logits["s_on_u_aug_and_l"] = student_model(
                    u_aug_and_l_images, training=True
                )
                logits["s_on_u"], logits["s_on_l_old"] = tf.split(
                    logits["s_on_u_aug_and_l"],
                    [aug_images.shape[0], labeled_images.shape[0]],
                    0,
                )

                # for backprop
                cross_entropy["s_on_u"] = student_unlabeled_loss(
                    y_true=tf.stop_gradient(tf.nn.softmax(logits["u_aug"], -1)),
                    y_pred=logits["s_on_u"],
                )
                cross_entropy["s_on_u"] = tf.reduce_sum(
                    cross_entropy["s_on_u"]
                ) / float(unlabeled_batch_size)

                # for Taylor
                cross_entropy["s_on_l_old"] = student_labeled_loss(
                    y_true=labels["l"], y_pred=logits["s_on_l_old"]
                )
                cross_entropy["s_on_l_old"] = tf.reduce_sum(
                    cross_entropy["s_on_l_old"]
                ) / float(train_config.mpl_batch_size)

            student_grad_unlabeled = stape.gradient(
                cross_entropy["s_on_u"], student_model.trainable_variables
            )
            student_grad_unlabeled, _ = tf.clip_by_global_norm(
                student_grad_unlabeled, train_config.mpl_optimizer_grad_bound
            )
            student_optimizer.apply_gradients(
                zip(student_grad_unlabeled, student_model.trainable_variables)
            )

            logits["s_on_l_new"] = student_model(labeled_images)
            cross_entropy["s_on_l_new"] = student_labeled_loss(
                y_true=labels["l"], y_pred=logits["s_on_l_new"]
            )
            cross_entropy["s_on_l_new"] = tf.reduce_sum(
                cross_entropy["s_on_l_new"]
            ) / float(train_config.mpl_batch_size)

            dot_product = cross_entropy["s_on_l_new"] - cross_entropy["s_on_l_old"]
            # limit = 3.0**(0.5)
            moving_dot_product = tf.compat.v1.get_variable(
                "moving_dot_product",
                # initializer=tf.compat.v1.random_uniform_initializer(
                #    minval=-limit, maxval=limit, seed=train_config.seed
                # ),
                trainable=False,
                shape=dot_product.shape,
            )
            moving_dot_product -= 0.01 * (moving_dot_product - dot_product)
            dot_product = dot_product - moving_dot_product
            dot_product = tf.stop_gradient(dot_product)

            with ttape:
                cross_entropy["mpl"] = mpl_loss(
                    y_true=tf.stop_gradient(tf.nn.softmax(logits["u_aug"], -1)),
                    y_pred=logits["u_aug"],
                )
                cross_entropy["mpl"] = tf.reduce_sum(cross_entropy["mpl"]) / float(
                    unlabeled_batch_size
                )

                # teacher train op
                teacher_loss = (
                    cross_entropy["u"] * uda_weight
                    + cross_entropy["l"]
                    + cross_entropy["mpl"] * dot_product
                )

            teacher_grad = ttape.gradient(
                teacher_loss, teacher_model.trainable_variables
            )
            teacher_grad, _ = tf.clip_by_global_norm(
                teacher_grad, train_config.mpl_optimizer_grad_bound
            )
            teacher_optimizer.apply_gradients(
                zip(teacher_grad, teacher_model.trainable_variables)
            )

            return {
                "mpl-uda/u-ratio": tf.reduce_mean(masks["u"]),
                "mpl-uda/l-ratio": tf.reduce_mean(masks["l"]),
                "mpl/dot-product": dot_product,
                "mpl/moving-dot-product": moving_dot_product,
                "mpl-cross-entropy/teacher-on-l": cross_entropy["l"],
                "mpl-cross-entropy/teacher-on-u": cross_entropy["u"],
                "mpl-cross-entropy/student-on-u": cross_entropy["s_on_u"],
                "mpl-cross-entropy/student-on-l": cross_entropy["s_on_l_new"],
            }

        def dist_train_step(
            step_tensor, labeled_images, label_indices, org_images, aug_images
        ):
            uda_weight = train_config.uda_weight * tf.math.minimum(
                1.0, float(step_tensor) / float(train_config.uda_steps)
            )

            per_replica_losses = strategy.run(
                train_step,
                args=(
                    step_tensor,
                    labeled_images,
                    label_indices,
                    org_images,
                    aug_images,
                    uda_weight,
                ),
            )

            averaged_values = {"mpl-uda/weight": uda_weight}
            for key, value in per_replica_losses.items():
                averaged_values[key] = strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, value, axis=None
                )
            return averaged_values

        def dist_val_step(images, label_indices):
            strategy.run(val_step, args=(images, label_indices))

        def update_val_metrics(val_loss_metric, val_acc_metric):
            val_loss = val_loss_metric.result()
            val_acc = val_acc_metric.result()
            val_loss_metric.reset_states()
            val_acc_metric.reset_states()
            return val_loss, val_acc

        dist_labeled_dataset = strategy.experimental_distribute_dataset(labeled_dataset)
        dist_unlabeled_dataset = strategy.experimental_distribute_dataset(
            unlabeled_dataset
        )
        dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)

        dist_unlabeled_dataset = iter(dist_unlabeled_dataset)
        for epoch in range(train_config.mpl_epochs):

            # optimizer.learning_rate false positive
            # pylint: disable=no-member

            for labeled_images, label_indices in dist_labeled_dataset:
                org_images, aug_images = dist_unlabeled_dataset.next()
                step += 1

                student_optimizer.learning_rate.assign(
                    train_util.get_learning_rate(
                        step,
                        train_config.student_learning_rate,
                        total_steps,
                        train_config.student_learning_rate_warmup,
                        train_config.student_learning_rate_numwait,
                    )
                )
                teacher_optimizer.learning_rate.assign(
                    train_util.get_learning_rate(
                        step,
                        train_config.teacher_learning_rate,
                        total_steps,
                        train_config.teacher_learning_rate_warmup,
                        train_config.teacher_learning_rate_numwait,
                    )
                )

                averaged_values = dist_train_step(
                    step, labeled_images, label_indices, org_images, aug_images
                )

                train_util.update_ema_weights(train_config, ema_model, student_model, step)

                for key, metric in training_mean_metrics.items():
                    metric(averaged_values[key])

                if step % log_every_step == 0:
                    if step >= 500:
                        for key, value in averaged_values.items():
                            mean_metric = training_mean_metrics.get(key)
                            if mean_metric:
                                mean_value = mean_metric.result()
                                mean_metric.reset_states()
                            else:
                                mean_value = value
                            tf.summary.scalar(
                                key, data=mean_value, step=step + continue_step
                            )
                    tf.summary.scalar(
                        "mpl-lr/student",
                        data=student_optimizer.learning_rate,
                        step=step + continue_step,
                    )
                    tf.summary.scalar(
                        "mpl-lr/teacher",
                        data=teacher_optimizer.learning_rate,
                        step=step + continue_step,
                    )
                    tf.summary.flush()

            if (
                epoch % train_config.eval_every_epoch == 0
                or epoch == train_config.mpl_epochs - 1
            ):
                for images, label_indices in dist_val_dataset:
                    dist_val_step(images, label_indices)

                val_ema_loss, val_ema_acc = update_val_metrics(
                    val_ema_loss_metric, val_ema_acc_metric
                )
                val_student_loss, val_student_acc = update_val_metrics(
                    val_student_loss_metric, val_student_acc_metric
                )

                tf.summary.scalar(
                    "mpl-val/ema-loss", data=val_ema_loss, step=step + continue_step
                )
                tf.summary.scalar(
                    "mpl-val/ema-acc", data=val_ema_acc, step=step + continue_step
                )
                tf.summary.scalar(
                    "mpl-val/student-loss", data=val_student_loss, step=step + continue_step
                )
                tf.summary.scalar(
                    "mpl-val/student-acc", data=val_student_acc, step=step + continue_step
                )
                LOGGER.info(
                    "Epoch %s: val-ema-loss=%s, val-ema-acc=%s",
                    epoch + 1,
                    val_ema_loss,
                    val_ema_acc,
                )
                LOGGER.info(
                    "Epoch %s: val-student-loss=%s, val-student-acc=%s",
                    epoch + 1,
                    val_student_loss,
                    val_student_acc,
                )

                teacher_checkpoint_manager.save()
                student_checkpoint_manager.save()
                ema_checkpoint_manager.save()
                if val_ema_loss <= best_val_loss:
                    best_val_loss = val_ema_loss
                    tf.keras.models.save_model(
                        student_model, exported_student_checkpoint_dir
                    )
                    tf.keras.models.save_model(ema_model, exported_ema_checkpoint_dir)
            else:
                LOGGER.info(
                    "Epoch %s: skipped validation, eval done every %s epoch",
                    epoch + 1,
                    train_config.eval_every_epoch,
                )
