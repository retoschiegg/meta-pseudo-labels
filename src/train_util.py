"""Utility functions for training."""
import numpy as np
import tensorflow as tf


def update_ema_weights(train_config, ema_model, student_model, step):
    """Update according to ema and return new weights."""
    ema_step = float(step - train_config.ema_start)
    decay = 1.0 - min(train_config.ema_decay, (ema_step + 1.0) / (ema_step + 10.0))
    decay = 1.0 if step < train_config.ema_start else decay

    new_weights = []
    for curr, new in zip(ema_model.get_weights(), student_model.get_weights()):
        new_weights.append(curr * (1 - decay) + new * decay)
    ema_model.set_weights(new_weights)


def get_learning_rate(
    global_step, learning_rate_base, total_steps, num_warmup_steps=0, num_wait_steps=0
):
    """Get learning rate."""
    if global_step < num_wait_steps:
        return 1e-9

    global_step = global_step - num_wait_steps
    if num_warmup_steps > total_steps:
        num_warmup_steps = total_steps - 1
    rate = cosine_decay_with_warmup(
        global_step,
        learning_rate_base,
        total_steps - num_wait_steps,
        warmup_steps=num_warmup_steps,
    )
    return rate


def cosine_decay_with_warmup(
    global_step,
    learning_rate_base,
    total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=0,
    hold_base_rate_steps=0,
):
    """Cosine decay schedule with warm up period.

    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError("total_steps must be larger or equal to " "warmup_steps.")
    learning_rate = (
        0.5
        * learning_rate_base
        * (
            1
            + np.cos(
                np.pi
                * float(global_step - warmup_steps - hold_base_rate_steps)
                / float(total_steps - warmup_steps - hold_base_rate_steps)
            )
        )
    )
    if hold_base_rate_steps > 0:
        learning_rate = np.where(
            global_step > warmup_steps + hold_base_rate_steps,
            learning_rate,
            learning_rate_base,
        )
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError(
                "learning_rate_base must be larger or equal to " "warmup_learning_rate."
            )
        slope = (learning_rate_base - warmup_learning_rate) / float(warmup_steps)
        warmup_rate = slope * float(global_step) + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate, learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


def get_onehot(indices, labels):
    """Get onehot vectors for all indices."""
    # false positive
    # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
    return tf.one_hot(indices, len(labels), dtype=tf.float32)


def uda_cross_entropy_fn(train_config, num_steps):
    """Calc uda cross entropy loss (copied from google-research repo)."""

    uda_batch_size = (
        train_config.mpl_batch_size * train_config.mpl_unlabeled_batch_size_multiplier
    )
    uda_cross_entropy_loss = tf.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=train_config.mpl_label_smoothing,
        reduction=tf.losses.Reduction.NONE,
    )

    def uda_cross_entropy(all_logits, label_indices, step):
        labels = {"l": get_onehot(label_indices, train_config.labels)}

        masks = {}
        logits = {}
        cross_entropy = {}

        logits["l"], logits["u_ori"], logits["u_aug"] = tf.split(
            all_logits, [train_config.mpl_batch_size, uda_batch_size, uda_batch_size], 0
        )

        # sup loss
        cross_entropy["l"] = uda_cross_entropy_loss(labels["l"], logits["l"])
        probs = tf.nn.softmax(logits["l"], axis=-1)
        correct_probs = tf.reduce_sum(labels["l"] * probs, axis=-1)
        steps_pct = float(step) / float(num_steps)
        labeled_threshold = steps_pct * (
            1.0 - 1.0 / len(train_config.labels)
        ) + 1.0 / len(train_config.labels)
        masks["l"] = tf.less_equal(correct_probs, labeled_threshold)
        masks["l"] = tf.cast(masks["l"], tf.float32)
        masks["l"] = tf.stop_gradient(masks["l"])
        cross_entropy["l"] = tf.reduce_sum(cross_entropy["l"]) / float(
            train_config.mpl_batch_size
        )

        # unsup loss
        labels["u_ori"] = tf.nn.softmax(
            logits["u_ori"] / train_config.uda_label_temperature, axis=-1
        )
        labels["u_ori"] = tf.stop_gradient(labels["u_ori"])

        cross_entropy["u"] = labels["u_ori"] * tf.nn.log_softmax(
            logits["u_aug"], axis=-1
        )
        largest_probs = tf.reduce_max(labels["u_ori"], axis=-1, keepdims=True)
        masks["u"] = tf.greater_equal(largest_probs, train_config.uda_threshold)
        masks["u"] = tf.cast(masks["u"], tf.float32)
        masks["u"] = tf.stop_gradient(masks["u"])
        cross_entropy["u"] = tf.reduce_sum(-cross_entropy["u"] * masks["u"]) / float(
            uda_batch_size
        )
        return logits, labels, masks, cross_entropy

    return uda_cross_entropy
