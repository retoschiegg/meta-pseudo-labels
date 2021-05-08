"""Meta pseudo label training, finetuning and evaluation of models."""
import os
import logging
import click
import gorilla
import tensorflow as tf

from utils import log_util
from train_config import get_train_config, get_hyperparams
from train import train_finetune, train_mpl, get_exported_ema_file
from evaluate import evaluate_model

LOGGER = logging.getLogger(__name__)

try:
    # optional package only used to log to dagshub
    from dagshub import DAGsHubLogger
    DAGS_LOGGER = DAGsHubLogger()
except ImportError:
    DAGS_LOGGER = None


def evaluate_classifier(classifier_dir, data_dir, train_config, config_name, logging_suffix):
    """Evaluate exported model."""
    acc_top1, acc_top5, _, _ = evaluate_model(
        classifier_dir, data_dir, train_config, config_name
    )

    if logging_suffix and DAGS_LOGGER:
        DAGS_LOGGER.log_metrics(
            {
                "acc-top1" + logging_suffix: acc_top1,
                "acc-top5" + logging_suffix: acc_top5,
            },
        )


def log_training_to_dagshub(train_config):
    """Log every scalar of tensorboard also to dagshub."""
    tensor2value = lambda x: x.numpy() if tf.is_tensor(x) else x

    def scalar(name, data, step=None, description=None):
        original = gorilla.get_original_attribute(tf.summary, "scalar")
        retval = original(name, data, step, description)
        DAGS_LOGGER.log_metrics(
            {name: tensor2value(data)},
            step_num=tensor2value(step),
        )
        return retval

    settings = gorilla.Settings(allow_hit=True)
    patch = gorilla.Patch(tf.summary, "scalar", scalar, settings)
    gorilla.apply(patch)

    DAGS_LOGGER.log_hyperparams(get_hyperparams(train_config))


def train_classifier(
    model_dir,
    data_dir,
    config_name,
    mpl_epochs,
    mpl_batch_size,
    finetune_epochs,
    finetune_batch_size,
    log_every_step,
    continue_mpl,
    log_to_dagshub,
):
    """Train classifier model."""
    mpl_dir = os.path.join(model_dir, "mpl")
    finetune_dir = os.path.join(model_dir, "finetune")

    train_config = get_train_config(config_name)
    train_config.mpl_batch_size = mpl_batch_size
    train_config.mpl_epochs = mpl_epochs
    train_config.finetune_batch_size = finetune_batch_size
    train_config.finetune_epochs = finetune_epochs

    if log_to_dagshub and DAGS_LOGGER:
        log_training_to_dagshub(train_config)

    mpl_model_dir = get_exported_ema_file(mpl_dir)
    if mpl_epochs > 0:
        train_mpl(
            mpl_dir,
            data_dir,
            config_name,
            train_config,
            log_every_step,
            continue_mpl,
        )
        # run eval without finetuning
        evaluate_classifier(
            mpl_model_dir, data_dir, train_config, config_name, "-mpl" if log_to_dagshub else None
        )

    if finetune_epochs > 0:
        finetune_model_dir = train_finetune(
            finetune_dir,
            mpl_model_dir,
            data_dir,
            train_config,
            config_name,
        )
        # run eval without finetuning
        evaluate_classifier(
            finetune_model_dir, data_dir, train_config, config_name, "-finetune" if log_to_dagshub else None
        )

    if log_to_dagshub and DAGS_LOGGER:
        DAGS_LOGGER.save()
        DAGS_LOGGER.close()


@click.command()
@click.option(
    "--model-dir", help="Folder to store training artifacts",
)
@click.option(
    "--data-dir", help="Folder with training dataset",
)
@click.option(
    "--config-name", help="Name of the config to load"
)
@click.option(
    "--mpl-epochs",
    help="Number of epochs to train with meta pseudo label",
    type=int,
    default=0,
)
@click.option(
    "--mpl-batch-size", help="batch size for mpl training", type=int, default=64,
)
@click.option(
    "--finetune-epochs",
    help="Number of epochs to finetune student",
    type=int,
    default=0,
)
@click.option(
    "--finetune-batch-size", help="batch size for finetuning", type=int, default=512,
)
@click.option(
    "--log-every-step", help="Log to tensorboard every x step", type=int, default=250,
)
@click.option("--continue-mpl", is_flag=True)
@click.option("--log-to-dagshub", is_flag=True)
def main(
    model_dir,
    data_dir,
    config_name,
    mpl_epochs,
    mpl_batch_size,
    finetune_epochs,
    finetune_batch_size,
    log_every_step,
    continue_mpl,
    log_to_dagshub,
):
    """Entrypoint."""
    log_util.initialize()
    train_classifier(
        model_dir,
        data_dir,
        config_name,
        mpl_epochs,
        mpl_batch_size,
        finetune_epochs,
        finetune_batch_size,
        log_every_step,
        continue_mpl,
        log_to_dagshub,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
