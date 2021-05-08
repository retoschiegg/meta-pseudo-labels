"""Evaluation of an saved model."""
import logging
import click
import numpy as np
import sklearn.metrics
import tensorflow as tf

from utils import log_util
from train_config import get_train_config
from dataset import get_val_ds

LOGGER = logging.getLogger(__name__)


def _get_metrics(val_gt_indices, val_predictions, labels):
    """Calculate metrics."""
    val_predictions_indices = np.argmax(val_predictions, axis=1)

    acc_top1 = sklearn.metrics.top_k_accuracy_score(val_gt_indices, val_predictions, k=1)
    acc_top5 = sklearn.metrics.top_k_accuracy_score(val_gt_indices, val_predictions, k=5)
    conf_matrix = sklearn.metrics.confusion_matrix(
        val_gt_indices, val_predictions_indices
    )
    metrics_report = sklearn.metrics.classification_report(
        val_gt_indices,
        val_predictions_indices,
        labels=list(range(len(labels))),
        target_names=labels,
    )
    return acc_top1 * 100, acc_top5 * 100, conf_matrix, metrics_report


def evaluate_model(
    model_dir, data_dir, train_config, config_name
):
    """Evaluate model."""
    val_dataset = get_val_ds(train_config, data_dir, train_config.finetune_batch_size, config_name)
    classifier = tf.keras.models.load_model(model_dir)

    val_gt_indices = np.empty((0,))
    val_predictions = np.empty((0, len(train_config.labels)))
    for image, label in val_dataset:
        predictions = tf.nn.softmax(classifier(image))
        val_gt_indices = np.concatenate((val_gt_indices, label))
        val_predictions = np.concatenate((val_predictions, predictions))

    val_gt_indices = np.array(val_gt_indices, dtype=np.int32)

    acc_top1, acc_top5, conf_matrix, metrics_report = _get_metrics(
        val_gt_indices, val_predictions, train_config.labels
    )

    LOGGER.info("Top1 Accuracy: %s", acc_top1)
    LOGGER.info("Top5 Accuracy: %s", acc_top5)
    LOGGER.info("Confusion Matrix: \n%s", conf_matrix)
    LOGGER.info("Metrics: \n%s", metrics_report)

    return acc_top1, acc_top5, conf_matrix, metrics_report


@click.command()
@click.option(
    "--data-dir", help="Folder with training dataset",
)
@click.option("--saved-model-dir", help="Folder with tf saved model")
@click.option(
    "--config-name", help="Name of the config to load"
)
@click.option("--batch-size", default=64)
def main(
    data_dir, saved_model_dir, config_name, batch_size
):
    """Entrypoint."""
    log_util.initialize()

    train_config = get_train_config(config_name)
    train_config.finetune_batch_size = batch_size
    evaluate_model(saved_model_dir, data_dir, train_config, config_name)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
