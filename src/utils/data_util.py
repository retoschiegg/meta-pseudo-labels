"""This module provides loaders for dataset such as cifar."""
import os
import tensorflow as tf


def _cifar10_parser_fn(image_width, image_height, with_label):
    """Get Cifar10 sample parser function."""

    def parser(value):
        value = tf.io.decode_raw(value, tf.uint8)
        image = tf.reshape(value[1:], [3, 32, 32])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.transpose(image, [1, 2, 0])
        if image_width != 32 or image_height != 32:
            image = tf.image.resize([image], [image_height, image_width])[0]
        image.set_shape([image_height, image_width, 3])
        if with_label:
            return image, tf.cast(value[0], tf.int32)
        return image

    return parser


def _get_cifar10_files(data_dir):
    """Get binary files in folder."""
    return [os.path.join(data_dir, f"data_batch_{i}.bin") for i in range(1, 6)]


def get_cifar10_labeled_train_ds(data_dir, image_width, image_height):
    """Load labeled cifar10 dataset for training."""
    num_samples = 4000
    filenames = _get_cifar10_files(data_dir)
    record_bytes = 1 + (3 * 32 * 32)
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    dataset = dataset.take(num_samples)
    dataset = dataset.map(_cifar10_parser_fn(image_width, image_height, True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset, num_samples


def get_cifar10_unlabeled_train_ds(data_dir, image_width, image_height):
    """Load unlabeled cifar10 dataset for training."""
    filenames = _get_cifar10_files(data_dir)
    record_bytes = 1 + (3 * 32 * 32)
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    dataset = dataset.map(_cifar10_parser_fn(image_width, image_height, False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def get_cifar10_val_ds(data_path, image_width, image_height):
    """Load cifar10 dataset for validation."""
    filenames = [os.path.join(data_path, "test_batch.bin")]
    record_bytes = 1 + (3 * 32 * 32)
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    dataset = dataset.map(_cifar10_parser_fn(image_width, image_height, True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
