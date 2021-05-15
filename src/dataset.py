"""Functions to create tf datasets for training."""
import numpy as np
import tensorflow as tf

import autoaugment
from utils import data_util


def get_labeled_train_ds(train_config, data_dir, batch_size, name):
    """Return tf dataset for training."""
    datasets = {
        "cifar10": lambda: data_util.get_cifar10_labeled_train_ds(data_dir, train_config.image_width, train_config.image_height),
    }
    preprocess_image = _preprocess_image_fn(name)

    dataset, num_samples = datasets[name]()
    dataset = dataset.map(
        lambda img, lbl: (preprocess_image(img), lbl),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    num_steps = (num_samples // batch_size) + 1
    return dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE), num_steps


def get_val_ds(train_config, data_dir, batch_size, name):
    """Return tf dataset for training."""
    datasets = {
        "cifar10": lambda: data_util.get_cifar10_val_ds(data_dir, train_config.image_width, train_config.image_height),
    }
    preprocess_image = _preprocess_image_fn(name, False)

    dataset = datasets[name]()
    dataset = dataset.map(lambda img, lbl: (preprocess_image(img), lbl))
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def get_unlabeled_train_ds(train_config, data_dir, batch_size, name):
    """Return tf dataset for training."""
    datasets = {
        "cifar10": lambda: data_util.get_cifar10_unlabeled_train_ds(data_dir, train_config.image_width, train_config.image_height),
    }
    augment_image = _augment_image_fn(train_config, name)
    preprocess_image = _preprocess_image_fn(name)

    def preprocess(org_image):
        aug_image = augment_image(org_image)
        aug_image = preprocess_image(aug_image)
        org_image = preprocess_image(org_image)
        return org_image, aug_image

    dataset = datasets[name]()
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def _preprocess_image_fn(name, with_augment=True):
    """Preprocess for training."""
    cifar_mean = np.array([0.491400, 0.482158, 0.4465309]).reshape([1, 1, 3])
    cifar_stdev = np.array([0.247032, 0.243485, 0.26159]).reshape([1, 1, 3])

    def preprocess_cifar10(image):
        if with_augment:
            image = _augment_flip_and_jitter(image, replace_value=0.5)
        return (image - cifar_mean) / cifar_stdev

    datasets = {
        "cifar10": preprocess_cifar10,
    }
    return datasets[name]


def _augment_image_fn(train_config, name):
    """RandAugment image for training."""

    rand_augment = autoaugment.RandAugment(
        train_config.mpl_num_augments,
        train_config.mpl_augment_magnitude,
        train_config.mpl_augment_cutout_const,
        train_config.mpl_augment_translate_const,
        train_config.mpl_augment_ops,
    )

    def augment_cifar10(image):
        image_uint = tf.image.convert_image_dtype(image, tf.uint8)
        image_uint = _augment_flip_and_jitter(image_uint, replace_value=128)
        image_uint = rand_augment.distort(image_uint)
        image_uint = autoaugment.cutout(
            image_uint, pad_size=train_config.image_height // 4, replace=128
        )
        image_float = tf.image.convert_image_dtype(image_uint, tf.float32)
        return image_float

    datasets = {
        "cifar10": augment_cifar10,
    }

    return datasets[name]


def _augment_flip_and_jitter(x, replace_value=0):
    """Flip left/right and jitter."""
    x = tf.image.random_flip_left_right(x)
    image_size = min([x.shape[0], x.shape[1]])
    pad_size = image_size // 8
    x = tf.pad(
        x,
        paddings=[[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
        constant_values=replace_value,
    )
    x = tf.image.random_crop(x, [image_size, image_size, 3])
    x.set_shape([image_size, image_size, 3])
    return x
