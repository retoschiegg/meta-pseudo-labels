"""Model building for classifier."""
import numpy as np
import tensorflow as tf

from . import wideresnet


def _get_wideresnet(image_width, image_height, image_channels, dropout_rate, labels):
    """Get wideresnet architecture for student/teacher."""
    img_input = tf.keras.layers.Input(shape=(image_height, image_width, image_channels))
    output = wideresnet.Wrn28k()(img_input)

    if dropout_rate > 0:
        output = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(output)

    init_range = 1.0 / np.sqrt(len(labels))
    kernel_initializer = tf.keras.initializers.RandomUniform(
        minval=-init_range, maxval=init_range
    )
    output = tf.keras.layers.Dense(
        len(labels), kernel_initializer=kernel_initializer, name="predictions",
    )(output)

    return tf.keras.models.Model(img_input, output)


def get_teacher_model(train_config):
    """Get teacher model."""
    return _get_wideresnet(
        train_config.image_width,
        train_config.image_height,
        train_config.image_channels,
        train_config.teacher_dropout_rate,
        train_config.labels,
    )


def get_student_model(train_config, dropout_rate=None):
    """Get student model."""
    return _get_wideresnet(
        train_config.image_width,
        train_config.image_height,
        train_config.image_channels,
        dropout_rate or train_config.student_dropout_rate,
        train_config.labels,
    )


def get_finetune_model(train_config, exported_model_dir):
    """Load exported student model and return model to finetune."""
    loaded_model = tf.keras.models.load_model(exported_model_dir)
    model = get_student_model(train_config, train_config.finetune_dropout_rate)
    model.set_weights(loaded_model.get_weights())
    for i, layer in enumerate(model.layers):
        if i < len(model.layers) - 2:
            layer.trainable = False
    return model
