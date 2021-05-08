"""WideResNet Model."""
import tensorflow as tf
from tensorflow.keras import layers


class Wrn28k(layers.Layer):
    """WideResnet-28k."""

    # block architecture
    # pylint: disable=too-many-instance-attributes

    def __init__(self, multiplier=2, **kwargs):
        super().__init__(**kwargs)
        size = (
            [16, 135, 135 * 2, 135 * 4]
            if multiplier == 135
            else [16, 16 * multiplier, 32 * multiplier, 64 * multiplier]
        )

        self.conv1 = conv2d(size[0])
        self.bn1 = batch_norm()

        self.block1 = NetworkBlock(size[1], 1)
        self.block2 = NetworkBlock(size[1], 1)
        self.block3 = NetworkBlock(size[1], 1)
        self.block4 = NetworkBlock(size[1], 1)

        self.block5 = NetworkBlock(size[2], 2)
        self.block6 = NetworkBlock(size[2], 1)
        self.block7 = NetworkBlock(size[2], 1)
        self.block8 = NetworkBlock(size[2], 1)

        self.block9 = NetworkBlock(size[3], 2)
        self.block10 = NetworkBlock(size[3], 1)
        self.block11 = NetworkBlock(size[3], 1)
        self.block12 = NetworkBlock(size[3], 1)

    def call(self, inputs, **kwargs):
        output = self.conv1(inputs)

        output = self.block1(output, **kwargs)
        output = self.block2(output, **kwargs)
        output = self.block3(output, **kwargs)
        output = self.block4(output, **kwargs)
        output = self.block5(output, **kwargs)
        output = self.block6(output, **kwargs)
        output = self.block7(output, **kwargs)
        output = self.block8(output, **kwargs)
        output = self.block9(output, **kwargs)
        output = self.block10(output, **kwargs)
        output = self.block11(output, **kwargs)
        output = self.block12(output, **kwargs)

        output = self.bn1(output, **kwargs)
        output = relu(output)
        output = tf.reduce_mean(output, axis=[1, 2], name="global_avg_pool")
        return output


def batch_norm(momentum=0.01, epsilon=1e-3):
    """Get batch normalization layer."""
    return layers.BatchNormalization(momentum=momentum, epsilon=epsilon)


def conv2d(filters, kernel_size=3, stride=1, padding="same", use_bias=False):
    """Get conv2d layer."""
    return layers.Conv2D(
        filters,
        kernel_size,
        strides=(stride, stride),
        padding=padding,
        use_bias=use_bias,
    )


def relu(inputs, leaky=0.2, name="relu"):
    """Leaky ReLU."""
    return tf.nn.leaky_relu(inputs, alpha=leaky, name=name)


class NetworkBlock(layers.Layer):
    """WideResNet Block Layer."""

    def __init__(self, filters, stride, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride

        self.bn1 = batch_norm()
        self.conv1 = conv2d(filters, stride=stride)
        self.bn2 = batch_norm()
        self.conv2 = conv2d(filters)
        self.conv3 = None

    def call(self, inputs, **kwargs):
        num_inp_filters = inputs.shape[-1]
        residual = inputs

        output = self.bn1(inputs, **kwargs)
        if self.stride == 2 or num_inp_filters != self.filters:
            residual = output

        output = relu(output)
        output = self.conv1(output, **kwargs)
        output = self.bn2(output, **kwargs)
        output = relu(output)
        output = self.conv2(output, **kwargs)

        if self.stride == 2 or num_inp_filters != self.filters:
            residual = relu(residual)
            if self.conv3 is None:
                self.conv3 = conv2d(self.filters, kernel_size=1, stride=self.stride)
            residual = self.conv3(residual, **kwargs)

        return output + residual
