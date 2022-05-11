import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa


class ResBlock_up_top(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.bn = layers.BatchNormalization()
        self.unsampling = layers.UpSampling1D()
        self.conv = tfa.layers.SpectralNormalization(layers.Conv1D(filters=output_channels,
                                                                   kernel_size=3,
                                                                   strides=1,
                                                                   padding='same',
                                                                   use_bias=False))
        self.conv_skip = tfa.layers.SpectralNormalization(layers.Conv1D(filters=output_channels,
                                                                        kernel_size=1,
                                                                        strides=1,
                                                                        padding='same',
                                                                        use_bias=False))
        self.relu = layers.LeakyReLU()

    def __call__(self, inputs):
        res = inputs
        res = self.unsampling(res)
        res = self.conv_skip(res)

        x = self.bn(inputs)
        x = self.relu(x)
        x = self.unsampling(x)
        x = self.conv_skip(x)
        x = self.relu(x)
        x = self.conv(x)

        return x + res


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.Conv = tfa.layers.SpectralNormalization(layers.Conv1D(filters=output_channels,
                                                                   kernel_size=3,
                                                                   strides=1,
                                                                   padding='same',
                                                                   use_bias=False))
        self.Relu = layers.LeakyReLU()

    def __call__(self, inputs):
        res = inputs
        x = self.Conv(inputs)
        x = self.Relu(x)
        x = self.Conv(x)
        x = x + res
        return x


class ResBlock_up(tf.keras.layers.Layer):
    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.bn = layers.BatchNormalization()
        self.relu = layers.LeakyReLU()
        self.unsampling = layers.UpSampling1D()
        self.conv = tfa.layers.SpectralNormalization(layers.Conv1D(filters=output_channels,
                                                                   kernel_size=3,
                                                                   strides=1,
                                                                   padding='same',
                                                                   use_bias=False))
        self.conv_skip = tfa.layers.SpectralNormalization(layers.Conv1D(filters=output_channels,
                                                                        kernel_size=1,
                                                                        padding='same',
                                                                        use_bias=False))

    def __call__(self, inputs):
        res = inputs
        res = self.unsampling(res)
        res = self.conv_skip(res)
        x = self.bn(inputs)
        x = self.relu(x)
        x = self.unsampling(x)
        x = self.conv_skip(x)
        x = self.relu(x)
        x = self.conv(x)
        return x + res


class ResBlock_Down(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.relu = layers.LeakyReLU()
        self.conv = tfa.layers.SpectralNormalization(layers.Conv1D(filters=output_channels,
                                                                   kernel_size=3,
                                                                   padding='same',
                                                                   use_bias=False))
        self.conv_skip = tfa.layers.SpectralNormalization(layers.Conv1D(filters=output_channels,
                                                                        kernel_size=1,
                                                                        padding='same',
                                                                        use_bias=False))
        self.avg_pooling = layers.AveragePooling1D(padding='same')

    def __call__(self, inputs):
        res = self.conv_skip(inputs)
        res = self.avg_pooling(res)

        x = self.relu(inputs)
        x = self.conv_skip(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avg_pooling(x)

        return x + res


class Res_Dense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.dense = tfa.layers.SpectralNormalization(layers.Dense(units))
        self.drop = layers.Dropout(0.3)
        self.relu = layers.LeakyReLU()

    def __call__(self, inputs):
        res = inputs
        res = self.dense(res)
        res = self.relu(res)

        x = self.dense(inputs)
        x = self.drop(x)
        x = self.relu(x)

        return x + res


class bottle_neck(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.filters = output_channels
        self.bn = layers.BatchNormalization()
        self.conv = layers.Conv1D(filters=output_channels,
                                  kernel_size=3,
                                  padding='same',
                                  use_bias=False)
        self.conv_skip = layers.Conv1D(filters=output_channels,
                                       kernel_size=1,
                                       strides=1,
                                       padding='same',
                                       use_bias=False)
        self.conv_trans = layers.Conv1D(filters=output_channels,
                                                 kernel_size=1,
                                                 strides=1,
                                                 padding='same',
                                                 use_bias=False)
        self.unsampling = layers.UpSampling1D()
        self.relu = layers.LeakyReLU()

    def __call__(self, inputs):
        if self.filters != inputs.get_shape().as_list()[-1]:
            res = self.conv_skip(inputs)
        else:
            res = inputs
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_trans(x)
        x = self.bn(x)
        x = self.relu(x)
        z = x + res

        return x + res


class bottleneck_rev_s(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.filters = channels
        self.unit = bottle_neck(int(channels//2.0))
        self.dense = layers.Dense(int(channels//2.0))

    def __call__(self, inputs):
        x1, x2 = tf.split(inputs, 2, 2)
        if self.filters != inputs.get_shape().as_list()[-1]:
            x1 = self.dense(x1)
        y1 = x1 + self.unit(x2)
        y2 = x2
        z = tf.concat([y2, y1], axis=2)
        return z


class ResBlock_no_sn(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.dense1 = layers.Dense(units)
        self.dense2 = layers.Dense(units)
        self.dropout = layers.Dropout(0.3)
        self.relu = layers.LeakyReLU()

    def __call__(self, inputs):
        res = self.dense1(inputs)
        res = self.relu(res)

        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x+res
