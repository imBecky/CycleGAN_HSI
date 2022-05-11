import tensorflow as tf
import tensorflow.keras.layers as layers
from param import *
import numpy as np
from utils import *
from matplotlib import pyplot as plt
from IPython import display


def down_sample(filters, size, apply_bn=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv1D(filters, size, strides=1, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    if apply_bn:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def up_sample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv1D(filters, size, strides=1,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def make_generator():
    inputs = tf.keras.layers.Input(shape=[72, 1])

    down_stack = [
        down_sample(64, 4, apply_bn=False),  # (batch_size, 128, 128, 64)
        down_sample(128, 4),  # (batch_size, 64, 64, 128)
        down_sample(256, 4),  # (batch_size, 32, 32, 256)
        down_sample(512, 4),  # (batch_size, 16, 16, 512)
        down_sample(512, 4),  # (batch_size, 8, 8, 512)
        down_sample(512, 4),  # (batch_size, 4, 4, 512)
        down_sample(512, 4),  # (batch_size, 2, 2, 512)
        down_sample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        up_sample(512, 4, apply_dropout=True),
        up_sample(512, 4, apply_dropout=True),
        up_sample(512, 4, apply_dropout=True),
        up_sample(512, 4),
        up_sample(256, 4),
        up_sample(128, 4),
        up_sample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv1D(1, 4,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def make_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[72, 1], name='input_spectral')
    tar = tf.keras.layers.Input(shape=[72, 1], name='target_spectral')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = down_sample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = down_sample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = down_sample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding1D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv1D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding1D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv1D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    # plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.plot(np.arange(72), display_list[i])
    plt.show()


def train_epoch(model, source_batch, target_batch):
    xs, ys = get_data_from_batch(source_batch)
    xt, yt = get_data_from_batch(target_batch)
    for _ in range(BATCH_SIZE):
        generate_images(model, xs, xt)
