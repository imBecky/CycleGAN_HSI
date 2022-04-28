import time
import datetime
from utils import *
<<<<<<< HEAD
from CycleGAN import *
=======
from Pix2Pix1D import *
>>>>>>> 7515c7a591a233706ffb267db55eabeec60d0593
import os

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(source_train_batch, target_train_batch,
               generator, discriminator,
               epoch):
    xs, ys = get_data_from_batch(source_train_batch)
    xt, yt = get_data_from_batch(target_train_batch)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(xs, training=True)

        disc_real_output = discriminator([xs, xt], training=True)
        disc_generated_output = discriminator([xs, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, xt)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch*BATCH_SIZE // 1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch*BATCH_SIZE // 1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch*BATCH_SIZE // 1000)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch*BATCH_SIZE // 1000)


def fit(source_train_ds, target_train_ds,
        generator, discriminator,
        epochs):
    for epoch in range(epochs):
        start = time.time()
        for source_batch in source_train_ds.as_numpy_iterator():
            for target_batch in target_train_ds.as_numpy_iterator():
                if (epoch * BATCH_SIZE) % 1000 == 0:
                    display.clear_output(wait=True)
                    if (epoch * BATCH_SIZE) != 0:
                        print(f'Time taken for 1000 epoch is:{time.time() - start:.2f} sec\n')
                    start = time.time()
                    train_epoch(generator, source_batch, target_batch)
                    print(f"Step: {epoch* BATCH_SIZE // 1000}k")
                train_step(source_batch, target_batch,
                           generator, discriminator, epochs)
