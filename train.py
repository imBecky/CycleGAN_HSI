import time
import datetime
from utils import *
from CycleGAN import *
import os

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def generator_train_step(source_train_batch, target_train_batch,
                         generator, discriminator,
                         epoch):
    xs, ys = get_data_from_batch(source_train_batch)
    xt, yt = get_data_from_batch(target_train_batch)
    if xs.shape == xt.shape:
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            print('aaa')
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
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def classify_train_step(generator, classifier,
                        source_batch, target_batch, epoch):
    xs, ys = get_data_from_batch(source_batch)
    xt, yt = get_data_from_batch(target_batch)
    with tf.GradientTape() as tape:
        prediction = classifier(xt, training=True)
        classify_loss = classifier_loss(prediction, yt)

        generated_target = generator(xs, training=False)
        prediction_fake = classifier(generated_target, training=True)
        classify_loss += classifier_loss(prediction_fake, ys)

        classify_gradient = tape.gradient(classify_loss, classifier.trainable_variables)
        classifier_optimizer.apply_gradients(zip(classify_gradient,
                                                 classifier.trainable_variables))
        train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', classify_loss, step=epoch)
            tf.summary.scalar('gen_gan_acc', train_accuracy.result(), step=epoch)


def fit(source_train_ds, target_train_ds,
        generator, discriminator,
        classifier,
        epochs):
    for epoch in range(epochs):
        start = time.time()
        # early stopping is not recommended in GAN!!!
        for source_batch in source_train_ds.as_numpy_iterator():
            for target_batch in target_train_ds.as_numpy_iterator():
                generator_train_step(source_batch, target_batch,
                                     generator, discriminator, epochs)
        print('Time for epoch {} is {:.2f} sec'.format(epoch+1, time.time()-start))
    patience = PATIENCE
    wait = 0
    best = 0
    for epoch in range(epochs):
        start = time.time()
        for source_batch in source_train_ds.as_numpy_iterator():
            for target_batch in target_train_ds.as_numpy_iterator():
                classify_train_step(generator, classifier,
                                    source_batch, target_batch, epoch)
        acc = calculate_acc(target_train_ds, classifier, epoch)
        print(acc)
        print(f'Time taken for epoch {epoch + 1} is:{time.time() - start:.2f} sec\n')
        if epoch > EPOCHS * 0.5:
            wait += 1
            if acc > best:
                best = acc
                wait = 0
            if wait >= patience:
                break
