# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:16:11 2019
@author: viryl
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from param import *

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def gen_dataset_from_dict(file_dict, Val=False):
    data = file_dict['data']
    data = np.transpose(data, (0, 2, 1))
    label = file_dict['gt']
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=TEST_FRAC, random_state=42)
    if Val:
        data_test, data_val, label_test, label_val = train_test_split(data_test, label_test, test_size=VAL_FRAC,
                                                                      random_state=43)
    data_train = tf.data.Dataset.from_tensor_slices(data_train)
    data_test = tf.data.Dataset.from_tensor_slices(data_test)
    label_train = tf.data.Dataset.from_tensor_slices(label_train)
    label_test = tf.data.Dataset.from_tensor_slices(label_test)
    if Val:
        data_val = tf.data.Dataset.from_tensor_slices(data_val)
        label_val = tf.data.Dataset.from_tensor_slices(label_val)
        val_ds = tf.data.Dataset.zip((data_val, label_val))
        val_ds = val_ds.map(lambda x, y: {'data': x, 'label': y}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    train_ds = tf.data.Dataset.zip((data_train, label_train))
    test_ds = tf.data.Dataset.zip((data_test, label_test))

    train_ds = train_ds.map(lambda x, y: {'data': x, 'label': y}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.map(lambda x, y: {'data': x, 'label': y}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    if Val:
        return train_ds, test_ds, val_ds
    else:
        return train_ds, test_ds


def generate_and_save_Images(model, epoch, test_input):
    """Notice `training` is set to False.
       This is so all layers run in inference mode (batch norm)."""
    """To-do: reshape the curves as they were normalized"""
    prediction = model(test_input, training=False)
    plt.plot(np.arange(72), prediction[0, :, 0])
    plt.savefig('./pics/image_at_{:04d}_epoch.png'.format(epoch))
    plt.show()


def get_data_from_batch(batches):
    return batches['data'], batches['label']


def calculate_acc(target_test_ds,
                  classifier,
                  epoch):
    target_batch = target_test_ds.shuffle(BUFFER_SIZE).as_numpy_iterator().next()
    target_data, target_label = get_data_from_batch(target_batch)
    prediction_t = classifier(target_data, training=False)
    accuracy_t = tf.metrics.Accuracy()
    accuracy_t.update_state(y_true=target_label,
                            y_pred=prediction_t)
    print('Target accuracy for epoch {} is'.format(epoch + 1),
          '{}%'.format(accuracy_t.result().numpy() * 100))


def plot_acc_loss(acc, gen_loss, disc_loss, cls_loss,
                  generator_loss, discriminator_loss, classifier_loss,
                  source_test_ds, target_test_ds,
                  generator, discriminator, classifier,
                  epoch):
    g_loss, d_loss, c_loss, a = [], [], [], []
    for source_test_batch in source_test_ds.as_numpy_iterator():
        for target_test_batch in target_test_ds.as_numpy_iterator():
            X_s, Y_s = get_data_from_batch(source_test_batch)
            X_t, Y_t = get_data_from_batch(target_test_batch)
            generated_target = generator(X_s, training=False)
            real_decision = discriminator(X_t, training=False)
            fake_decision = discriminator(generated_target, training=False)
            prediction = classifier(X_t, training=False)
            accuracy_t = tf.metrics.Accuracy()
            accuracy_t.update_state(y_true=Y_t,
                                    y_pred=prediction)
            a.append(accuracy_t.result().numpy())
            c_loss.append(classifier_loss(prediction, Y_t).numpy())
            g_loss.append(generator_loss(fake_decision).numpy())
            d_loss.append(discriminator_loss(real_decision, fake_decision).numpy())
    a = np.average(a)
    acc.append(a)
    cls_loss.append(np.average(c_loss))
    gen_loss.append(np.average(g_loss))
    disc_loss.append(np.average(d_loss))
    epochs_range = range(epoch+1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, gen_loss, label='Generator_loss')
    plt.plot(epochs_range, disc_loss, label='Discriminator_loss')
    plt.plot(epochs_range, cls_loss, label='Classifier_loss')
    plt.legend(loc='lower right')
    plt.title('Generator and discriminator loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, label='Test accuracy')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    return acc, gen_loss, disc_loss, cls_loss
