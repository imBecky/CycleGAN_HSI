import scipy.io as sio
from train import *
from utils import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NEW_DATA_PATH = './new data'
"""load data"""
source_dict = sio.loadmat(os.path.join(NEW_DATA_PATH, 'Source.mat'))
source_train_ds, source_test_ds = gen_dataset_from_dict(source_dict)
target_dict = sio.loadmat(os.path.join(NEW_DATA_PATH, 'Target.mat'))
target_train_ds, target_test_ds, target_val_ds = gen_dataset_from_dict(target_dict, Val=True)

plt.plot(np.arange(72), source_train_ds.as_numpy_iterator().next()['data'][0, :, 0])
plt.title('Source')
plt.show()

generator = make_generator()
discriminator = make_discriminator()
classifier = make_classifier_model()
fit(source_train_ds, target_train_ds,
    generator, discriminator, classifier, EPOCHS)
val_accuracy = tf.keras.metrics.CategoricalAccuracy('val_accuracy')
acc = []
for val_batch in target_val_ds.as_numpy_iterator():
    x, y = get_data_from_batch(val_batch)
    prediction = classifier(x, training=False)
    val_accuracy(y, prediction)
    acc.append(val_accuracy.result()*100)
    template = 'average Accuracy: {:.2f}%'
    print(template.format(np.average(acc)))
