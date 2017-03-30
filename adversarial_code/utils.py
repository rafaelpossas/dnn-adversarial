import numpy as np
import keras
from keras.datasets import cifar10

class Utils(object):

    def get_samples_per_class(self, y):
        samples = [smp.argmax() for smp in y]
        return np.unique(samples, return_counts=True)

    def get_indexes_to_remove(self, y, values_to_remove, class_number):
        class_labels = [label.argmax() for label in y]
        indexes = []
        for ix in range(len(class_labels)):
            if values_to_remove > 0:
                if class_labels[ix] == class_number:
                    indexes.append(ix)
                    values_to_remove -= 1
        return indexes

    def load_cifar10(self):
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        return x_train, y_train, x_test, y_test

