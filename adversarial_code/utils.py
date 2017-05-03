import numpy as np
import keras
from keras.datasets import cifar10


class Utils(object):

    targets = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    def get_indexes_to_remove(self, y, values_to_remove, class_number, invert=False):
        class_labels = [label.argmax() for label in y]
        indexes = []
        values_to_remove = values_to_remove if not invert else (9*values_to_remove)
        for ix in range(len(class_labels)):
            if values_to_remove > 0:
                if not invert and class_labels[ix] == class_number:
                    indexes.append(ix)
                    values_to_remove -= 1
                if invert and class_labels[ix] != class_number:
                    indexes.append(ix)
                    values_to_remove -= 1
        return indexes

    def get_samples_by_class(self, x, y, class_index, num_samples=1000):
        samples = x[self.get_indexes_to_remove(y, num_samples, class_index)]
        return samples

    def get_adversaries(self, model, x, fake_class_idx, epsilon=0.02, n_steps=1):
        from adversarial_code.adversarial_tf import Adversarial
        adv_cls = Adversarial()
        adversaries = []
        for cls in x:
            adv, perturbation = adv_cls._fgsm_k_iter(model=model, fake_class_idx=fake_class_idx,
                                                     epsilon=epsilon, n_steps=n_steps,
                                                     img=cls[np.newaxis, :, :, :])
            adversaries.append(adv)
        adversaries = np.reshape(adversaries, (len(x), 32, 32, 3))
        return adversaries

    def format_results_cifar10(self, pred):

        import json

        formatted_result = {}
        for ix, tgt in enumerate(self.targets):
            formatted_result[tgt] = pred[0][ix] * 100
        print(json.dumps(formatted_result, indent=1, sort_keys=True))

        return formatted_result

    def load_cifar10(self, normalize=True):
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        if normalize:
            x_train = x_train.astype("float") / 255.0
            x_test = x_test.astype("float") / 255.0

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        return x_train, y_train, x_test, y_test

    def get_count_by_class(self, preds, categorical=True):
        if categorical:
            preds = [pred.argmax() for pred in preds]
        return np.unique(preds, return_counts=True)
