
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
from adversarial_code.cifar_keras_vgg import VGG
from adversarial_code.utils import Utils
# tf.python.control_flow_ops = tf

img_width, img_height = 32, 32

vgg = VGG(32, 32, 3)
utils = Utils()


nb_epoch = 200
nb_classes = 10

X_train, y_train, X_test, y_test = utils.load_cifar10(normalize=False)

nb_train_samples = X_train.shape[0]
nb_validation_samples = X_test.shape[0]

number_of_values_to_remove = 4000


for class_number in range(nb_classes):
    model = vgg.model(dropout=True)
    indexes = utils.get_indexes_to_remove(y_train, number_of_values_to_remove, class_number)

    cur_x_train = np.reshape([img for ix, img in enumerate(X_train) if ix not in indexes], (46000, 32, 32, 3))
    cur_y_train = np.reshape([lbl for ix, lbl in enumerate(y_train) if ix not in indexes], (46000, 10))
    model_name = "unbalanced_"+str(class_number)+"_vgg_custom.h5"
    vgg.fit(model, cur_x_train, cur_y_train, X_test, y_test, model_name=model_name, data_augmentation=False)

model_name = "balanced_" + str(class_number) + "_vgg_custom.h5"
vgg.fit(model, X_train, y_train, X_test, y_test, model_name=model_name)
#
# model = vgg.model(dropout=True)
# model.load_weights("./adversarial_code/unbalanced_2_vgg_custom.h5")
# scores = model.evaluate(X_test, y_test)
# acc = (scores[1] * 100)
# y_pred = model.predict_classes(X_test)
#
# targets = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
#
# from sklearn.metrics import classification_report, confusion_matrix
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# rp = classification_report(np.argmax(y_test, axis=1), y_pred)
# cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
# cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#
# df_cm = pd.DataFrame(cm, index=targets, columns=targets)
#
# plt.figure(figsize=(10, 7))
# plt.xticks(rotation=45)
# plt.yticks(rotation=60)
# hm = sns.heatmap(df_cm, annot=True)
# hm.axes.set_title("CIFAR-10 Confusion Matrix")
# hm.axes.set_xlabel("Predicted")
# hm.axes.set_ylabel("True")
