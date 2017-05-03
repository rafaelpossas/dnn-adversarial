import numpy as np
from adversarial_code.cifar_keras_vgg import VGG
from adversarial_code.utils import Utils
import _pickle as pickle
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

targets = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
utils = Utils()


def get_all_data(class_number, x_test, y_test, x_adv):
    arr = []
    j=0
    for i in range(len(x_test)):
        if np.argmax(y_test[i]) == class_number:
            arr.append(x_adv[j])
            j += 1
        else:
            arr.append(x_test[i])
    return np.array(arr)

def test_by_class(x_adv, model_path=""):
    vgg = VGG(32, 32, 3)

    model = vgg.model(dropout=True)
    model.load_weights(model_path)
    scores = model.evaluate(x_adv, y_test)
    print((scores[1] * 100))
    y_pred = model.predict_classes(x_adv)
    return y_pred

def plot_confusion_matrix(y_test, y_pred):
    rp = classification_report(np.argmax(y_test, axis=1), y_pred)
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(cm, index=targets, columns=targets)

    plt.figure(figsize=(10, 7))
    plt.xticks(rotation=45)
    plt.yticks(rotation=60)
    hm = sns.heatmap(df_cm, annot=True)
    hm.axes.set_title("CIFAR-10 Confusion Matrix")
    hm.axes.set_xlabel("Predicted")
    hm.axes.set_ylabel("True")
    # img_name = str(model_path.split('/')[-1])+".jpg"
    # plt.savefig(img_name)
    return plt



model_path = "../adversarial_code/normalized/balanced_vgg_custom.h5"
X_train, y_train, X_test, y_test = utils.load_cifar10(normalize=True)
with open('../adversarial_code/0.01_class_0_array_adv.pickle', 'rb') as f:
    x_adv = pickle.load(f)
x_adv = get_all_data(0, X_test, y_test, x_adv)
test_by_class(x_adv, model_path)