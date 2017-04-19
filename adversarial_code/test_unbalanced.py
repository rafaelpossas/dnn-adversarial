
from PIL import Image
import numpy as np
from adversarial_code.adversarial_tf import Adversarial
from adversarial_code.utils import Utils
from adversarial_code.vgg import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.engine import Model
from keras.optimizers import SGD

utils = Utils()
adv_cls = Adversarial()

x_train, y_train, x_test, y_test = utils.load_cifar10(normalize=False)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
last = base_model.get_layer('block3_pool').output
# Add classification layers on top of it
x = Flatten()(last)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(10, activation='sigmoid')(x)
model = Model(base_model.input, pred)
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

model.load_weights("./adversarial_code/cifar10-vgg16_model_alllayers_unbalanced_cls6.h5")

model.evaluate(x_test, y_test)

x_test_adv = []

for ix, img in enumerate(x_test):
    adv, perturbation = adv_cls._fgsm_k_iter(model=model,fake_class_idx=1, epsilon=0.1, n_steps=10,
                                             img=img[np.newaxis, :, :, :], descent=False)
    x_test_adv.append(adv)
    if ix % 10 == 0:
        print("Current Index: "+str(ix))