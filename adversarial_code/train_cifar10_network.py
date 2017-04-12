import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.datasets import cifar10
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
from adversarial_code.vgg import VGG16

# tf.python.control_flow_ops = tf

img_width, img_height = 32, 32
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

nb_epoch = 50
nb_classes = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

nb_train_samples = X_train.shape[0]
nb_validation_samples = X_test.shape[0]

# Extract the last layer from third block of vgg16 model
last = base_model.get_layer('block3_pool').output
# Add classification layers on top of it
x = Flatten()(last)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(10, activation='sigmoid')(x)

model = Model(base_model.input, pred)

# set the base model's layers to non-trainable
# uncomment next two lines if you don't want to
# train the base model
# for layer in base_model.layers:
#     layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

model.summary()

number_of_values_to_remove = 2500
class_number = 6
numbers = [x.argmax() for x in Y_train]
print(np.unique(numbers, return_counts=True))

indexes = []

for ix in range(len(numbers)):
    if number_of_values_to_remove > 0:
        if numbers[ix] == class_number:
            indexes.append(ix)
            number_of_values_to_remove = number_of_values_to_remove - 1

X_train = np.reshape([img for ix, img in enumerate(X_train) if ix not in indexes], (47500, 32, 32, 3))
Y_train = np.reshape([lbl for ix, lbl in enumerate(Y_train) if ix not in indexes], (47500, 10))

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(X_test, Y_test, batch_size=32)

# callback for tensorboard integration
tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[tb])

# save the model
model.save('cifar10-vgg16_model_alllayers_unbalanced.h5')

# from adversarial_code.cifar_keras_vgg import VGG
# from adversarial_code.utils import Utils
# from keras.optimizers import SGD, Adadelta
#
# if __name__ == '__main__':
#     network = VGG(img_rows=32, img_cols=32, img_channels=3)
#     utils = Utils()
#
#     model = network.model(10, True)
#     x_train, y_train, x_test, y_test = utils.load_cifar10()
#
#     adadelta = Adadelta(lr=1, rho=0.9, decay=0.00001)
#     sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     # Let's train the model using RMSprop
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=sgd,
#                   metrics=['accuracy'])
#     print("[INFO] starting training...")
#
#     model.fit(x_train, y_train, batch_size=32, epochs=40)
#
#     # show the accuracy on the testing set
#     (loss, accuracy) = model.evaluate(x_test, y_test,
#                                       batch_size=32, verbose=1)
#     print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
#
#
