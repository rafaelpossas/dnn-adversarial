from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping


class VGG(object):

    def __init__(self, img_rows, img_cols, img_channels):
        # input image dimensions
        self.img_rows = img_rows
        self.img_cols = img_cols
        # The CIFAR10 images are RGB.
        self.img_channels = img_channels

    def model(self,  num_classes=10,):

        model = Sequential()

        # Block 1
        model.add(
            Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
                   input_shape=(self.img_rows, self.img_cols, self.img_channels)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        # Classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dense(num_classes, activation='softmax', name='predictions'))

        adadelta = Adadelta(lr=1, rho=0.9, decay=0.00001)
        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=adadelta,
                      metrics=['accuracy'])
        return model

    def fit(self, model, x_train, y_train, x_test, y_test,
            batch_size=32, nb_epochs=200, data_augmentation=True,
            early_stopping_delta=0.0001, early_stopping_patience=10):

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_delta,
                                       patience=early_stopping_patience)
        if not data_augmentation:
            print('Not using data augmentation.')
            # model.fit(x_train, y_train,
            #           batch_size=batch_size,
            #           epochs=epochs,
            #           validation_data=(x_test, y_test),
            #           shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=batch_size),
                                steps_per_epoch=x_train.shape[0] // batch_size,
                                epochs=nb_epochs,
                                validation_data=(x_test, y_test),
                                callbacks=[early_stopping])
            model.save_weights("../model_vgg.hdf5")







