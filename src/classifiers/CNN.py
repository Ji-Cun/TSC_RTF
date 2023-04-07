# -*- coding: utf-8 -*-
from tensorflow import keras
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from src.classifiers.TimeHistory import TimeHistory


class CNN:
    def __init__(self, input_shape, nb_classes):
        self.model = self.build_model(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)
        conv1 = keras.layers.Conv1D(filters=64, kernel_size=8, padding='same', activation='relu')(input_layer)
        conv1 = keras.layers.AveragePooling1D()(conv1)

        conv2 = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(conv1)
        conv2 = keras.layers.AveragePooling1D()(conv2)

        conv3 = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv2)
        conv3 = keras.layers.AveragePooling1D()(conv3)

        flatten_layer = keras.layers.Flatten()(conv3)

        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        # optimizer = keras.optimizers.Adam()
        optimizer = keras.optimizers.Nadam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def fit(self, x_train, x_test, Y_train, Y_test, nb_epochs=2000):
        batch_size = int(min(x_train.shape[0] / 10, 16))

        es = EarlyStopping(monitor='loss', min_delta=0.0001, patience=50)
        rp = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50, min_lr=0.0001)
        hist = self.model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=0, validation_data=(x_test, Y_test), callbacks=[es, rp])
        # Print the testing results which has the lowest training loss.
        # log = pd.DataFrame(hist.history)
        # print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_accuracy'])
        log = pd.DataFrame(hist.history)
        acc = log.iloc[-1]['val_accuracy']
        return acc

    def predict(self, x_test):
        y_pred = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
