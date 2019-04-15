from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.advanced_activations import LeakyReLU
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

class Evaluator(object):

    def __init__(self, **kwargs):
        """ Constructor """
        self._defaults()
        self._args(kwargs)
        self._train()

    def _args(self, kwargs):
        for key, value in kwargs.items():
            if key == 'layers':
                self.layers = value
            if key == 'input_shape':
                self.input_shape = value
            if key == 'dropout':
                self.dropout = value
            if key == 'units':
                self.units = value
            if key == 'num_classes':
                self.num_classes = value
            if key == 'layer_count':
                self.layer_count = value
            if key == 'train_data':
                self.train_data = value
            if key == 'train_labels':
                self.train_labels = value
            if key == 'validation_data':
                self.validation_data = value
            if key == 'validation_labels':
                self.validation_labels = value
            if key == 'test_data':
                self.test_data = value
            if key == 'test_labels':
                self.test_labels = value

    def _defaults(self):
        self.evaluator = None
        self.layers = [32, 16, 8]
        self.layer_count = len(self.layers)
        self.units = 32
        self.input_shape = (41,)
        self.dropout = 0.5
        self.num_classes = 1
        self.validation_data = None
        self.train_data = None
        self.performance = 0


    def _build(self):
        """ Builds the discriminator """
        model = Sequential()

        model.add(Dropout(rate=self.dropout, input_shape=self.input_shape))

        for _ in range(self.layer_count - 1):
            model.add(Dense(units=self.units, activation='relu'))
            model.add(Dropout(rate=self.dropout))

        model.add(Dense(units=self.num_classes, activation='sigmoid'))
        model.compile(optimizer=Adam(0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def _train(self):
        self.evaluator = self._build()

        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2)]

        history = self.evaluator.fit(self.train_data,
                                self.train_labels,
                                epochs=5,
                                batch_size=1024,
                                callbacks=callbacks,
                                validation_data=(self.validation_data, self.validation_labels),
                                verbose=2)


        y_pred =self.evaluator.predict(self.test_data)

        self.performance = self.evaluator.evaluate(self.test_data, self.test_labels)


    def get_model(self):
        """ Returns discriminator model """
        return self.evaluator

    #TODO: Fix tostring methods (never used, but not accurate per model)
    def __str__(self):
        """ toString """
        return "Layer 1: 41\nLayer 2: 30\nLayer 3: 15\nLayer 4: 1"


def main():
    """ Auto run main method """
    gen = Evaluator()
    print(gen)


if __name__ == "__main__":
    main()