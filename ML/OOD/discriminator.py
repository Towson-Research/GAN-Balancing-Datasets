#!/usr/bin/env python3

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.advanced_activations import LeakyReLU

class Discriminator(object):

    def __init__(self, **kwargs):
        """ Constructor """
        self._defaults()
        self._args(kwargs)
        self._build()

    def _args(self, kwargs):
        for key, value in kwargs.items():
            if key == 'layers':
                self.layers = value
            if key == 'alpha':
                self.alpha = value
            if key == 'dropout':
                self.dropout = value

    def _defaults(self):
        self.discriminator = None
        self.layers = [32, 16, 8,1]
        self.alpha = 0.1
        self.dropout = 0.3

    def _build(self):
        """ Builds the discriminator """
        model = Sequential()

        # needs 41 for input and needs 1 for output
        model.add(Dense(self.layers[0], input_dim=41, activation='relu'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.layers[1]))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.layers[2]))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.layers[3], activation='sigmoid'))  # outputs 0 to 1, 1 being read and 0 being fake

        attack = Input(shape=(41,))
        validity = model(attack)

        self.discriminator = Model(attack, validity)

    def get_model(self):
        """ Returns discriminator model """
        return self.discriminator

    def __str__(self):
        """ toString """
        return "Layer 1: 41\nLayer 2: 30\nLayer 3: 15\nLayer 4: 1"


def main():
    """ Auto run main method """
    gen = Discriminator()
    print(gen)


if __name__ == "__main__":
    main()
