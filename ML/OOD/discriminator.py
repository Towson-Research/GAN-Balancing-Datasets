#!/usr/bin/env python3

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input

class Discriminator(object):

    def __init__(self, layers=[(30, 'relu'), (15, 'relu')]):
        """ Constructor """
        self.discriminator = None
        # list of tuples
        self.layers = layers
        self._build()


    def _build(self):
        """ Builds the discriminator """
        model = Sequential()

        # needs 41 for input and needs 1 for output
        model.add(Dense(41, input_dim=41, activation='relu'))
        # discriminator takes 41 values from our dataset
        for lay in self.layers:
            model.add(Dense(lay[0], activation=lay[1]))
        model.add(Dense(1, activation='sigmoid'))  # outputs 0 to 1, 1 being read and 0 being fake

        print("Discriminator: (41, 'relu'), " + str(self.layers) + ", (1, 'sigmoid')" )
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
