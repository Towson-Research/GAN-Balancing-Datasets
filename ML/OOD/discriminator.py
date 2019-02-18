#!/usr/bin/env python3
# Matt

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input

class Discriminator(object):

    def __init__(self):
        """ Constructor """
        self.discriminator = None
        self._build()


    def _build(self):
        """ Builds the discriminator """
        model = Sequential()
        model.add(Dense(41, input_dim=41, activation='relu'))  # discriminator takes 41 values from our dataset
        model.add(Dense(30, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # outputs 0 to 1, 1 being read and 0 being fake

        attack = Input(shape=(41,))
        validity = model(attack)

        self.discriminator = Model(attack, validity)

    def get(self):
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
