#!/usr/bin/env python3

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from mysql import SQLConnector

class Generator(object):

    def __init__(self, attack, layer1 = 0, layer2 = 0, layer3 = 0):
        """ Constructor """
        self.attack_type = attack
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.generator = None
        self._setLayers()
        self._build()

    def _setLayers(self):
        """ Sets layers """

        # pull layers from database
        conn = SQLConnector()
        jsonlist = conn.pull_best(self.attack_type, True)
        json = jsonlist[0]
        layersstr = json['layers']

        # parse ints from string
        comma_index = layersstr.index(",")
        num1 = int(layersstr[:layersstr.index(",")])
        layersstr = layersstr[comma_index + 1:]

        comma_index = layersstr.index(",")
        num2 = int(layersstr[:layersstr.index(",")])
        layersstr = layersstr[comma_index + 1:]

        num3 = int(layersstr)
        

        self.layer1 = num1
        self.layer2 = num2
        self.layer3 = num3


    def _build(self):
        """ Builds the generator """
        model = Sequential()
        model.add(Dense(self.layer1, input_dim=41))  # arbitrarily selected 100 for our input noise vector?
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.layer2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.layer3))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(41, activation='relu'))  # outputs a generated vector of the same size as our data (41)

        noise = Input(shape=(41,))
        attack = model(noise)
        self.generator = Model(noise, attack)


    def get_model(self):
        """ Returns generator model """
        return self.generator

    def __str__(self):
        """ toString """
        return "Layer 1: " + str(self.layer1) +"\nLayer 2: " + str(self.layer2) + "\nLayer 3: " + str(self.layer3) + "\n"


def main():
    """ Auto run main method """
    attack_type = "neptune"
    gen = Generator(attack_type)
    print(gen)


if __name__ == "__main__":
    main()
