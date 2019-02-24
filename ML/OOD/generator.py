#!/usr/bin/env python3

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from mysql import SQLConnector

class Generator(object):

    # add kwargs?
    def __init__(self, **kwargs):
        """ Constructor """
        self._defaults()
        self._args(kwargs)   # override defaults with args passed
        self.generator = None
        
        # if no layer values are passed pull from sql
        if self.layer1 == 0 and self.layer2 == 0 and self.layer3 == 0:
            self._pull_layers()
        self._build()

    def _defaults(self):
        """ Sets default variable values """
        self.attack_type = 'neptune'
        self.layer1 = 0
        self.layer2 = 0
        self.layer3 = 0
        self.alpha = 0.2
        self.momentum = 0.8

    def _args(self, kwargs):
        """ kwargs handler """
        for key, value in kwargs.items():
            if key == "attack_type":
                self.attack_type = value
            elif key == "layer1":
                self.layer1 = value
            elif key == "layer2":
                self.layer2 = value
            elif key == "layer3":
                self.layer3 = value
            elif key == "alpha":
                self.alpha = value
            elif key == "momentum":
                self.momentum = value

    def _pull_layers(self):
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
        model.add(Dense(self.layer1, input_dim=41))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Dense(self.layer2))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Dense(self.layer3))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(BatchNormalization(momentum=self.momentum))
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
    gen_args = {
            'attack_type': 'neptune'
    }
    gen = Generator(**gen_args)
    print(gen)


if __name__ == "__main__":
    main()
