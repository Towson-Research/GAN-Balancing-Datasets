#!/usr/bin/env python3
# Matt

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


class Generator(object):

    def __init__(self, attack):
        """ Constructor """
        self.attack_type = attack
        self.layer1 = 0
        self.layer2 = 0
        self.layer3 = 0
        self.generator = None
        self.__setLayers__()
        self.__build__()

    def __setLayers__(self):
        if(self.attack_type == "neptune"):
            self.layer1 = 10
            self.layer2 = 20
            self.layer3 = 30

    def __build__(self):
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

    
    def get(self):
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
