from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU

class Critic(object):

    def __init__(self, **kwargs):
        """ Constructor """
        self._defaults()
        self._args(kwargs)
        self._build()

    def _args(self, kwargs):
        for key, value in kwargs.items():
            if key == 'layers':
                self.layers = value
            if key == 'optimizer':
                self.optimizer = value
            if key == 'alpha':
                self.alpha = value
            if key == 'dropout':
                self.dropout = value

    def _defaults(self):
        self.alpha = 0.1
        self.critic = None
        self.layers = [32, 16, 8]
        self.optimizer = RMSprop(lr=0.00005)
        self.dropout = 0.3

    def _build(self):
        """ Builds the discriminator """
        model = Sequential()

        # needs 41 for input and needs 1 for output
        print(self.layers)
        print(type(self.layers))
        model.add(Dense(self.layers[0], input_dim=41, activation='relu'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.layers[1]))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.layers[2]))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dense(1))  # outputs a scalar value, different than our discriminator which outputs 0 to 1

        attack = Input(shape=(41,))
        validity = model(attack)

        self.critic = Model(attack, validity)

    def get_model(self):
        """ Returns discriminator model """
        return self.critic

    def __str__(self):
        """ toString """
        return "Layer 1: 41\nLayer 2: 30\nLayer 3: 15\nLayer 4: 1"


def main():
    """ Auto run main method """
    gen = Critic()
    print(gen)


if __name__ == "__main__":
    main()
