from collections import defaultdict
import numpy as np
import pandas as pd

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.advanced_activations import LeakyReLU

import signal
import sys

from discriminator import Discriminator
from generator import Generator
from mysql import SQLConnector
from utilities import Utilities as util

try:
    import cPickle as pickle
except:
    import pickle

class NDGAN(object):

    def __init__(self, **kwargs):
        """ Constructor """
        self._defaults()
        self._args(kwargs)  # override defaults with args passed
        self.setup()
        self.build()


    def _defaults(self):
        """ Sets default variable values """
        # TODO: Change params as appropriate for this GAN mdoel
        self.df_dim = 41
        self.gf_dim = 41
        self.attack_type = None
        self.discriminator = None
        self.generator = None
        self.gan = None

        # saved_states can be used to save states of a GAN, say
        # 5 of them so that the best can be saved when breaking out.
        self.saved_states = []
        self.confusion_matrix = None
        self.classification_report = None

        self.optimizer_learning_rate = 0.001
        self.optimizer = Adam(self.optimizer_learning_rate)

        self.max_epochs = 20000
        self.batch_size = 255
        self.sample_size = 500

        self.valid = None
        self.fake = None
        self.X_train = None

        self.generator_alpha = 0.1
        self.generator_momentum = 0.0
        self.generator_layers = [8, 16, 32]
        self.r_alpha = 0.2

        self.confusion_matrix = None
        self.classification_report = None

        self.save_file = None
        self.alpha = 0.1
        self.dropout = 0.3


    def _args(self, kwargs):
        """ kwargs handler """
        for key, value in kwargs.items():
            if key == 'attack_type':
                self.attack_type = value
            elif key == 'max_epochs':
                self.max_epochs = value
            elif key == 'batch_size':
                self.batch_size = value
            elif key == 'sample_size':
                self.sample_size = value
            elif key == 'optimizer_learning_rate':
                self.optimizer_learning_rate = value
            elif key == 'discriminator_layers':
                self.discriminator_layers = value
            elif key == 'generator_layers':
                self.generator_layers = value
            elif key == 'generator_alpha':
                self.generator_alpha = value
            elif key == 'generator_momentum':
                self.generator_momentum = value

    def setup(self):
        """ Setups the GAN """
        # TODO new method  called from init opt passed

        print("Attack type: " + self.attack_type)

        conn = SQLConnector()
        data = conn.pull_kdd99(attack=self.attack_type, num=4000)
        dataframe = pd.DataFrame.from_records(data=data,
                                              columns=conn.pull_kdd99_columns(allQ=True))

        # ==========
        # ENCODING
        # ==========
        # https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn

        d = defaultdict(LabelEncoder)

        fit = dataframe.apply(lambda x: d[x.name].fit_transform(x))  # fit is encoded dataframe
        dataset = fit.values  # transform to ndarray

        # to visually judge encoded dataset
        print("Real encoded " + self.attack_type + " attacks:")
        print(dataset[:1])

        # Set X as our input data and Y as our label
        self.X_train = dataset[:, 0:41].astype(float)
        Y_train = dataset[:, 41]

        # labels for data. 1 for valid attacks, 0 for fake (generated) attacks
        self.valid = np.ones((self.batch_size, 1))
        self.fake = np.zeros((self.batch_size, 1))

    def build(self):
        """ Build the GAN """
        # build the discriminator portion

        self.discriminator = self.build_discriminator()  # self.discriminator_layers
        self.discriminator.trainable = False  # this helps, I guess
        self.discriminator.compile(
            loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # build the generator portion
        gen_args = {
            'attack_type': self.attack_type,
            'layers': self.generator_layers,
            'alpha': self.generator_alpha,
            'momentum': self.generator_momentum
        }
        self.generator = self.build_generator()  # **gen_args

        # input and output of our combined model
        z = Input(shape=(41,))
        reconstructed_attack = self.generator(z)
        validity = self.discriminator(reconstructed_attack)


        # build combined model from generator and discriminator
        self.gan = Model(z, validity)
        self.gan = Model(z, [reconstructed_attack, validity])
        self.gan.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                       loss_weights=[self.r_alpha, 1],
                                       optimizer=self.optimizer)

    def train(self):
        """ Trains the GAN system """
        # break condition for training (when diverging)
        loss_increase_count = 0
        prev_g_loss = 0


        conn = SQLConnector()

        idx = np.arange(self.batch_size)

        ones = np.ones((self.batch_size, 1))
        zeros = np.zeros((self.batch_size, 1))

        for epoch in range(50000):
            # print('Epoch ({}/{})-------------------------------------------------'.format(epoch, self.max_epochs))
            # selecting batch_size random attacks from our training data
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            attacks = self.X_train[idx]

            # generate a matrix of noise vectors
            noise = np.random.normal(0, 1, (self.batch_size, 41))

            # create an array of generated attacks
            gen_attacks = self.generator.predict(attacks)

            # loss functions, based on what metrics we specify at model compile time
            d_loss_real = self.discriminator.train_on_batch(
                attacks, self.valid)
            d_loss_fake = self.discriminator.train_on_batch(
                gen_attacks, self.fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # generator loss function
            g_loss = self.gan.train_on_batch(attacks, [gen_attacks, ones])
            g_loss = self.gan.train_on_batch(attacks, [gen_attacks, ones])



            if epoch % 499 == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
                print('Real attack:')
                print(attacks[150])
                print('Reconstructed attack:')
                print(gen_attacks[150].round(3))


    def test(self):
        """ A GAN should know how to test itself and save its results into a confusion matrix. """
        # TODO
        pass

    ##########################################################################################
    # Uses Sklearn's confusion matrix maker
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    ##########################################################################################
    def make_confusion_matrix(self, y_true, y_pred):
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        self.classification_report = classification_report(y_true, y_pred)

    ################################################################################
    # Use these to save instances of a trained network with some desirable settings
    # Suggestion to save and load from the object's __dict__ taken from:
    # https://stackoverflow.com/questions/2709800/how-to-pickle-yourself
    ################################################################################


    def signal_handler(sig, frame):
        """ Catches Crl-C command to print from database before ending """
        conn = SQLConnector()
        hypers = conn.read_hyper()  # by epoch?
        gens = conn.read_gens()  # by epoch?
        print("\n\nMYSQL DATA:\n==============")
        print("hypers  " + str(hypers))
        print("\ngens  " + str(gens) + "\n")
        sys.exit(0)


        signal.signal(signal.SIGINT, signal_handler)

    def build_generator(self, ):
        model = Sequential()

        attack = Input(shape=(41,), name='z')
        # model.add(Dense(25))
        # model.add(LeakyReLU(alpha=self.alpha))
        # model.add(Dropout(self.dropout))
        model.add(Dense(10))
        encoded = model.add(LeakyReLU(alpha=self.alpha))
        # model.add(Dropout(self.dropout))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=self.alpha))
        # model.add(Dense(25))
        # model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dense(41))
        model.add(LeakyReLU(alpha=self.alpha))

        decoded = model(attack)
        return Model(attack, decoded)

    def build_discriminator(self, ):
        """ Builds the discriminator """
        model = Sequential()

        # needs 41 for input and needs 1 for output
        model.add(Dense(41, input_dim=41, activation='relu'))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dropout(self.dropout))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dropout(self.dropout))
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=self.alpha))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='sigmoid'))  # outputs 0 to 1, 1 being read and 0 being fake

        attack = Input(shape=(41,))
        validity = model(attack)

        return Model(attack, validity)


def main():
    """ Auto run main method """
    args = {
        'attack_type': "neptune",  # optional v
        'max_epochs': 7000,
        'batch_size': 255,
        'sample_size': 500,
        'optimizer_learning_rate': 0.001
    }
    gan = NDGAN(**args)
    gan.train()


if __name__ == "__main__":
    main()
