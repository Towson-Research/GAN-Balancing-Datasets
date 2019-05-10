#!/usr/bin/env python3
# Team TJ-MAM
from collections import defaultdict
import numpy as np
import pandas as pd
import keras.backend as K

from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

import signal
import sys

from critic import Critic
from generator import Generator
from evaluator import Evaluator
from mysql import SQLConnector
from utilities import Utilities as util

try:
    import cPickle as pickle
except:
    import pickle


# TODO: Implement weight clipping
# TODO: Implement WGAN loss function
# TODO: Implement RMSprop loss function

class WGAN(object):

    def __init__(self, **kwargs):
        """ Constructor """
        self._defaults()
        self._args(kwargs)  # override defaults with args passed
        self.setup()
        self.build()

    def _defaults(self):
        """ Sets default variable values """
        self.attack_type = None
        self.critic = None
        self.generator = None
        self.gan = None
        self.evaluator = None

        # saved_states can be used to save states of a GAN, say
        # 5 of them so that the best can be saved when breaking out.
        self.saved_states = []
        self.confusion_matrix = None
        self.classification_report = None
        self.scaler = None

        self.optimizer_learning_rate = 0.001
        self.optimizer = RMSprop(lr=0.00005)

        self.max_epochs = 7000
        self.batch_size = 255
        self.sample_size = 500
        self.clip_value = 0.01

        self.valid = None
        self.fake = None
        self.X_train = None

        self.generator_alpha = 0.1
        self.generator_momentum = 0.0
        self.generator_layers = [8, 16, 32]

        self.confusion_matrix = None
        self.classification_report = None

        self.save_file = None

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
            elif key == 'critic':
                self.critic = value
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
        data = conn.pull_kdd99(attack=self.attack_type, num=5000)
        dataframe = pd.DataFrame.from_records(data=data,
                columns=conn.pull_kdd99_columns(allQ=True))

        # ==========
        # ENCODING
        # ==========
        # https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn

        d = defaultdict(LabelEncoder)

        # Splitting the data from features and lablels. Want labels to be consistent with evaluator encoding, so
        # we use the utils attack_to_num function
        features = dataframe.iloc[:, :41]
        attack_labels = dataframe.iloc[:, 41:]

        for i in range(0, attack_labels.size):
            attack_labels.at[i, 'attack_type'] = util.attacks_to_num(attack_labels.at[i, 'attack_type'])

        features = features.apply(lambda x: d[x.name].fit_transform(x))  # fit is encoded dataframe

        # feature scaling, reccomended from github implementation
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_features = self.scaler.fit_transform(features.astype(float))
        scaled_df = pd.DataFrame(data=scaled_features)


        # Join the seperately encoded sections back into one dataframe
        dataframe = scaled_df.join(attack_labels)
        dataset = dataframe.values   # transform to ndarray
        print(dataset)

        # TODO: Feature scaling? May be necessary. Has to be on a per-feature basis?

        # Splitting up the evaluation dataset. Should maybe be moved?
        eval_dataset = pd.read_csv('PortsweepAndNonportsweep.csv', header=None)
        eval_dataset = eval_dataset.values

        self.eval_dataset_X = eval_dataset[:,0:41].astype(int)
        self.eval_dataset_Y = eval_dataset[:, 41]

        validationToTrainRatio = 0.05
        validationSize = int(validationToTrainRatio * len(self.eval_dataset_X))
        self.eval_validation_data = self.eval_dataset_X[:validationSize]
        self.eval_validation_labels = self.eval_dataset_Y[:validationSize]
        self.eval_dataset_X = self.eval_dataset_X[validationSize:]
        self.eval_dataset_Y = self.eval_dataset_Y[validationSize:]

        testToTrainRatio = 0.05
        testSize = int(testToTrainRatio * len(self.eval_dataset_X))
        self.eval_test_data = self.eval_dataset_X[:testSize]
        self.eval_test_labels = self.eval_dataset_Y[:testSize]
        self.eval_dataset_X = self.eval_dataset_X[testSize:]
        self.eval_dataset_Y = self.eval_dataset_Y [testSize:]


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
        eval_args = {
            'train_data': self.eval_dataset_X,
            'train_labels':  self.eval_dataset_Y,
            'validation_data': self.eval_validation_data,
            'validation_labels': self.eval_validation_labels,
            'test_data': self.eval_test_data,
            'test_labels': self.eval_test_labels,

        }

        # Doing this so we can read the data from the evaluator object
        evaluator_object = Evaluator(**eval_args)
        self.evaluator = evaluator_object.get_model()


        print("Evaluator metrics after training:")
        print(evaluator_object.performance)
        critic_layers = self.generator_layers.copy()
        critic_layers.reverse()
        print(critic_layers)
        critic_args = {
                 'layers': critic_layers,
                 'alpha': self.generator_alpha,
                 'optimizer': self.optimizer,
                 }
        self.critic = Critic(**critic_args).get_model()#self.discriminator_layers
        self.critic.compile(
                loss=self.wasserstein_loss, optimizer=self.optimizer, metrics=['accuracy'])

        # build the generator portion
        gen_args = {
                 'layers': self.generator_layers,
                 'alpha': self.generator_alpha,
                 }
        self.generator = Generator(**gen_args).get_model()#**gen_args

        # input and output of our combined model
        z = Input(shape=(41,))
        attack = self.generator(z)
        validity = self.critic(attack)

        # build combined model from generator and discriminator
        self.gan = Model(z, validity)
        self.gan.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)

    def train(self):
        """ Trains the GAN system """
        # break condition for training (when diverging)
        loss_increase_count = 0
        prev_g_loss = 0


        conn = SQLConnector()

        idx = np.arange(self.batch_size)

        for epoch in range(self.max_epochs):
            #selecting batch_size random attacks from our training data
            #idx = np.random.randint(0, X_train.shape[0], batch_size)
            attacks = self.X_train[idx]

            # generate a matrix of noise vectors
            noise = np.random.normal(0, 1, (self.batch_size, 41))

            # create an array of generated attacks
            gen_attacks = self.generator.predict(noise)

            # loss functions, based on what metrics we specify at model compile time
            c_loss_real = self.critic.train_on_batch(
                    attacks, self.valid)
            c_loss_fake = self.critic.train_on_batch(
                    gen_attacks, self.fake)
            d_loss = 0.5 * np.add(c_loss_real, c_loss_fake)

            for l in self.critic.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                l.set_weights(weights)

            # generator loss function
            g_loss = self.gan.train_on_batch(noise, self.valid)

            if epoch % 500 == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Loss change: %.3f, Loss increases: %.0f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss, g_loss - prev_g_loss, loss_increase_count))


        gen_attacks = self.scaler.inverse_transform(gen_attacks)
        predicted_gen_attack_labels = self.evaluator.predict(gen_attacks).transpose().astype(int)
        gen_attack_labels = np.full(predicted_gen_attack_labels.shape, 1)

        print("Generated attack labels: ")
        print(gen_attack_labels)
        print("Predicted labels of generated attacks: ")
        print(predicted_gen_attack_labels)

        right = (predicted_gen_attack_labels == 1).sum()
        wrong = (predicted_gen_attack_labels != 1).sum()

        accuracy = (right / float(right + wrong))

        print("5 generated attacks: ")
        print(gen_attacks[:5, :])
        print()
        print("Accuracy of evaluator on generated data: %.4f " % accuracy)
        if accuracy > .50:
            conn.write_gens(gen_attacks, util.attacks_to_num(self.attack_type))

        layersstr = str(self.generator_layers[0]) + "," + str(self.generator_layers[1]) + "," + str(
           self.generator_layers[2])
        attack_num = util.attacks_to_num(self.attack_type)

        conn.write_hypers(layerstr=layersstr, attack_encoded=attack_num, accuracy=accuracy)

        # TODO: Add foreign key for attack type in hypers table

    def test(self):
        """ A GAN should know how to test itself and save its results into a confusion matrix. """
        # TODO
        pass

    # This functions should only be passed the FEATURES, we don't want to scale the labels
    def feature_scale(self, dataset):
        # Scale all features, minus the label
        for i in range(0, len(dataset[0, :])):
            col_avg = np.mean(dataset[:, i])
            col_sd = np.std(dataset[:, i])
            dataset[:, i] = (dataset[:, i] - col_avg) / col_sd

    ##########################################################################################
    # Uses Sklearn's confusion matrix maker
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    ##########################################################################################
    def make_confusion_matrix(self, y_true, y_pred):
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        self.classification_report = classification_report(y_true, y_pred)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    ################################################################################
    # Use these to save instances of a trained network with some desirable settings
    # Suggestion to save and load from the object's __dict__ taken from:
    # https://stackoverflow.com/questions/2709800/how-to-pickle-yourself
    ################################################################################
    def save_this(self, filename):
        '''
            Provide a basic filename to pickle this object for recovery later.
            Unlike the load function, this requires a save file, so that it will
            never accidentally overwrite a previous file.
        '''
        self.save_file = filename + '.pickle'
        with open(self.save_file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load_state_from_file(self, filename=None):
        if not filename:
            if self.save_file:
                filename = self.save_file
            else:
                print("Error: No savefile for this object. \
                        \n Using save_this(filename) will set the save filename.")
                return
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict.__dict__)
            f.close()




def signal_handler(sig, frame):
    """ Catches Crl-C command to print from database before ending """
    conn = SQLConnector()
    writeOut(conn)
    sys.exit(0)
    print("did it work?")
signal.signal(signal.SIGINT, signal_handler)


def writeOut(conn):
   hypers = conn.read_hyper()  # by epoch?
   gens = conn.read_gens()   # by epoch?
   print("\n\nMYSQL DATA:\n==============")
   print("hypers  " + str(hypers))
   print("\ngens  " + str(gens) + "\n")

def main():
    """ Auto run main method """
    args = {
            'attack_type': "smurf",    # optional v
            'max_epochs': 7000,
            'batch_size': 255,
            'sample_size': 500,
            'optimizer_learning_rate': 0.001
            }
    gan = WGAN(**args)
    gan.train()


if __name__ == "__main__":
    main()
