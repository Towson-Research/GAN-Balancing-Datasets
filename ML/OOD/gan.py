#!/usr/bin/env python3
# Team TJ-MAM
from collections import defaultdict
import numpy as np
import pandas as pd

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import signal
import sys

from discriminator import Discriminator
from generator import Generator
from evaluator import Evaluator
from mysql import SQLConnector
from utilities import Utilities as util

try:
    import cPickle as pickle
except:
    import pickle


class GAN(object):

    def __init__(self, **kwargs):
        """ Constructor """
        self._defaults()
        self._args(kwargs)  # override defaults with args passed
        self.setup()
        self.build()

    def _defaults(self):
        """ Sets default variable values """
        self.attack_type = None
        self.discriminator = None
        self.generator = None
        self.gan = None
        self.evaluator = None

        # saved_states can be used to save states of a GAN, say
        # 5 of them so that the best can be saved when breaking out.
        self.saved_states = []
        self.confusion_matrix = None
        self.classification_report = None

        self.optimizer_learning_rate = 0.001
        self.optimizer = Adam(self.optimizer_learning_rate)

        self.max_epochs = 7000
        self.batch_size = 255
        self.sample_size = 500

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
        data = conn.pull_kdd99(attack=self.attack_type, num=5000)
        dataframe = pd.DataFrame.from_records(data=data,
                columns=conn.pull_kdd99_columns(allQ=True))

        # ==========
        # ENCODING
        # ==========
        # https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn

        d = defaultdict(LabelEncoder)
        features = dataframe.iloc[:, :41]
        attack_labels = dataframe.iloc[:, 41:]

        for i in range(0, attack_labels.size):
            attack_labels.at[i, 'attack_type'] = util.attacks_to_num(attack_labels.at[i, 'attack_type'])

        fit = features.apply(lambda x: d[x.name].fit_transform(x))  # fit is encoded dataframe

        dataframe = fit.join(attack_labels)
        dataset = dataframe.values   # transform to ndarray

        #TODO: Move this entire process outside of gan.py? creating the evaluation model may take time and doesn't need to be redone for every GAN model Moving this
        #TODO: and then handling evaluation and database uploading to another script (like in the automation script) may be more efficient


        #pulling and encoding data for evaluation model
        '''
        eval_data = np.asarray(conn.pull_evaluator_data(1000000, self.attack_type))
        eval_dataframe = pd.DataFrame.from_records(data=eval_data,
                                              columns=conn.pull_kdd99_columns(allQ=True))
        encoded_eval_df = eval_dataframe.apply(lambda x: d[x.name].fit_transform(x))
        '''
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



        #print(fit)

        # ==========
        # DECODING
        # ==========

#         print("===============================================")
#         print("decoded:")
#         print("===============================================")
#         decode_test = dataset[:5]  # take a slice from the ndarray that we want to decode
#         decode_test_df = pd.DataFrame(decode_test, columns=conn.pull_kdd99_columns())  # turn that ndarray into a dataframe with correct column names and order
#         decoded = decode_test_df.apply(lambda x: d[x.name].inverse_transform(x))  # decode that dataframe
#         print(decoded)


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

        #doing this so we can read the data from the evaluator object
        evaluator_object = Evaluator(**eval_args)
        self.evaluator = evaluator_object.get_model()
        print("Evaluator metrics after training:")
        print(evaluator_object.performance)
        disc_layers = self.generator_layers.copy()
        disc_layers.reverse()
        print(disc_layers)
        disc_args = {
                 'layers': disc_layers,
                 'alpha': self.generator_alpha,
                 'momentum': self.generator_momentum
                 }
        self.discriminator = Discriminator(**disc_args).get_model()#self.discriminator_layers
        self.discriminator.compile(
                loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # build the generator portion
        gen_args = {
                 'layers': self.generator_layers,
                 'alpha': self.generator_alpha,
                 }
        self.generator = Generator(**gen_args).get_model()#**gen_args

        # input and output of our combined model
        z = Input(shape=(41,))
        attack = self.generator(z)
        validity = self.discriminator(attack)

        # build combined model from generator and discriminator
        self.gan = Model(z, validity)
        self.gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)

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
            d_loss_real = self.discriminator.train_on_batch(
                    attacks, self.valid)
            d_loss_fake = self.discriminator.train_on_batch(
                    gen_attacks, self.fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # generator loss function
            g_loss = self.gan.train_on_batch(noise, self.valid)

            if epoch % 500 == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Loss change: %.3f, Loss increases: %.0f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss, g_loss - prev_g_loss, loss_increase_count))

<<<<<<< HEAD
            '''
            # ======================
            # Decoding attacks
            # ======================
            if epoch % 20 == 0:
                decode = gen_attacks[:1]  # take a slice from the ndarray that we want to decode
                #MAX QUESTION: Do we plan on changing the shape of this at some
                #point? If not just do
                #decode = gen_attacks[0]
                #decode_ints = decode.astype(int)
                #print("decoded floats ======= " + str(decode))
                #print("decoded ints ======= " + str(decode_ints))
                accuracy_threshold = 55
                accuracy = (d_loss[1] * 100)
                if(accuracy > accuracy_threshold):
                    # print out first result
                    list_of_lists = util.decode_gen(decode)
                    print(list_of_lists)

                    # ??????
                    gennum = 1  # pickle
                    modelnum = 1

                    layersstr = str(self.generator_layers[0]) + "," + str(self.generator_layers[1]) + "," + str(self.generator_layers[2])
                    attack_num = util.attacks_to_num(self.attack_type)

                    # send all to database
                    print(np.shape(list_of_lists))
                    for lis in list_of_lists:
                        #print(len(lis))
                        conn.write(gennum=gennum, modelnum=modelnum, layersstr=layersstr,
                                attack_type=attack_num, accuracy=accuracy, gen_list=lis)

                        # peek at our results
            '''

        ''' Calculating values for database, and writing to db
        '''
        '''accuracy = (d_loss[1] * 100)'''


        predicted_gen_attack_labels = self.evaluator.predict(gen_attacks).transpose().astype(int)
        gen_attack_labels = np.full(predicted_gen_attack_labels.shape, 1)

        print("Generated attack labels: ")
        print(gen_attack_labels)
        print("Predicted labels of generated attacks: ")
        print(predicted_gen_attack_labels)

        right = (predicted_gen_attack_labels == 1).sum()
        wrong = (predicted_gen_attack_labels != 1).sum()

        accuracy = (right / float(right + wrong))

        print("Accuracy of evaluator on generated data: %.4f " % accuracy)
        if accuracy > .50:
            conn.write_gens(gen_attacks, util.attacks_to_num(self.attack_type))

        layersstr = str(self.generator_layers[0]) + "," + str(self.generator_layers[1]) + "," + str(
           self.generator_layers[2])
        attack_num = util.attacks_to_num(self.attack_type)
||||||| merged common ancestors
            '''
            # ======================
            # Decoding attacks
            # ======================
            if epoch % 20 == 0:
                decode = gen_attacks[:1]  # take a slice from the ndarray that we want to decode
                #MAX QUESTION: Do we plan on changing the shape of this at some
                #point? If not just do
                #decode = gen_attacks[0]
                #decode_ints = decode.astype(int)
                #print("decoded floats ======= " + str(decode))
                #print("decoded ints ======= " + str(decode_ints))
                accuracy_threshold = 55
                accuracy = (d_loss[1] * 100)
                if(accuracy > accuracy_threshold):
                    # print out first result
                    list_of_lists = util.decode_gen(decode)
                    print(list_of_lists)

                    # ??????
                    gennum = 1  # pickle
                    modelnum = 1

                    layersstr = str(self.generator_layers[0]) + "," + str(self.generator_layers[1]) + "," + str(self.generator_layers[2])
                    attack_num = util.attacks_to_num(self.attack_type)

                    # send all to database
                    print(np.shape(list_of_lists))
                    for lis in list_of_lists:
                        #print(len(lis))
                        conn.write(gennum=gennum, modelnum=modelnum, layersstr=layersstr,
                                attack_type=attack_num, accuracy=accuracy, gen_list=lis)

                        # peek at our results
            '''
            accuracy = (d_loss[1] * 100)
            layersstr = str(self.generator_layers[0]) + "," + str(self.generator_layers[1]) + "," + str(
                self.generator_layers[2])
            attack_num = util.attacks_to_num(self.attack_type)
=======
            accuracy = (d_loss[1] * 100)
            layersstr = str(self.generator_layers[0]) + "," + str(self.generator_layers[1]) + "," + str(
                self.generator_layers[2])
            attack_num = util.attacks_to_num(self.attack_type)
>>>>>>> master

        conn.write_hypers(layerstr=layersstr, attack_encoded=attack_num, accuracy=accuracy)

        # TODO: Log our generated attacks to the gens table
        # TODO: Add foreign key for attack type in hypers table

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
            'attack_type': "portsweep",    # optional v
            'max_epochs': 7000,
            'batch_size': 255,
            'sample_size': 500,
            'optimizer_learning_rate': 0.001
            }
    gan = GAN(**args)
    gan.train()


if __name__ == "__main__":
    main()

