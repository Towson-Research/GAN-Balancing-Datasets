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

from discriminator import Discriminator
from generator import Generator
from mysql import SQLConnector


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

        # saved_states can be used to save states of a GAN, say
        # 5 of them so that the best can be saved when breaking out.
        self.saved_states = []
        self.confusion_matrix = None
        self.classification_report = None

        self.optimizer_learning_rate = 0.0002
        self.optimizer_beta = 0.5
        self.optimizer = Adam(self.optimizer_learning_rate, self.optimizer_beta)

        self.max_epochs = 7000
        self.batch_size = 255
        self.sample_size = 500

        self.valid = None
        self.fake = None
        self.X_train = None

        self.discriminator_layers = [
            (30, 'relu'),
            (15, 'relu')
        ]
        self.generator_layers = [0, 0, 0]
        self.generator_alpha: 0.5
        self.generator_momentum: 0.8

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
            elif key == 'optimizer_beta':
                self.optimizer_beta = value
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
        data = conn.pull_kdd99(attack=self.attack_type, num=500)
        dataframe = pd.DataFrame.from_records(data=data, columns=conn.pull_kdd99_columns(all=True))

        # ==========
        # ENCODING
        # ==========
        # https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn

        d = defaultdict(LabelEncoder)

        fit = dataframe.apply(lambda x: d[x.name].fit_transform(x))  # fit is encoded dataframe
        dataset = fit.values   # transform to ndarray

        #print(fit)


        # ==========
        # DECODING
        # ==========
        '''
        print("===============================================")
        print("decoded:")
        print("===============================================")
        decode_test = dataset[:5]  # take a slice from the ndarray that we want to decode
        decode_test_df = pd.DataFrame(decode_test, columns=conn.pull_kdd99_columns())  # turn that ndarray into a dataframe with correct column names and order
        decoded = decode_test_df.apply(lambda x: d[x.name].inverse_transform(x))  # decode that dataframe
        print(decoded)
        '''


        # to visually judge results
        print("Real " + self.attack_type + " attacks:")
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

        self.discriminator = Discriminator(self.discriminator_layers).get_model()
        self.discriminator.compile(
            loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # build the generator portion
        gen_args = {
            'attack_type': self.attack_type,
            'layers': self.generator_layers,
            'alpha': self.generator_alpha,
            'momentum': self.generator_momentum
        }
        self.generator = Generator(**gen_args).get_model()

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

        for epoch in range(self.max_epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # selecting batch_size random attacks from our training data
            idx = np.random.randint(0, self.X_train.shape[0], self.batch_size)
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
            if epoch % 100 == 0:
                if(epoch == 0):
                    print("\n")
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Loss change: %.3f, Loss increases: %.0f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss, g_loss - prev_g_loss, loss_increase_count))

            # if our generator loss increased this iteration, increment the counter by 1
            if (g_loss - prev_g_loss) > 0:
                loss_increase_count = loss_increase_count + 1
            else:
                loss_increase_count = 0  # otherwise, reset it to 0, we are still training effectively

            prev_g_loss = g_loss

            if loss_increase_count > 5:
                print('Stoping on iteration: ', epoch)
                break

            if epoch % 20 == 0:

                
                decode = gen_attacks[:1]  # take a slice from the ndarray that we want to decode
                #decode_ints = decode.astype(int)

                #print("decoded floats ======= " + str(decode))
                #print("decoded ints ======= " + str(decode_ints))

                self.decode_gen(decode)

                '''
                f = open("ABC.txt", "a")
                np.savetxt("ABC.txt", gen_attacks, fmt="%.0f")
                f.close()
                #self._push_results(epoch, gen_attacks)
                '''

        # peek at our results
        #results = self._pull_results(epoch)
        results = np.loadtxt("Results.txt")
        print("Generated " + self.attack_type + " attacks: ")
        print(results[:2])

    def decode_gen(self, array):

        # index 24 - 39 floats
        rows = array.shape[0]
        cols = array.shape[1]

        list_of_lists = []

        for r in range(rows):
            details = []  # list
            for c in range(cols):
                #print(int_ndarray[r][c])
                if(c == 0):   # duration
                    details.append(int(array[r][c]))
                elif(c == 1):   # protocol type
                    if(int(array[r][c]) == 0):
                        details.append('tcp')
                    elif(int(array[r][c]) == 1):
                        details.append('udp')
                    elif(int(array[r][c]) == 2):
                        details.append('icmp')
                elif(c == 2):   # service
                    if(int(array[r][c]) == 0):
                        details.append('http')
                    elif(int(array[r][c]) == 1):
                        details.append('smtp')
                    elif(int(array[r][c]) == 2):
                        details.append('domain_u')
                    elif(int(array[r][c]) == 3):
                        details.append('auth')
                    elif(int(array[r][c]) == 4):
                        details.append('finger')
                    elif(int(array[r][c]) == 5):
                        details.append('telnet')
                    elif(int(array[r][c]) == 6):
                        details.append('eco_i')
                    elif(int(array[r][c]) == 7):
                        details.append('ftp')
                    elif(int(array[r][c]) == 8):
                        details.append('ntp_u')
                    elif(int(array[r][c]) == 9):
                        details.append('ecr_i')
                    elif(int(array[r][c]) == 10):
                        details.append('other')
                    elif(int(array[r][c]) == 11):
                        details.append('urp_i')
                    elif(int(array[r][c]) == 12):
                        details.append('private')
                    elif(int(array[r][c]) == 13):
                        details.append('pop_3')
                    elif(int(array[r][c]) == 14):
                        details.append('ftp_data')
                    elif(int(array[r][c]) == 15):
                        details.append('netstat')
                    elif(int(array[r][c]) == 16):
                        details.append('daytime')
                    elif(int(array[r][c]) == 17):
                        details.append('ssh')
                    elif(int(array[r][c]) == 18):
                        details.append('echo')
                    elif(int(array[r][c]) == 19):
                        details.append('time')
                    elif(int(array[r][c]) == 20):
                        details.append('name')
                    elif(int(array[r][c]) == 21):
                        details.append('whois')
                    elif(int(array[r][c]) == 22):
                        details.append('domain')
                    elif(int(array[r][c]) == 23):
                        details.append('mtp')
                    elif(int(array[r][c]) == 24):
                        details.append('gopher')
                    elif(int(array[r][c]) == 25):
                        details.append('remote_job')
                    elif(int(array[r][c]) == 26):
                        details.append('rje')
                    elif(int(array[r][c]) == 27):
                        details.append('ctf')
                    elif(int(array[r][c]) == 28):
                        details.append('supdup')
                    elif(int(array[r][c]) == 29):
                        details.append('link')
                    elif(int(array[r][c]) == 30):
                        details.append('systat')
                    elif(int(array[r][c]) == 31): 
                        details.append('discard')
                    elif(int(array[r][c]) == 32):
                        details.append('X11')
                    elif(int(array[r][c]) == 33):
                        details.append('shell')
                    elif(int(array[r][c]) == 34):
                        details.append('login')
                    elif(int(array[r][c]) == 35):
                        details.append('imap4')
                    elif(int(array[r][c]) == 36):
                        details.append('nntp')
                    elif(int(array[r][c]) == 37):
                        details.append('uucp')
                    elif(int(array[r][c]) == 38):
                        details.append('pm_dump')
                    elif(int(array[r][c]) == 39):
                        details.append('IRC')
                    elif(int(array[r][c]) == 40):
                        details.append('Z39_50')
                    elif(int(array[r][c]) == 41):
                        details.append('netbios_dgm')
                    elif(int(array[r][c]) == 42):
                        details.append('ldap')
                    elif(int(array[r][c]) == 43):
                        details.append('sunrpc')
                    elif(int(array[r][c]) == 44):
                        details.append('courier')
                    elif(int(array[r][c]) == 45):
                        details.append('exec')
                    elif(int(array[r][c]) == 46):
                        details.append('bgp')
                    elif(int(array[r][c]) == 47):
                        details.append('csnet_ns')
                    elif(int(array[r][c]) == 48):
                        details.append('http_443')
                    elif(int(array[r][c]) == 49):
                        details.append('klogin')
                    elif(int(array[r][c]) == 50):
                        details.append('printer')
                    elif(int(array[r][c]) == 51):
                        details.append('netbios_ssn')
                    elif(int(array[r][c]) == 52):
                        details.append('pop_2')
                    elif(int(array[r][c]) == 53):
                        details.append('nnsp')
                    elif(int(array[r][c]) == 54):
                        details.append('efs')
                    elif(int(array[r][c]) == 55):
                        details.append('hostnames')
                    elif(int(array[r][c]) == 56):
                        details.append('uucp_path')
                    elif(int(array[r][c]) == 57):
                        details.append('sql_net')
                    elif(int(array[r][c]) == 58):
                        details.append('vmnet')
                    elif(int(array[r][c]) == 59):
                        details.append('iso_tsap')
                    elif(int(array[r][c]) == 60):
                        details.append('netbios_ns')
                    elif(int(array[r][c]) == 61):
                        details.append('kshell')
                    elif(int(array[r][c]) == 62):
                        details.append('urh_i')
                    elif(int(array[r][c]) == 63):
                        details.append('http_2784')
                    elif(int(array[r][c]) == 64):
                        details.append('harvest')
                    elif(int(array[r][c]) == 65):
                        details.append('aol')
                    elif(int(array[r][c]) == 66):
                        details.append('tftp_u')
                    elif(int(array[r][c]) == 67):
                        details.append('http_8001')
                    elif(int(array[r][c]) == 68):
                        details.append('tim_i')
                    elif(int(array[r][c]) == 69):
                        details.append('red_i')
                elif(c == 2):   # flag
                    if(int(array[r][c]) == 0):
                        details.append('SF')
                    elif(int(array[r][c]) == 1):
                        details.append('S2')
                    elif(int(array[r][c]) == 2):
                        details.append('S1')
                    elif(int(array[r][c]) == 3):
                        details.append('S3')
                    elif(int(array[r][c]) == 4):
                        details.append('OTH')
                    elif(int(array[r][c]) == 5):
                        details.append('REJ')
                    elif(int(array[r][c]) == 6):
                        details.append('RSTO')
                    elif(int(array[r][c]) == 7):
                        details.append('S0')
                    elif(int(array[r][c]) == 8):
                        details.append('RSTR')
                    elif(int(array[r][c]) == 9):
                        details.append('RSTOS0')
                    elif(int(array[r][c]) == 10):
                        details.append('SH')
                elif(c == 3):
                    details.append(int(array[r][c]))
                elif(c == 4):
                    details.append(int(array[r][c]))
                elif(c == 5):
                    details.append(int(array[r][c]))
                elif(c == 6):
                    details.append(int(array[r][c]))
                elif(c == 7):
                    details.append(int(array[r][c]))
                elif(c == 8):
                    details.append(int(array[r][c]))
                elif(c == 9):
                    details.append(int(array[r][c]))
                elif(c == 10):
                    details.append(int(array[r][c]))
                elif(c == 11):
                    details.append(int(array[r][c]))
                elif(c == 12):
                    details.append(int(array[r][c]))
                elif(c == 13):
                    details.append(int(array[r][c]))
                elif(c == 14):
                    details.append(int(array[r][c]))
                elif(c == 15):
                    details.append(int(array[r][c]))
                elif(c == 16):
                    details.append(int(array[r][c]))
                elif(c == 17):
                    details.append(int(array[r][c]))
                elif(c == 18):
                    details.append(int(array[r][c]))
                elif(c == 19):
                    details.append(int(array[r][c]))
                elif(c == 20):
                    details.append(int(array[r][c]))
                elif(c == 21):
                    details.append(int(array[r][c]))
                elif(c == 22):
                    details.append(int(array[r][c]))
                elif(c == 23):
                    details.append(int(array[r][c]))
                elif(c == 24):  # floats
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 25):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 26):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 27):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 28):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 29):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 30):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 31):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 32):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 33):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 34):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 35):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 36):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 37):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 38):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 39):
                    details.append(float("%0.2f" % array[r][c]))
                elif(c == 40):
                    if(int(array[r][c]) == 0):
                        details.append('normal')
                    elif(int(array[r][c]) == 1):
                        details.append('buffer_overflow')
                    elif(int(array[r][c]) == 2):
                        details.append('loadmodule')
                    elif(int(array[r][c]) == 3):
                        details.append('perl')
                    elif(int(array[r][c]) == 4):
                        details.append('neptune')
                    elif(int(array[r][c]) == 5):
                        details.append('smurf')
                    elif(int(array[r][c]) == 6):
                        details.append('guess_passwd')
                    elif(int(array[r][c]) == 7):
                        details.append('pod')
                    elif(int(array[r][c]) == 8):
                        details.append('teardrop')
                    elif(int(array[r][c]) == 9):
                        details.append('portsweep')
                    elif(int(array[r][c]) == 10):
                        details.append('ipsweep')
                    elif(int(array[r][c]) == 11):
                        details.append('land')
                    elif(int(array[r][c]) == 12):
                        details.append('ftp_write')
                    elif(int(array[r][c]) == 13):
                        details.append('back')
                    elif(int(array[r][c]) == 14):
                        details.append('imap')
                    elif(int(array[r][c]) == 15):
                        details.append('satan')
                    elif(int(array[r][c]) == 16):
                        details.append('phf')
                    elif(int(array[r][c]) == 17):
                        details.append('nmap')
                    elif(int(array[r][c]) == 18):
                        details.append('multihop')
                    elif(int(array[r][c]) == 19):
                        details.append('warezmaster')
                    elif(int(array[r][c]) == 20):
                        details.append('warezclient')
                    elif(int(array[r][c]) == 21):
                        details.append('spy')
                    elif(int(array[r][c]) == 22):
                        details.append('rootkit')

            list_of_lists.append(details)
        
        print("LOL: " + str(list_of_lists))

    '''
    def _push_results(self, epoch, gen_attacks):
        """ Pushes results into database """
        conn = SQLConnector()
        print(gen_attacks)
        d = defaultdict(LabelEncoder)
        decoded = gen_attacks.apply(lambda x: d[x.name].inverse_transform(x))
        print(decoded)
        #conn.write_gens(gen_attacks)
        #conn.write_hyper(1, "2,3,4", 5, 80.3)
        #conn.write_gens(1, 1, 1, 0, "tcp", "ftp_data", "REJ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0.00, 171, 62, 0.27, 0.02, 0.01, 0.03, 0.01, 0, 0.29, 0.02, 10)

    def _pull_results(self, epoch):
        """ Pulls results from database, returns list of lists """
        conn = SQLConnector()
        #TODO mysql sorts keys alphabetically 
        results = {}
        return results
        #conn.write_hyper(1, "2,3,4", 5, 80.3)
        #conn.write_gens(1, 1, 1, 0, "tcp", "ftp_data", "REJ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0.00, 171, 62, 0.27, 0.02, 0.01, 0.03, 0.01, 0, 0.29, 0.02, 10)
        #np.loadtxt(self.results_path + self.results_name)
    '''

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


def main():
    """ Auto run main method """
    args = {
        'attack_type': "neptune",    # optional v
        'max_epochs': 7000,
        'batch_size': 255,
        'sample_size': 500,
        'optimizer_learning_rate': 0.0002,
        'optimizer_beta': 0.5,
        'discriminator_layers': [(30, 'relu'), (15, 'relu')],
        'generator_layers': [20, 40, 30],
        'generator_alpha': 0.5,
        'generator_momentum': 0.8
    }
    gan = GAN(**args)
    gan.train()


if __name__ == "__main__":
    main()
