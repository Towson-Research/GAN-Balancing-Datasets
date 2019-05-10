from collections import defaultdict
import numpy as np
import pandas as pd

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from discriminator import Discriminator
from mysql import SQLConnector
from utilities import Utilities as util
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
import keras

def main():

    print()
    conn = SQLConnector()
    data = conn.pull_all_attacks(num=10000)
    dataframe = pd.DataFrame.from_records(data=data,
                                          columns=conn.pull_kdd99_columns(allQ=True))
    d = defaultdict(LabelEncoder)
    features = dataframe.iloc[:, :41]
    attack_labels = dataframe.iloc[:, 41:]

    for i in range(0, attack_labels.size):
        attack_labels.at[i, 'attack_type'] = util.attacks_to_num(attack_labels.at[i, 'attack_type'])

    fit = features.apply(lambda x: d[x.name].fit_transform(x))

    unbalanced_df = fit.join(attack_labels)
    balanced_df = unbalanced_df.copy(deep=True)



    gen_data = np.asarray(conn.read_gen_attacks_acc_thresh(.90, 1000))
    gen_df = pd.DataFrame.from_records(gen_data,
                                          columns=conn.pull_kdd99_columns(allQ=True))
    gen_df = gen_df.fillna(0)
    balanced_df = pd.concat([balanced_df, gen_df])
    print(len(balanced_df))

    unbalanced_array = unbalanced_df.values
    balanced_array = balanced_df.values



    # BEGIN LOOP
    # Create two identical multi-class classifiers, make sure their output dimensions match the number of classes in our data

    layers = [16, 32, 16]
    alpha = 0.1
    dropout = 0.3

    unb_labels = unbalanced_array[:, 41]
    [unb_classes, unb_counts] = np.unique(unb_labels, return_counts=True)
    print("Unique classes in unbalanced labels: ")
    print(unb_classes)
    print("Counts for the classes in unbalanced labels: ")
    print(unb_counts)
    unb_class_count = len(unb_classes)
    print("Number of classes in unbalanced dataset: " + str(unb_class_count))

    bal_labels = balanced_array[:, 41]
    [bal_classes, bal_counts] = np.unique(bal_labels, return_counts=True)

    dummy_bal_labels = np_utils.to_categorical(bal_labels)
    bal_class_count = len(bal_classes)
    print("Number of classes in balanced dataset: " + str(bal_class_count))

    print("Unique classes in balanced labels: ")
    print(bal_classes)
    print("Counts for the classes in balanced labels: ")
    print(bal_counts)

    for j in range (0, 100):
        unbalanced_classifier = build_discriminator(layers, alpha, dropout, unb_class_count)
        balanced_classifier = build_discriminator(layers, alpha, dropout, bal_class_count)

        optimizer = Adam(.001)
        unbalanced_classifier.compile(
                    loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        balanced_classifier.compile(
                    loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # encoding labels, classifier wants them in range 0 to num_classes
        unb_enc = LabelEncoder()
        bal_enc = LabelEncoder()

        unb_labels = unbalanced_array[:, 41]
        bal_labels = balanced_array[:, 41]

        unb_enc = unb_enc.fit(unb_labels)
        bal_enc = bal_enc.fit(bal_labels)

        unbalanced_array[:, 41] = unb_enc.transform(unbalanced_array[:, 41])
        balanced_array[:, 41] = bal_enc.transform(balanced_array[:, 41])
        [unb_classes, _] = np.unique(unbalanced_array[:, 41], return_counts=True)
        unb_cm = train(unbalanced_classifier, unbalanced_array)
        bal_cm = train(balanced_classifier, balanced_array)

        print("Metrics for iteration " + str(j))
        # print("Confusion matrix of unbalanced: ")
        # print
        print("Accuracy of unbalanced: " + str(getmetrics(unb_cm)))

        # print("Confusion matrix of balanced: ")
        # print(bal_cm)
        print("Accuracy of balanced" + str(getmetrics(bal_cm)))

        print("Diff: " + str(getmetrics(bal_cm) - getmetrics(unb_cm)))

        # TODO: Use the same test set and get y_predicted from both classifiers. What if there are classes in the test set that are only in the gens pulled?
        # TODO: Actually use useful gen attacks to balance the classes most needing of it. Test set should only be from the real kdd99
        # Build confusion matrices for both classifiers
        # Measure some metrics based on the confusion matrix
        # Figure out how to structure our data and measurements and upload the data
        # END LOOP
def train(classifier, data):
    callbacks = [keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # ==================================
    # Splitting Data into subsets
    # ==================================

    train_data = data[:, :41].astype(int)
    train_labels = data[:, 41]

    validationToTrainRatio = 0.05
    validationSize = int(validationToTrainRatio * len(train_data))
    val_data = train_data[:validationSize]
    val_labels = train_labels[:validationSize]
    train_data = train_data[validationSize:]
    train_labels = train_labels[validationSize:]

    testToTrainRatio = 0.05
    testSize = int(testToTrainRatio * len(train_data))
    test_data = train_data[:testSize]
    test_labels = train_labels[:testSize]
    train_data = train_data[testSize:]
    train_labels = train_labels[testSize:]
    
    history = classifier.fit(train_data,
                                 train_labels,
                                 epochs=10,
                                 batch_size=128,
                                 callbacks=callbacks,
                                 validation_data=(val_data, val_labels),
                                 verbose=0)

    y_pred = classifier.predict(test_data)
    y_class = y_pred.argmax(axis=-1)
    performance = classifier.evaluate(test_data, test_labels)
    cm = confusion_matrix(test_labels.astype(int), y_class.astype(int))
    return cm

def build_discriminator(layers, alpha, dropout, num_of_classes):
    """ Builds the discriminator """
    model = Sequential()

    # needs 41 for input and needs 1 for output
    model.add(Dense(layers[0], input_dim=41, activation='relu'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))
    model.add(Dense(layers[1]))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))
    model.add(Dense(layers[2]))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))
    model.add(Dense(num_of_classes, activation='sigmoid'))  # outputs 0 to 1, 1 being read and 0 being fake

    attack = Input(shape=(41,))
    validity = model(attack)

    return Model(attack, validity)

def getmetrics(cm):
    return np.trace(cm) / np.sum(cm)

if __name__ == "__main__":
    main()