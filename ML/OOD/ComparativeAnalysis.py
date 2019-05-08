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
def main():

    print()
    conn = SQLConnector()
    data = conn.pull_all_attacks(num=5000)
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

    unbalanced_array = unbalanced_df.values
    balanced_array = balanced_df.values

    gen_data = np.asarray(conn.read_gen_attacks_acc_thresh(.90, 1000))
    gen_df = pd.DataFrame.from_records(data=data,
                                          columns=conn.pull_kdd99_columns(allQ=True))

    balanced_df = pd.concat([balanced_df, gen_df])
    print(len(balanced_df))



    # BEGIN LOOP
    # Create two identical multi-class classifiers, make sure their output dimensions match the number of classes in our data

    # Train both classifiers on the dataset

    # Use the same test set and get y_predicted from both classifiers
    # Build confusion matrices for both classifiers
    # Measure some metrics based on the confusion matrix
    # Figure out how to structure our data and measurements and upload the data
    # END LOOP
    
if __name__ == "__main__":
    main()
