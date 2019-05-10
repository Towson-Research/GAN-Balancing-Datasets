from mysql import SQLConnector
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
from collections import defaultdict
from utilities import Utilities as util
from sklearn.utils import shuffle

def main():
    conn = SQLConnector()
    data = np.asarray(conn.pull_evaluator_data(30000, 'satan'))
    dataframe = pd.DataFrame.from_records(data=data,
                                          columns=conn.pull_kdd99_columns(allQ=True))

    features = dataframe.iloc[:, :41]
    attacks = dataframe.iloc[:, 41:]

    print(attacks.at[0,'attack_type'])
    print(type(attacks.at[0, 'attack_type']))
    for i in range(0, attacks.size):
        attacks.at[i, 'attack_type'] = util.attacks_to_num(attacks.at[i, 'attack_type'])

    # using 0 as the label for non-neptune data
    for i in range(0, attacks.size):
        if(attacks.at[i, 'attack_type'] == 16):
            attacks.at[i, 'attack_type'] = 1
        else:
            attacks.at[i, 'attack_type'] = 0


    print(attacks)

    d = defaultdict(LabelEncoder)
    encoded_features_df = features.apply(lambda x: d[x.name].fit_transform(x))
    eval_dataset_df = encoded_features_df.join(attacks)
    eval_dataset_df = shuffle(eval_dataset_df)
    print(eval_dataset_df)

    #Print encoded values to a csv
    eval_dataset_df.to_csv('SatanAndNonsatan.csv', header=False, index=False)



if __name__ == "__main__":
    main()