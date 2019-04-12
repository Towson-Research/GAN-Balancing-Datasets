from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from mysql import SQLConnector

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator / denominator
    return eta


def main():
    conn = SQLConnector()
    data = conn.pull_all_attacks(num=40000)
    dataframe = pd.DataFrame.from_records(data=data,
                                          columns=conn.pull_kdd99_columns(allQ=True))

    columns = conn.pull_kdd99_columns()
    print(type(columns))
    # ==========
    # ENCODING
    # ==========
    # https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn

    d = defaultdict(LabelEncoder)

    fit = dataframe.apply(lambda x: d[x.name].fit_transform(x))  # fit is encoded dataframe
    dataset = fit.values  # transform to ndarray

    df = pd.DataFrame(data=dataset, columns=columns)

    #TODO: Figure out what the fuck the method actually takes as params
    correlation_matrix = np.zeros(shape=(41,41))
    for i in range(1, len(columns) - 1):
        for j in range(0, len(columns) - 1):
            print(df.iloc[:i])
            print()
            print(df.iloc[:j])

            correlation_matrix[i, j] = correlation_ratio(df.iloc[:i], df.iloc[:j])
    categories = correlation_ratio(columns, dataset)
    print(type(categories))
    print(categories)




def correlation_heatmap(data):

    fig, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(, vmax=1.0, center=0, fmt='.2f',
    #            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();




if __name__ == "__main__":
    main()