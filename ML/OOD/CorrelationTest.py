from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import argparse
from os import path, makedirs
from sys import argv

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

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", "-m", type = str, dest = "mode", 
                        required = False, default = "show",
                        help = "Whether to show or save the heatmap. Use -m show or -m save.")
    parser.add_argument("--save_dir", "-dir", type = str, dest = "save_dir", 
                        required = False, default = "figs",
                        help = "Directory to save heatmap figures to, if any. Will be created if does not exist.")
    parser.add_argument("--num", "-n", type = str, dest = "num", 
                        required = False, default = 40000,
                        help = "Number of samples to pull from the database.")
    parser.add_argument("--host", "-ht", type = str, dest = "host", 
                        required = False, default = "localhost",
                        help = "Database host.")

    args = parser.parse_args()
    mode = args.mode
    save_dir = args.save_dir
    try:
        num = int(args.num)
    except Exception as e:
        print(e)
    host = args.host

    conn = SQLConnector(host = host)
    data = conn.pull_all_attacks(num, nodupes = True)
    columns = conn.pull_kdd99_columns()
    col_len = len(columns) - 1
    dataframe = pd.DataFrame(data=data, columns=columns)
    dataframe = dataframe.iloc[:, :col_len]

    print(type(columns))
    
    # ==========
    # ENCODING
    # ==========
    # https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn

    d = defaultdict(LabelEncoder)

    fit = dataframe.apply(lambda x: d[x.name].fit_transform(x))  # fit is encoded dataframe
    dataset = fit.values  # transform to ndarray

    print(dataset)
    print(dataset.size)

    #TODO: Figure out what the fuck the method actually takes as params
    correlation_matrix = np.zeros(shape=(col_len,col_len))

    for i in range(1, len(columns) - 1):
        for j in range(0, len(columns) - 1):
            correlation_matrix[i, j] = correlation_ratio(dataset[:, i], dataset[:, j])

    print(type(columns))
    correlation_dataframe = pd.DataFrame(data = correlation_matrix, index = columns[:col_len], columns = columns[:col_len])
    print(correlation_dataframe)
    print(correlation_matrix.shape)
    
    correlation_heatmap(correlation_dataframe, mode = mode, save_dir = save_dir, num = num)

def correlation_heatmap(data, mode = "show", save_dir = "figs", num = None):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(data, vmax=1.0, center=0, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .70})
    if mode == "show":
        plt.show()
    elif mode == "save":
        if not path.exists(save_dir):
                makedirs(save_dir)
        filename = "correlation_matrix_" + str(num) + ".png"
        if save_dir:
            filename = save_dir + "/" + filename
        plt.savefig(filename)

if __name__ == "__main__":
    main(argv)