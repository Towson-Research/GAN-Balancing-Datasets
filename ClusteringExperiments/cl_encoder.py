from sklearn import preprocessing

def vecs_from_CSV(filename, has_labels=True, label_last=True, separator=","):
    '''
        Converts a CSV to X, y vector lists, where X is the data and
        y is the label. Each line is assumed to be a vector.
        Requires a filename.
        Optional:
            has_labels: True if there are labels in the file. Default is true.
            label_last: True if the label is the last entry in a line. Default is true.
            separator: defaults to comma, otherwise supply.
        returns X and y as lists.
    '''
    X = []
    y = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split(separator)
            if has_labels:
                if label_last:
                    X.append(line[:-1])
                    y.append(line[-1])
                else:
                    X.append(line[1:])
                    y.append(line[1])
    return X, y

def encode(X, y):
    '''
        Uses Scikitlearn's preprocessing to numerically encode
        symbolic data.
        Returns the encoded vector lists X, y.
        Note: this does not normalize.
    '''

    le = preprocessing.LabelEncoder()

    X = [le.fit(x) and le.transform(x) for x in X]

    if y:
        le_y = preprocessing.LabelEncoder()
        le_y.fit(y)
        y = le_y.transform(y)

    return X, y
