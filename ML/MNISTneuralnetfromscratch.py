
import numpy as np
import math
from mlxtend.data import mnist_data
import matplotlib.pyplot as plt

def run():
    iterations = 3000
    learning_rate = .01

    X, y = mnist_data()  # load mnist dataset into X and y
    X = feature_scale(X)
    labels = one_hot(y)  # one-hot encode our labels

    X, labels = shuffle(X, labels)  # shuffle our dataset (originally it's sorted by number)

    # partition our dataset into test and train, to evaluate the performance of our neural net on data it hasn't seen
    X_train = X[:500]
    labels_train = labels[:500]
    X = X[500:]
    labels = labels[500:]

    #  define our neural network layers as column vectors (each element is a perceptron)
    input = np.zeros((784, 1))
    hidden1 = np.zeros((200, 1))
    output = np.zeros((10, 1))

    # weight_x = matrix of weights going INTO layer x
    weight_2 = np.random.randn(len(hidden1), len(input))
    weight_3 = np.random.randn(len(output), len(hidden1))
    # bias nodes have the shape of the layer the feed into
    h1_bias = np.ones(hidden1.shape)
    out_bias = np.ones(output.shape)

    m = len(X)
    for h in range(0, iterations):
        # initializing our gradient matrices and MSE as a measure of our error
        h1_bias_grad = np.zeros(h1_bias.shape)
        out_bias_grad = np.zeros(out_bias.shape)
        weight_2_grad = np.zeros(weight_2.shape)
        weight_3_grad = np.zeros(weight_3.shape)
        MSE = np.zeros(output.shape)

        for i in range(0, m):

            # forward propagation
            input = X[i]
            input = input.reshape(len(X[0]), 1)  # need to reshape it so it plays nice with numpy matrix operations
            z2 = (weight_2 @ input) + h1_bias  # z is the weighted sum of the inputs to a perceptron
            hidden1 = tanh_vect(z2)  # tanh is our activation function for the hidden layer (-1 to 1)
            z3 = (weight_3 @ hidden1) + out_bias
            output = sigmoid_vect(z3)  #  output activation function is sigmoid

            # back propagation
            cur_label = labels[i].reshape(len(labels[i]), 1)  # reshaping to play nice with matrix ops
            MSE += .5 * ((output - cur_label) ** 2)  # measure MSE so we can see if error decreases with iterations

            # error functions
            output_err = np.multiply(output - cur_label, sigmoid_deriv(output))
            hidden1_err = np.multiply(weight_3.transpose() @ output_err, tanh_deriv(hidden1))

            # calculating the gradient of each weight and bias using the calculated error
            weight_2_grad += hidden1_err @ input.transpose()  # weights gradients are activations in * error out
            weight_3_grad += output_err @ hidden1.transpose()
            h1_bias_grad += hidden1_err  # biases gradients are equal to the error of their layer
            out_bias_grad += output_err

        MSE = MSE / m
        # average the gradient for all of our training samples (batch gradient descent)
        weight_2_grad = weight_2_grad / m
        weight_3_grad = weight_3_grad / m
        h1_bias_grad = h1_bias_grad / m
        out_bias_grad = out_bias_grad / m

        # update our weights using gradient descent
        weight_2 = weight_2 - (learning_rate * weight_2_grad)
        weight_3 = weight_3 - (learning_rate * weight_3_grad)
        h1_bias = h1_bias - (learning_rate * h1_bias_grad)
        out_bias = out_bias - (learning_rate * out_bias_grad)

        if h % 100 == 0:  # measuring our error every 100 iterations, helps to see nonconvergence without waiting
            print("MSE vector for iteration: " + str(h))
            print(MSE)
            print("============================================")

    # Test our performance on unseen data
    print("============================================")
    print("============================================")
    print("Label vs prediction on training data:")
    print("============================================")

    # this entire ugly thing makes an array of the true label of a input, and appends the neural nets prediction
    predictions = np.empty(shape=(len(labels_train), 1), dtype=object)
    predictions[0] = "hi"
    for i in range(0, len(X_train)):
        input = X_train[i]
        input = input.reshape(len(X_train[0]), 1)

        z2 = (weight_2 @ input) + h1_bias
        hidden1 = sigmoid_vect(z2)
        z3 = (weight_3 @ hidden1) + out_bias
        output = sigmoid_vect(z3)
        # round our output vector to 3 decimal places and reshape it to (10,)
        predictions[i] = np.array2string(output.reshape(1, len(output)), precision=2)

    pred_vs_label_array = np.empty((len(labels_train), 2), dtype=object)
    for i in range(0, len(labels_train)):
        pred_vs_label_array[i, 0] = np.array2string(labels_train[i], precision=1)
        pred_vs_label_array[i, 1] = np.array2string(predictions[i])

    print(pred_vs_label_array)


def sigmoid_vect(z):
    act = np.zeros(z.shape)
    for i in range(0, len(z)):
        act[i] = 1.0/(1.0+np.exp(-z[i]))
    return act


def tanh_vect(z):
    act = np.zeros(z.shape)
    for i in range(0, len(z)):
        act[i] = math.tanh(z[i])
    return act


def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()


def one_hot(y):
    labels = np.zeros((len(y), 10))
    for i in range(0, len(labels)):
        a = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        a[y[i]] = 1
        labels[i] = a
    return labels


def sigmoid_deriv(x):
    return np.multiply(x, 1-x)


def tanh_deriv(x):
    return 1 - x ** 2


def feature_scale(data):
    return data / 255


# shuffles the data and labels so they still correspond


def shuffle(data, labels):
    # Generate the permutation index array.
    permutation = np.random.permutation(data.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = data[permutation]
    shuffled_b = labels[permutation]
    return shuffled_a, shuffled_b


if __name__ == '__main__':
    run()