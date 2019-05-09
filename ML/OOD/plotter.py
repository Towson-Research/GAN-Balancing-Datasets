# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import matplotlib.pyplot as plt
import random

def plot_3D(axes_data, X_label = "X", Y_label = "Y", Z_label = "Z", 
            savefile = None, mode = "show", create_Z = True):
    fig = plt.figure()
    if len(axes_data) == 1:
        # Make a 2D plot instead.
        return plot_2D(axes_data[0])
    ax = fig.add_subplot(111, projection='3d')
    X = axes_data[0]
    Y = axes_data[1]
    if len(axes_data) == 3:
        Z = axes_data[2]
    elif len(axes) == 2:
        Z = list(range(min(min(x), min(y)), max(max(x), max(y))))
    ax.plot(axes_data[0], axes_data[1], Z)

    # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlim(min(X), max(X))
    ax.set_ylim(min(Y), max(Y))
    ax.set_zlim(min(Z), max(Z))
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.set_zlabel(Z_label)

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=-35)

    if mode == "save" and savefile:
        plt.savefig(savefile)
    else:
        plt.show()

def plot_2D_multiline(axes_data, X_label = "X", Y_label = "Y", 
            savefile = None, mode = "show", create_X = True):
    fig = plt.figure()
    Xs = []
    for data in axes_data:
        x = list(range(1, len(data)+ 1))
        Xs.append(x)
    for X, Y in zip(Xs, axes_data):
        plt.plot(X, Y)
    
    if mode == "save":
        plt.savefig(savefile)
    else:
        plt.show()

def test():
    n = 100
    X = [random.randrange(25) for i in range(n)]
    Y = [random.randrange(25) for i in range(n)]
    Z = [random.randrange(25) for i in range(n)]
    plot_3D([X, Y, Z], savefile = "test3d.png")

    plot_2D_multiline([X, Y, Z], savefile = "test3d.png")

if __name__ == "__main__":
    test()