# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import matplotlib.pyplot as plt
import random

'''
Plotter.py instructions:

plot_2D_multiline plots multiple lines on the same graph. Just pass it a list of lists as the first argument, 
all of which will be interpreted as y-values (instead of xy pairs, or something else entirely like plt.plot() 
normally does.

plot_3D will take a list of either two or three lists as its first argument. It will generate the Z axis as 
the datapoint index if you only pass it two lists. Otherwise if you pass three lists, it will interpret the 
lists as (x,y,z) tuples.

In other words, if you have two sets of a certain number of data points with only one output value, use the 
first function. If you have a certain number of datapoints with (x,y) output values, use the second function.

All the lists have to be the same length, though.
(I can still generate the figures but if anyone else just wants to use them I wanted to make sure the instructions 
were here.)
I will add the bar graph portion in the morning.

But I think that might actually just be native to plt without any need for additional interpretation.

Also, if you want to generate a bunch of figures during a loop without having to save and close the figure each 
time, pass savefile = (something), mode = “save” and it’ll save the file. Just make sure you generate a unique name.
'''

def plot_3D(axes_data, x_label = "X", y_label = "Y", z_label = "Z", 
            savefile = None, mode = "show", create_z = True, title = None):
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
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.title(title)

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=-35)

    if mode.lower() == "save" and savefile:
        plt.savefig(savefile)
    else:
        plt.show()

def plot_2D_multiline(axes_data, x_label = "X", y_label = "Y", 
            savefile = None, mode = "show", create_x = True, title = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Xs = []
    for data in axes_data:
        x = list(range(1, len(data)+ 1))
        Xs.append(x)
    for X, Y in zip(Xs, axes_data):
        plt.plot(X, Y)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)
    
    if mode.lower() == "save" and savefile:
        plt.savefig(savefile)
    else:
        plt.show()

def bar_graph(categories_values_lists, category_names = None, data_series_names = None, 
                x_label = "X", y_label = "Y", mode = "show", title = None):
    
    n_groups = len(categories_values_lists[0])
    n_bars = len(categories_values_lists)
    if not data_series_names:
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        data_series_names = []
        for i in range(n_bars):
            k = int(np.ceil((i + 1)/26))
            data_series_names.append(letters[i % 26] * (k))
    
    if not category_names:
        category_names = list(range(1, n_groups + 1))
    
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    print(category_names, data_series_names)

    for i, series_name in enumerate(data_series_names):
        p = bar_width * i + index
        rect = plt.bar(p, categories_values_lists[i], bar_width,
            alpha=opacity,
            color = colors[i],
            label = category_names[i])
        autolabel(rect, ax)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.xticks(index + bar_width, category_names)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if mode.lower() == "save" and savefile:
        plt.savefig(savefile)
    else:
        plt.show()

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom', rotation = 90)


def test():
    n = 10
    X = [random.randrange(25) for i in range(n)]
    Y = [random.randrange(25) for i in range(n)]
    Z = [random.randrange(25) for i in range(n)]
    #plot_3D([X, Y, Z], savefile = "test3d.png")

    #plot_2D_multiline([X, Y, Z], savefile = "test3d.png")

    bar_graph([X, Y, Z])

if __name__ == "__main__":
    test()