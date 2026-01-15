#!/usr/bin/env python3
"""
creating a stacked bar chart of fruit per person
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar chart for fruit quantities
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))  # 4 types of fruit, 3 people

    cats = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(cats))
    width = 0.5

    # Colors and labels for each row of fruit
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Stack bars
    bottom = np.zeros(len(cats))
    for i in range(fruit.shape[0]):
        plt.bar(x, fruit[i], width, bottom=bottom, label=labels[i], color=colors[i])
        bottom += fruit[i]  # increment bottom for next fruit

    # Axis labels, limits, ticks
    plt.xticks(x, cats)
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # Legend and title
    plt.legend()
    plt.title('Number of Fruit per Person')

    plt.show()