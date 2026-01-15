#!/usr/bin/env python3
"""
creating multiple colored bars
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    inside the function
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    
    people = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(people))
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    fruit_colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    w = 0.5
    bottom = np.zeros(len(people))
    
    for i in range(len(fruit)):
        plt.bar(x, fruit[i], width=w, bottom=bottom, color=fruit_colors[i],
                label=fruit_names[i])
        bottom += fruit[i]

    plt.xticks(x, people)
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()

    plt.title('Number of Fruit per Person')

    # plt.savefig('my_plot')
    # plt.close()
    plt.show()

bars()