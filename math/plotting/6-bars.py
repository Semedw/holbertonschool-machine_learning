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
    print(fruit)

    # your code here
    
    cats = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(cats))
    w = 0.5
    plt.bar(x, fruit[0], w, label='apple', color='red')
    plt.bar(x, fruit[1], w, bottom=fruit[0], label='bananas', color='yellow')
    plt.bar(x, fruit[2], w, bottom=fruit[1]+fruit[0], label='oranges', color='#ff8000')
    plt.bar(x, fruit[3], w, bottom=fruit[2]+fruit[1]+fruit[0], label='peach', color='#ffe5b4')

    plt.xticks(x, cats)
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()

    plt.title('Number of Fruit per Person')

    # plt.savefig('my_plot')
    # plt.close()
    plt.show()