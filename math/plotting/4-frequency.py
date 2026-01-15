#!/usr/bin/env python3
"""
frequency histogram
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    inside the func
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, align='mid', edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    plt.ylim(0, 30)
    plt.yticks(np.arange(0, 31, 5))
    plt.savefig('my_plot.png')
    plt.close()
    plt.show()

frequency()
