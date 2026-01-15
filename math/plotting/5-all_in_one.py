#!/usr/bin/env python3
"""
collecting all plots into a single frame
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    creating all the plots and collection them
    """
    y0 = np.arange(0, 11) ** 3
    x0 = np.arange(0, 11)

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # your code here
    fig, axs = plt.subplots(nrows=3, ncols=2)
    
    axs[0, 0].plot(x0, y0, color='red')
    axs[0, 0].set_xlim(0, 10)
    axs[0, 0].set_xticks(np.arange(0, 11 ,2))

    axs[0, 1].scatter(x1, y1, c='magenta')
    axs[0, 1].set_xlabel("Height (in)", fontsize='x-small')
    axs[0, 1].set_ylabel("Weight (lbs)", fontsize='x-small')
    axs[0, 1].set_title("Men's height vs Weight")

    axs[1, 0].plot(x2, y2)
    axs[1, 0].set_xlim(0, 28000)
    axs[1, 0].set_xticks(np.arange(0, 30000, 10000))
    axs[1, 0].set_xlabel("Time (years)", fontsize='x-small')
    axs[1, 0].set_ylabel("Fraction Remaining", fontsize='x-small')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title("Exponential Decay of C-14", fontsize='x-small')

    axs[1, 1].plot(x3, y31, color='red', linestyle='dashed')
    axs[1, 1].plot(x3, y32, color='green')
    axs[1, 1].set_xlabel('Time (years)', fontsize='x-small')
    axs[1, 1].set_ylabel("Fraction Remaining", fontsize='x-small')
    axs[1, 1].set_title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    axs[1, 1].set_xticks(np.arange(0, 20001, 5000))
    axs[1, 1].set_yticks(np.arange(0, 1.1, 0.5))
    axs[1, 1].set_xlim(0, 20000)
    axs[1, 1].set_ylim(0, 1)
    
    gs = axs[2, 0].get_gridspec()
    fig.delaxes(axs[2, 0])
    fig.delaxes(axs[2, 1])
    axs_bottom = fig.add_subplot(gs[2, :])

    bins = np.arange(0, 101, 10)
    axs_bottom.hist(student_grades, bins=bins, edgecolor='black')
    axs_bottom.set_xlim(0, 100)
    axs_bottom.set_xticks(np.arange(0, 101, 10))
    axs_bottom.set_ylim(0, 30)
    axs_bottom.set_yticks(np.arange(0, 31, 10))
    axs_bottom.set_xlabel('Grades', fontsize='x-small')
    axs_bottom.set_ylabel('Number of Students', fontsize='x-small')
    axs_bottom.set_title('Project A', fontsize='x-small')

    fig.suptitle("All in One")

    plt.tight_layout()

    fig.savefig('my_plot.png')


all_in_one()
