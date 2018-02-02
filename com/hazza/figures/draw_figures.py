#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@version: V1.0
@author: Hazza Cheng
@contact: hazzacheng@gmail.com
@time: 2018/02/01 
@file: draw_figures.py 
@description: 
@modify: 
"""
import matplotlib.pyplot as plt
import numpy as np


def draw_figure1():
    n_groups = 4
    before = (5110, 245745, 5110, 245745)
    after = (2295, 53235, 2295, 53235)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    rects1 = plt.bar(index, before, bar_width, alpha=opacity, color='b', hatch='/', label='Before')
    rects2 = plt.bar(index + bar_width, after, bar_width, alpha=opacity, color='r', hatch='\\', label='After')

    plt.xlabel('Datasets')
    plt.ylabel('The number of candidate FDs')
    plt.xticks(index + bar_width/2, ('10m_10', '10m_15', '20m_10', '20m_15'))
    plt.ylim(0, 250000)
    plt.legend()

    plt.tight_layout()
    plt.show()


def draw_figure2():
    n_groups = 9
    nums1 = (90, 252, 504, 630, 504, 252, 72, 9, 0)
    nums2 = (90, 70, 24, 1, 1, 1, 1, 3, 0)
    levels = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    plt.xlabel('Levels')
    plt.ylabel('The number of candidate FDs')
    plt.plot(levels, nums1, color='black', marker='o', linestyle='solid', label='Before pruning')
    plt.plot(levels, nums2, color='black', marker='x', linestyle='solid', label='After pruning')
    plt.legend()
    plt.show()


def draw_figure3():
    n_groups = 14
    nums1 = (210, 858, 2860, 6435, 10296, 12012, 10296, 6435, 2860, 858, 156, 13, 0, 0)
    nums2 = (210, 306, 330, 300, 162, 36, 10, 1, 1, 3, 2, 4, 0, 0)
    levels = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    plt.xlabel('Levels')
    plt.ylabel('The number of candidate FDs')
    plt.plot(levels, nums1, color='black', marker='o', linestyle='solid', label='Before pruning')
    plt.plot(levels, nums2, color='black', marker='x', linestyle='solid', label='After pruning')
    plt.legend()
    plt.show()


def draw_figure4():
    n_groups = 9
    nums1 = (90, 252, 504, 630, 504, 252, 72, 9, 0)
    nums2 = (90, 87, 48, 11, 4, 1, 3, 3, 0)
    levels = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    plt.xlabel('Levels')
    plt.ylabel('The number of candidate FDs')
    plt.plot(levels, nums1, color='black', marker='o', linestyle='solid', label='Before pruning')
    plt.plot(levels, nums2, color='black', marker='x', linestyle='solid', label='After pruning')
    plt.legend()
    plt.show()


def draw_figure5():
    n_groups = 14
    nums1 = (210, 858, 2860, 6435, 10296, 12012, 10296, 6435, 2860, 858, 156, 13, 0, 0)
    nums2 = (210, 290, 432, 387, 75, 6, 22, 19, 4, 2, 5, 4, 0, 0)
    levels = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    plt.xlabel('Levels')
    plt.ylabel('The number of candidate FDs')
    plt.plot(levels, nums1, color='black', marker='o', linestyle='solid', label='Before pruning')
    plt.plot(levels, nums2, color='black', marker='x', linestyle='solid', label='After pruning')
    plt.legend()
    plt.show()


def draw_figure6():
    n_groups = 9
    nums = (0.16, 0.35, 0.69, 0.81, 1.01, 1.21, 1.18, 1.35, 1.43)
    levels = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    plt.xlabel('Levels')
    plt.ylabel('The response time of verifying a candidate FDs/s')
    plt.plot(levels, nums, color='black', marker='o', linestyle='solid')
    plt.show()


def draw_figure7():
    n_groups = 14
    nums = (0.18, 0.33, 0.67, 0.72, 0.83, 0.92, 0.95, 1.03, 1.31, 1.20, 1.41, 1.52, 1.59, 1.69)
    levels = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    plt.xlabel('Levels')
    plt.ylabel('The response time of verifying a candidate FDs/s')
    plt.plot(levels, nums, color='black', marker='o', linestyle='solid')
    plt.show()


def draw_figure8():
    n_groups = 9
    nums = (0.27, 0.47, 0.80, 0.88, 1.08, 1.28, 1.35, 1.57, 1.69)
    levels = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    plt.xlabel('Levels')
    plt.ylabel('The response time of verifying a candidate FDs/s')
    plt.plot(levels, nums, color='black', marker='o', linestyle='solid')
    plt.show()


def draw_figure9():
    n_groups = 14
    nums = (0.24, 0.44, 0.78, 0.86, 0.99, 1.17, 1.27, 1.41, 1.63, 1.77, 1.84, 1.91, 2.01, 2.12)
    levels = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    plt.xlabel('Levels')
    plt.ylabel('The response time of verifying a candidate FDs/s')
    plt.plot(levels, nums, color='black', marker='o', linestyle='solid')
    plt.show()


def draw_figure10():
    time1 = (1.7, 2.9, 9.8, 18)
    time2 = (32, 58, 334, 623)
    time3 = (1.7, 2.9, 15, 28)
    time4 = (4.3, 6.2, 90, 115)
    x_labels = ('10m_10', '10m_15', '20m_10', '20m_15')

    pos = list(range(len(time1)))
    bar_width = 0.2

    opacity = 0.4
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    b1 = ax.bar(pos, time1, bar_width, alpha=opacity, color='black', hatch='o', label='FastMFDs')
    b2 = ax.bar([p + bar_width for p in pos], time2, bar_width, alpha=opacity, color='green', hatch='x', label='FDPar_Discover')
    b3 = ax.bar([p + bar_width * 2 for p in pos], time3, bar_width, alpha=opacity, color='red', hatch='/', label='FastMFDs_DF')
    b4 = ax.bar([p + bar_width * 3 for p in pos], time4, bar_width, alpha=opacity, color='blue', hatch='\\', label='FastMFDs_RDD')

    b5 = ax2.bar(pos, time1, bar_width, alpha=opacity, color='black', hatch='o')
    b6 = ax2.bar([p + bar_width for p in pos], time2, bar_width, alpha=opacity, color='green', hatch='x')
    b7 = ax2.bar([p + bar_width * 2 for p in pos], time3, bar_width, alpha=opacity, color='red', hatch='/')
    b8 = ax2.bar([p + bar_width * 3 for p in pos], time4, bar_width, alpha=opacity, color='blue', hatch='\\')

    ax.set_xticks([p + 1.5 * bar_width for p in pos])
    ax2.set_xticks([p + 1.5 * bar_width for p in pos])
    ax2.set_xticklabels(x_labels)

    plt.xlabel('Datasets')
    plt.ylabel('Response Time/minute')

    ax.set_ylim(300, 650)
    ax2.set_ylim(0, 120)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # for x, y in zip(pos, time1):
    #     ax2.text(x, y + 0.55, '%.1f' % y, ha='center', va='top')
    add_labels(b1, ax2)
    add_labels(b2, ax2)
    add_labels(b3, ax2)
    add_labels(b4, ax2)
    add_labels(b5, ax)
    add_labels(b6, ax)
    add_labels(b7, ax)
    add_labels(b8, ax)

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax.legend(loc='upper left')
    plt.show()


def add_labels(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom', fontsize=8)


def draw_figure11():
    nums = (4.2, 3.2, 2.8, 2, 1.7)
    sites = ('3', '4', '5', '6', '7')
    plt.xlabel('Sites')
    plt.ylabel('Response Time/minute')
    plt.plot(sites, nums, color='black', marker='o', linestyle='solid')
    plt.show()


def draw_figure12():
    nums = (21.5, 18.2, 15, 12.4, 9.8)
    sites = ('3', '4', '5', '6', '7')
    plt.xlabel('Sites')
    plt.ylabel('Response Time/minute')
    plt.plot(sites, nums, color='black', marker='o', linestyle='solid')
    plt.show()


def draw_figure13():
    nums = (6.2, 5.3, 4.6, 3.5, 2.9)
    sites = ('3', '4', '5', '6', '7')
    plt.xlabel('Sites')
    plt.ylabel('Response Time/minute')
    plt.plot(sites, nums, color='black', marker='o', linestyle='solid')
    plt.show()


def draw_figure14():
    nums = (39.2, 33.1, 27.2, 23.3, 18)
    sites = ('3', '4', '5', '6', '7')
    plt.xlabel('Sites')
    plt.ylabel('Response Time/minute')
    plt.plot(sites, nums, color='black', marker='o', linestyle='solid')
    plt.show()


def draw_figure15():
    nums = (1.7, 3.2, 1.7, 32)
    x_labels = ('FastMFDs', 'FDPar_Discover', 'FastMFDs_DF', 'FastMFDs_RDD')
    pos = list(range(len(nums)))
    bar_width = 0.2
    opacity = 0.4
    rects = plt.bar(pos, nums, bar_width, alpha=opacity, color='black', tick_label=x_labels)
    plt.xlabel('Algorithms')
    plt.ylabel('Response Time/minute')
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom', fontsize=8)
    plt.show()


def draw_figure16():
    nums = (2.9, 58, 2.9, 6.2)
    x_labels = ('FastMFDs', 'FDPar_Discover', 'FastMFDs_DF', 'FastMFDs_RDD')
    pos = list(range(len(nums)))
    bar_width = 0.2
    opacity = 0.4
    rects = plt.bar(pos, nums, bar_width, alpha=opacity, color='black', tick_label=x_labels)
    plt.xlabel('Algorithms')
    plt.ylabel('Response Time/minute')
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom', fontsize=8)
    plt.show()


def draw_figure17():
    nums = (9.8, 334, 15, 90)
    x_labels = ('FastMFDs', 'FDPar_Discover', 'FastMFDs_DF', 'FastMFDs_RDD')
    pos = list(range(len(nums)))
    bar_width = 0.2
    opacity = 0.4
    rects = plt.bar(pos, nums, bar_width, alpha=opacity, color='black', tick_label=x_labels)
    plt.xlabel('Algorithms')
    plt.ylabel('Response Time/minute')
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom', fontsize=8)
    plt.show()


def draw_figure18():
    nums = (18, 623, 28, 115)
    x_labels = ('FastMFDs', 'FDPar_Discover', 'FastMFDs_DF', 'FastMFDs_RDD')
    pos = list(range(len(nums)))
    bar_width = 0.2
    opacity = 0.4
    rects = plt.bar(pos, nums, bar_width, alpha=opacity, color='black', tick_label=x_labels)
    plt.xlabel('Algorithms')
    plt.ylabel('Response Time/minute')
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom', fontsize=8)
    plt.show()


if __name__ == '__main__':
    # draw_figure1()
    # draw_figure2()
    # draw_figure3()
    # draw_figure4()
    # draw_figure5()
    # draw_figure6()
    # draw_figure7()
    # draw_figure8()
    # draw_figure9()
    # draw_figure11()
    # draw_figure12()
    # draw_figure13()
    # draw_figure14()
    # draw_figure10()
    # draw_figure15()
    # draw_figure16()
    # draw_figure17()
    draw_figure18()
