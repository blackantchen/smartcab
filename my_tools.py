# -*- encoding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast
import csv

def logfile_descript(csv):
    data = pd.read_csv(os.path.join("logs", csv))

    if len(data) < 10:
    	print "Not enough data collected to create a visualization."
    	print "At least 20 trials are required."
    	return

    # Create additional features
    data['average_reward'] = (data['net_reward'] / (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['reliability_rate'] = (data['success'] * 100).rolling(window=10, center=False).mean()  # compute avg. net reward with window=10
    data['good_actions'] = data['actions'].apply(lambda x: ast.literal_eval(x)[0])
    data['good'] = (data['good_actions'] * 1.0 / \
    	(data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['minor'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[1]) * 1.0 / \
    	(data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['major'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[2]) * 1.0 / \
    	(data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['minor_acc'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[3]) * 1.0 / \
    	(data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['major_acc'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[4]) * 1.0 / \
    	(data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['epsilon'] = data['parameters'].apply(lambda x: ast.literal_eval(x)['e'])
    data['alpha'] = data['parameters'].apply(lambda x: ast.literal_eval(x)['a'])

    training_data = data[data['testing'] == False]
    testing_data = data[data['testing'] == True]

    train_actions = training_data[['trial','good', 'minor','major','minor_acc','major_acc']].dropna()

    print "The average value of actions:"
    print "Total Bad Accidents: {:.2f}".format(1 - train_actions['good'].mean())
    print "major accidents {:.2f}".format(train_actions['major_acc'].mean())
    print "minor accidents: {:.2f}".format(train_actions['minor_acc'].mean())
    print "major violations: {:.2f}".format(train_actions['major'].mean())
    print "minor violations: {:.2f}".format(train_actions['minor'].mean())


def show_epsilon_functions():
    # 显示4种 epsilon 衰减函数图形
    # 1: y1 = a^x
    # 2: y2 = 1/x^2
    # 3: y3 = e^(-ax)
    # 4: y4 = cos(ax)
    # 其中: 0 < a < 1

    # Data for plotting
    # 准备数据：
    # x: range=(0~2*pi),step=0.01, 用于X轴
    # y_*: Y轴数据

    x = np.arange(0.01, 2 * np.pi, 0.01)
    y1 = np.power(0.9, x)
    y2 = 1 / np.square(x)
    y3 = np.exp(-0.9 * x)
    y4 = np.cos(0.9 * x)

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure（） and then ax = fig.add_subplot(111)

    # create a figure and two subplots
    # fig1, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=2, ncols=2, squeeze=False)
    fig1, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    # plot x and y using default line style and color
    ax1.plot(x, y1)
    ax2.plot(x, y2)
    ax3.plot(x, y3)
    ax4.plot(x, y4)

    # set title, x-label, y-lable
    ax1.set(title='y = a^x')
    ax2.set(title='y = 1/(x^2)')
    ax3.set(title='y = exp(-ax)')
    ax4.set(title='y = cos(ax)')

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()

    plt.show()
