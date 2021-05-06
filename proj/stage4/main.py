############################################
#
#   Long Term Investment Classifier
#
#   main.py
#
#   Group Project Spring 2021
#
#
#   Authors:
#       Collin Gros
#       Zac Holt
#       Nha Quynh Nguyen
#       Phillip Sauers
#
#
#   Desc:
#           Applies one of a choice of nine (9) machine learning algorithms to
#       a dataset. Intended dataset contains consumer data from a Portuguese bank over a five (5)
#       year period following the financial crisis of 2008.
#
#
#   Required Arguments:
#       classifier - string
#       dataset - string - filename or filepath (can handle just name or filepath with .csv
#
#   --help for more info
#
############################################
#
#   Imports
#

# arg handling
import sys
import argparse
from zArgs import get_args

# pandas for data handling
import pandas as pd

# for timing
import time

# for splitting data into training and testing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# accuracy_score in score()
from sklearn.metrics import accuracy_score

# Custom imports
from models.myknn import knn
from models.myrandomforest import randomforest
from models.mydecisiontree import decisiontree

from models.proj_perceptron import Perceptron
from models.proj_adaline import Adaline
from models.proj_bagging import Bagging

from models.proj_svm import SVM as svm
from models.proj_adaboost import Boost as boost
from models.proj_sgd import SGD as sgd

# graphs
import matplotlib.pyplot as plt

# File handler
from zHelpers import getDataXY
from zHelpers import do_feature_scaling
import os
import numpy as np

def do_pcpn(args):
    print("Perceptron selected")

    # setup model values
    random_state = 1
    if args.random_state:
        random_state = args.random_state
    epochs = 10
    if args.epochs:
        epochs = args.epochs
    eta = 0.01
    if args.eta:
        eta = args.eta

    model = Perceptron(epochs, eta, random_state)

    #train
    model.fit(x_train, y_train)

    #test
    start = time.time()
    y_pred = model.predict(x_test)
    stop = time.time()
    elapsed = stop - start

    accuracy = accuracy_score(y_pred, y_test)

    return elapsed, accuracy


def do_adaline(args):
    print("Adaline selected")
    # setup model values
    random_state = 1
    if args.random_state:
        random_state = args.random_state
    epochs = 10
    if args.epochs:
        epochs = args.epochs
    eta = 0.01
    if args.eta:
        eta = args.eta

    model = Adaline(epochs, eta, random_state)

    #train
    model.fit(x_train, y_train)

    #test
    start = time.time()
    y_pred = model.predict(x_test)
    stop = time.time()
    elapsed = stop - start

    accuracy = accuracy_score(y_pred, y_test)

    return elapsed, accuracy


def do_bagging(args):
    print("Bagging selected")
    # setup model values
    criterion = 'entropy'
    if args.criterion:
        criterion = args.criterion
    max_depth = 4
    if args.max_depth:
        max_depth = args.max_depth
    random_state = 1
    if args.random_state:
        random_state = args.random_state
    n_estimators = 500
    if args.n_est:
        n_estimators = args.n_est
    n_jobs = 2
    if args.n_jobs:
        n_jobs = args.n_jobs


    model = Bagging(criterion, max_depth, random_state, n_estimators,
                               n_jobs)

    #train
    model.fit(x_train, y_train)

    #test
    start = time.time()
    y_pred = model.predict(x_test)
    stop = time.time()
    elapsed = stop - start

    accuracy = accuracy_score(y_test, y_pred)

    return elapsed, accuracy


def do_knn(args):
    print('KNN selected')
    #grab args
    n = 3
    if args.neighbors:
        n = args.neighbors
    model = knn(n)
    # train
    model.fit(x_train, y_train)

    # test
    start = time.time()
    model.predict(x_test)
    stop = time.time()
    elapsed = stop - start

    accuracy = model.score(x_test, y_test)

    return elapsed, accuracy


def do_svm(args):
    print("SVM selected")
    kernel = "linear"
    c = 100
    gamma = 1.0
    if args.kernel:
        kernel = args.kernel
    if args.c_num:
        c = args.c_num
    if args.gamma:
        gamma = args.gamma
    model = svm(kernel, c, gamma)
    # train
    model.fit(x_train, y_train)
    # test
    start = time.time()
    model.predict(x_test)
    stop = time.time()
    elapsed = stop - start

    accuracy = model.score(x_test, y_test)

    return elapsed, accuracy

def do_sgd(args):
    print("SGD selected")
    model = sgd()
    # train
    model.fit(x_train, y_train)
    # test
    start = time.time()
    model.run(x_test)
    stop = time.time()
    elapsed = stop - start

    accuracy = model.score(x_test, y_test)

    return elapsed, accuracy


def do_dt(args):
    print("Decision Tree selected")
    criterion = "gini"
    max_depth = 4
    if args.criterion:
        criterion = args.criterion
    if args.max_depth:
        max_depth = args.max_depth
    model = decisiontree(criterion, max_depth)
    # train
    model.fit(x_train, y_train)
    # test
    start = time.time()
    model.predict(x_test)
    stop = time.time()
    elapsed = stop - start

    accuracy = model.score(x_test, y_test)

    return elapsed, accuracy


def do_rf(args):
    print("Random Forest selected")
    n_estimators = 100
    criterion = "gini"
    max_depth = 4
    if args.n_est:
        n_estimators = args.n_est
    if args.criterion:
        criterion = args.criterion
    if args.max_depth:
        max_depth = args.max_depth
    model = randomforest(n_estimators, criterion, max_depth)
    # train
    model.fit(x_train, y_train)
    model.tree_fit(x_train, y_train)
    # test
    start = time.time()
    model.predict(x_test)
    stop = time.time()
    elapsed = stop - start

    accuracy = model.score(x_test, y_test)

    return elapsed, accuracy


def do_adaboost(args):
    print("AdaBoost selected")
    n_estimators = 500
    if (args.n_est):
        n_estimators = args.n_est
    model = boost(n_est=n_estimators)
    # train
    model.fit(x_train, y_train)
    # test
    start = time.time()
    model.predict(x_test)
    stop = time.time()
    elapsed = stop - start

    accuracy = model.score(x_test, y_test)

    return elapsed, accuracy


def test_all(models, args):
    time_l = []
    accuracy_l = []
    for key in models:
        elapsed, accuracy = models[key](args)
        time_l.append(elapsed)
        accuracy_l.append(accuracy)

    print(time_l)
    print(accuracy_l)

    return time_l, accuracy_l


def graph(x, y, x_lab, y_lab, color_):
    # plot the data vs method used

    # set labels of axis/title
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title('{0} vs {1}'.format(x_lab, y_lab))

    # plot data
    plt.bar(x, y, color=color_)

    # replace spaces in Y label name with underscores for
    # the filename
    plt.savefig('{0}-{1}.png'.format(x_lab, y_lab))

    # write
    plt.show()
    # erase
    plt.clf()



#
#   Main
#
if __name__ == "__main__":
    # Welcome Message
    print("Machine Learning Mega Classifier of Bigness Bank Edition!")
    print("\tThank you for using our software. Copyright 2021 - Team Quokking (Google It!)\n")

    #
    #   Handle Args
    #
    args = get_args()

    #
    #   Get Dataset From File
    #
    filename = args.dataset.replace('.csv', '')     # try to strip filetype, eliminates possible duplicate

    x, y = getDataXY(filename)  # Quits w/ message if unable to open

    # transform non-numerical labels to numerical labels
    le = LabelEncoder()
    columns = ["job","marital","education","default","housing","loan","contact","month","poutcome"] #columns to pass to label encoder
    for c in columns:
        x[c] = LabelEncoder().fit_transform(x[c])

    y["deposit"] = LabelEncoder().fit_transform(y["deposit"])

    print(pd.Series.sort_values(x.corrwith(y['deposit'])))
    print(x.columns)
    x = x.drop(columns=["education","balance","marital","job","age","default"])
    

    # split newly aquired X and y into seperate datasets for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=1,
                                                        stratify=y)

    #
    #   Classifier type based actions
    #
    if args.classifier.lower() == 'pcpn':
        elapsed, accuracy = do_pcpn(args)
    elif args.classifier.lower() == 'adaline':
        elapsed, accuracy = do_adaline(args)
    elif args.classifier.lower() == 'bagging':
        elapsed, accuracy = do_bagging(args)
    elif args.classifier.lower() == 'knn':
        elapsed, accuracy = do_knn(args)
    elif args.classifier.lower() == 'svm':
        elapsed, accuracy = do_svm(args)
    elif args.classifier.lower() == 'sgd':
        elapsed, accuracy = do_sgd(args)
    elif args.classifier.lower() == 'dt':
        elapsed, accuracy = do_dt(args)
    elif args.classifier.lower() == 'rf':
        elapsed, accuracy = do_rf(args)
    elif args.classifier.lower() == 'adaboost':
        elapsed, accuracy = do_adaboost(args)
    # test all models and graph the time and accuracies of them
    elif args.classifier.lower() == 'test_all':
        # all models mapped to their function references
        models = {
            'pcpn':do_pcpn,
            'adaline':do_adaline,
            'bagging':do_bagging,
            'knn':do_knn,
            'svm':do_svm,
            'sgd':do_sgd,
            'dt':do_dt,
            'rf':do_rf,
            'adaboost':do_adaboost
        }
        time_l, accuracy_l = test_all(models, args)

        model_l = list(models.keys())
        graph(model_l, time_l, 'Model', 'Testing_Time', 'green')
        graph(model_l, accuracy_l, 'Model', 'Accuracy', 'blue')


    else:
        print('Error: Invalid classifier type entered.')
        print('Valid classifier types:')
        print('knn, pcpn, adaline, bagging, svm, sgd, dt, rf, adaboost, test_all')
        quit(-2)

