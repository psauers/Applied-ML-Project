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
from models.proj_svm import SVM as svm
from models.proj_adaboost import Boost as boost
from models.proj_sgd import SGD as sgd


# File handler
from zHelpers import getDataXY
from zHelpers import do_feature_scaling
import os



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

    # split newly aquired X and y into seperate datasets for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=1,
                                                        stratify=y)

    #
    #   Data Optimizing
    #

    # TODO:
    #   convert string features like job type into enumerated lists

    # Feature Scaling
    #\/ needs the strings converted first
    # x_train_std, x_test_std = do_feature_scaling(x_train, x_test)

    #
    #   Classifier type based actions
    #
    if args.classifier.lower() == 'knn':
        print('KNN selected')
        #grab args
        n = 3
        if args.neighbors:
            n = args.neighbors
        model = knn(n)
        # train
        model.fit(x_train, y_train)
        # test
        model.predict(x_test)
        # report
        print("KNN score: " + str(model.score(x_test, y_test)))
    elif args.classifier.lower() == 'pcpn':
        print("Perceptron selected")

        # train

        # test

        # report

    elif args.classifier.lower() == 'svm':
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
        model.predict(x_test)
        # report
        print("SVM score: " + str(model.score(x_test, y_test)))
    elif args.classifier.lower() == 'sgd':
        print("SGD selected")
        model = sgd()
        # train
        model.fit(x_train, y_train)
        # test
        model.run(x_test)
        # report
        print("SGD score: " + str(model.score(x_test, y_test)))

    elif args.classifier.lower() == 'dt':
        print("Decision Tree selected")
        criterion = "gini"
        max_depth = 4
        if args.criterion:
            criterion = args.criterion
        if args.max_depth:
            max_depth = args.max_depth
        dt = decisiontree(criterion, max_depth)
        # train
        dt.fit(x_train, y_train)
        # test
        dt.predict(x_test)
        # report
        print("Decision tree score: " + str(dt.score(x_test, y_test)))

    elif args.classifier.lower() == 'rf':
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
        rf = randomforest(n_estimators, criterion, max_depth)
        # train
        rf.fit(x_train, y_train)
        rf.tree_fit(x_train, y_train)
        # test
        rf.predict(x_test)
        # report
        print("Random forest score: " + str(rf.score(x_test, y_test)))
        print("Tree score: " + str(rf.tree_score(x_test, y_test)))

    elif args.classifier.lower() == 'adaboost':
        print("AdaBoost selected")
        n_estimators = 500
        if (args.n_est):
            n_estimators = args.n_est
        model = boost(n_est=n_estimators)
        # train
        model.fit(x_train, y_train)
        # test
        model.predict(x_test)
        # report
        print("Adaboost score: " + str(model.score(x_test, y_test)))
        print("Tree score: " + str(model.tree_score(x_test, y_test)))

    else:
        print('Error: Invalid classifier type entered.')
        print('Valid classifier types:')
        print('knn, pcpn, svm, sgd, dt, rf, and adaboost')
        quit(-2)

