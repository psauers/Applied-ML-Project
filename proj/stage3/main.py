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

# accuracy_score in score()
from sklearn.metrics import accuracy_score

# Custom imports

# File handler
from zHelpers import getDataXY
from zHelpers import do_feature_scaling



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
    # print(y)

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

        # train

        # test

        # report

    elif args.classifier.lower() == 'pcpn':
        print("Perceptron selected")

        # train

        # test

        # report

    elif args.classifier.lower() == 'svm':
        print("SVM selected")

        # train

        # test

        # report

    elif args.classifier.lower() == 'sgd':
        print("SGD selected")

        # train

        # test

        # report

    elif args.classifier.lower() == 'dt':
        print("Decision Tree selected")

        # train

        # test

        # report

    elif args.classifier.lower() == 'rf':
        print("Random Forest selected")

        # train

        # test

        # report

    elif args.classifier.lower() == 'adaboost':
        print("AdaBoost selected")

        # train

        # test

        # report

    else:
        print('Error: Invalid classifier type entered.')
        print('Valid classifier types:')
        print('knn, pcpn, svm, sgd, dt, rf, and adaboost')
        quit(-2)

