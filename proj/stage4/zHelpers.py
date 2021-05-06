############################################
#
#   Long Term Investment Classifier
#
#   zArgs.py
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
#           getDataXY(filename)
#
#
############################################
#
#   Imports
#

# pandas for data handling
import pandas as pd

# for feature scaling
from sklearn.preprocessing import StandardScaler

import sys
from seaborn.matrix import heatmap
from matplotlib import pyplot as plt


#
#   getDataXY
#
#   Attempts to open the file given and pull a dataset from it
#
#   Assumptions:
#       File is a .csv
#       All rows are instances
#       All cols but the last are features
#       Last col is labels
#
#   Returns:
#       A 2 element tuple.
#       First element is the 2D list of feature data.
#       Second element is the 1D list of labels
#
#   Side Effects:
#       Will quit with message if unable to open file
#
def getDataXY(filename):
    try:
        datasetObj = pd.read_csv((filename + '.csv'), header=0, sep=',')
    except:
        print('Error opening dataset file. Please check the filepath and try again.')
        quit(-1)

    x = datasetObj.iloc[:, :-1]
    y = datasetObj.iloc[:, -1:]

    print('Assuming data is everything on row except last column...')
    print('Assuming labels are in the last column on every row...\n')

    return x, y


#
# do_feature_scaling()
#
# preforms feature scaling on training/testing data and returns them
#
# input: training data, testing daya
# output: training data feature scaled, testing data feature scaled
#
#   - Written by Collin Gros, slightly modified to conform to PEP8
#
def do_feature_scaling(x_train, x_test):
    # perform feature scaling on train and testing sets
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    return x_train_std, x_test_std
