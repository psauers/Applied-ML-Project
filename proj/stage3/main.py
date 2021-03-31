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
#   Arguments:
#       classifier - string
#       dataset - string - filename or filepath (can handle just name or filepath with .csv
#
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
import pandas

# for timing
import time

# for splitting data into training and testing data
from sklearn.model_selection import train_test_split

# for feature scaling
from sklearn.preprocessing import StandardScaler

# accuracy_score in score()
from sklearn.metrics import accuracy_score





#
#   Main
#
if __name__ == "__main__":
    #
    #   Handle Args
    #
    args = get_args()


    #
    #   Get Dataset From File
    #



    # Welcome Message
    print("Machine Learning Mega Classifier of Bigness!")
    print("\tThank you for using our software. Copyright 2021 - Team Quokking")