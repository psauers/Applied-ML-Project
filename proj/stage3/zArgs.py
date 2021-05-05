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
#           Handles command line arguments for main.py
#
#
############################################
#
#   Imports
#

# arg handling
import sys
import argparse


#
#   Handle Arguments
#
def get_args():
    parser = argparse.ArgumentParser(description='Classifies data with one of nine ML '
                                                 'classification algorithms.')
    parser.add_argument('classifier',
                        help='The classifier to use. '
                             'Supported classifiers include knn, '
                             'pcpn (Perceptron), svm, sgd '
                             'dt (Decision Tree), rf (Random Forest), '
                             'and adaboost.')
    parser.add_argument('-dataset',
                        help='The dataset filepath to use. Must be a file'
                             'in csv format, with data being on every '
                             'column except the last, and labels being '
                             'in the last column. Other formats are not '
                             'accepted. Default: bank.csv',
                        default='bank.csv')
    # KNN specific arguments
    parser.add_argument('-neighbors',
                        help='Number of neighbors to use '
                             'with KNN'
                             '. Must be an integer. '
                             'Default: 5',
                        type=int,
                        default=5)
    parser.add_argument('-p_num',
                        help='p argument for scikit-learn KNN. Must '
                             'be an integer. Default: 2',
                        type=int,
                        default=2)
    parser.add_argument('-metric',
                        help='Metric to use for scikit-learn '
                             'KNN. Default: minikowski',
                        default='minkowski')

    # perceptron specific arguments
    parser.add_argument('-epochs',
                        help='epochs for perceptron or AdaBoost. '
                             'Default: 40',
                        type=int,
                        default=40)
    parser.add_argument('-eta',
                        help='eta for perceptron or AdaBoost. '
                             'Default: 0.1',
                        type=float,
                        default=0.1)
    parser.add_argument('-random_state',
                        help='random state for perceptron or AdaBoost or DT or RF. '
                             'Default: 1',
                        type=int,
                        default=1)

    # DT specific arguments
    parser.add_argument('-criterion',
                        help='criterion for DT or RF, e.g., gini or entropy. '
                             'Default: gini',
                        default='gini')
    parser.add_argument('-max_depth',
                        help='maximum depth for DT or AdaBoost or RF. '
                             'Default: 4',
                        type=int,
                        default=4)

    # SVM specific arguments
    parser.add_argument('-kernel',
                        help='kernel for SVM. can be rbf or linear. '
                             'Default: linear',
                        default='linear')
    parser.add_argument('-c_num',
                        help='c value for SVM. Default: 10.0',
                        type=float,
                        default=10.0)
    parser.add_argument('-gamma',
                        help='gamma value for SVM. Default: 0.10',
                        type=float,
                        default=0.10)
    parser.add_argument('-n_jobs', help='number of jobs for Bagging.', type=int)

    # SGD specific arguments
    # TODO update when proj_sgd.py has customizable values

    # AdaBoost specific arguments
    # epoch, max depth, learning rate, and random state are handled in other algo types

    # Random Forest specific arguments
    parser.add_argument('-n_est',
                        help='Number of estimators for Random Forest. '
                             'Must be a positive integer. '
                             'Default: 100',
                        type=int,
                        default=100)
    # Criterion, max depth, and random state are handled in other algo types


    args = parser.parse_args()

    return args

# TODO - Add bounds checking function with error handling
# def check_bounds(args):
    # Error out w/ message if any params are out of bounds

