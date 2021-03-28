# collin gros
# 03-28-2021
# cs-487
# testing
#
# NOTE:  *** THIS FILE IS ONLY FOR TESTING MY CODE BEFORE INTEGRATION ***
#
#
# NOTE: if things look weird, make your tabsize=8
#

# arg handling
import sys
import argparse

# the implemented algorithms
import proj_perceptron
#import proj_bagging

# pandas for data handling
import pandas

# using small datasets (iris)
from sklearn import datasets
# for splitting data into training and testing data
from sklearn.model_selection import train_test_split
# for feature scaling
from sklearn.preprocessing import StandardScaler

# accuracy_score in score()
from sklearn.metrics import accuracy_score

# for timing
import time


# return the arguments
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('classifier', help='the classifier to use. '
				'supported classifiers include knn, '
				'pcpn, svm, and dt')
	parser.add_argument('dataset', help='the dataset to use. can be one '
				'of the presets, e.g., iris, digits, '
				'or realistic_sensor_displacement, or can be '
				'a custom file. if it is a custom file, it '
				'must be '
				'in csv format, with data being on every '
				'column except the last, and labels being '
				'in the last column. other formats are not '
				'implemented yet.')
	# KNN specific arguments
	parser.add_argument('-neighbors', help='number of neighbors to use '
				'with KNN'
				'. must be an integer.', type=int)
	parser.add_argument('-p', help='p argument for scikit-learn KNN. must '
				'be an integer.', type=int)
	parser.add_argument('-metric', help='metric to use for scikit-learn '
				'KNN.')

	# perceptron specific arguments
	parser.add_argument('-epochs', help='epochs.', type=int)
	parser.add_argument('-eta', help='eta.', type=float)
	parser.add_argument('-random_state', help='random state.', type=int)

	# DT specific arguments
	parser.add_argument('-criterion', help='criterion, e.g., gini.')
	parser.add_argument('-max_depth', help='maximumd depth.', type=int)

	# SVM specific arguments
	parser.add_argument('-kernel', help='kernel. can be rbf or linear.')
	parser.add_argument('-cnum', help='c value for svm.', type=float)
	parser.add_argument('-gamma', help='gamma value for svm.', type=float)

	# defaults
	parser.add_argument('-defaults', help='can be 1 or 0. 1 will make '
				'all required arguments their default vals.',
				type=int)
	
	args = parser.parse_args()

	if args.defaults == 1:
		# DT
		args.criterion = 'gini'
		args.max_depth = 4
		args.random_state = 1

		# KNN
		args.neighbors = 5
		args.p = 2
		args.metric = 'minkowski'

		# SVM
		args.cnum = 10.0
		args.random_state = 1
		args.gamma = 0.10
		if args.kernel is None:
			args.kernel = 'linear'

		# PCPN
		args.epochs = 40
		args.eta = 0.1

	return args


# prepare_data()
#
# input:
#	filename of general dataset
#	(optional) specific dataset to use
#		can be 'iris'
#	*** NOTE: all data must be a CSV file ***
# output:
#	X, y tuple where X is data and y is the labels
#
def prepare_data(dataset=''):
	X = None
	y = None

	# handle specific datasets
	if dataset == 'iris':
		iris = datasets.load_iris()
		# only using petal length and
		# petal width features
		X = iris.data[:, [2, 3]]
		y = iris.target

	elif dataset == 'digits':
		digits = datasets.load_digits()
		# transform from 8x8 to feature vector of length 64
		X = digits.images.reshape((len(digits.images), -1))
		y = digits.target

	elif dataset == 'realistic_sensor_displacement':
		df = pandas.read_csv('realistic_sensor_displacement/'
				'subject1_ideal.log', sep='\t')
		X = df.iloc[:, :-1]
		# select last column only for activity labels, and flatten
		# it
		y = df.iloc[:, -1:].values.ravel()

	# handle general dataset, specified in dataset
	else:
		print('specified dataset did not match pre-defined datasets.'
			' trying user-specified dataset...')
		try:
			df = pandas.read_csv(dataset)
			print('assuming data is everything on row except'
				' last column...')
			X = df.iloc[:, :-1]
			print('assuming labels are in the last column on '
				'every row...')
			y = df.iloc[:, -1:]
		except:
			print('ERROR: failed handling the dataset: \'{0}\''
				''.format(dataset))
			exit()


	return X, y


# do_feature_scaling()
#
# preforms feature scaling on training/testing data and returns them
#
# input: training data, testing daya
# output: training data feature scaled, testing data feature scaled
#
def do_feature_scaling(X_train, X_test):
	# perform feature scaling on train and testing sets
	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)

	return X_train_std, X_test_std





# get our command-line arguments.
# can access them like so:
#	python3 main.py -k 5
# print(args.k) -> 5
args = get_args()


# extract data into X (data) and y (labels) from the given dataset
X, y = prepare_data(dataset=args.dataset)

# split newly aquired X and y into seperate datasets for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
							random_state=1,
							stratify=y)

# perform feature scaling
X_train_std, X_test_std = do_feature_scaling(X_train, X_test)

# train and test all models
#test_all(X_train, X_test, X_train_std, X_test_std, y_train, y_test)


# use the correct classifier based on args
if args.classifier == 'pcpn':
	# load Perceptron
	# NOTE: using feature scaling here...
	pcpn = proj_perceptron.Perceptron(X_train_std, X_test_std,
					y_train, y_test,
					epochs_=args.epochs,
					eta_=args.eta,
					random_state_=args.random_state)

	# *** TRAIN ***
	begin_t = time.time()
	pcpn.fit()
	end_t = time.time()
	train_t = end_t - begin_t

	print('pcpn training time: {0:.2f}s'
		''.format(train_t))

	# *** TEST ***
	# transform test data
	begin_t = time.time()
	# actually test and get result
	y_pred = pcpn.predict()
	end_t = time.time()
	test_t = end_t - begin_t

	acc = 100 * accuracy_score(y_pred, pcpn.y_test)

	print('pcpn:\t\t\t{0:.2f}%\t{1:.2f}s\n'
		''.format(acc, test_t))

# TODO:
elif args.classifier == 'bag':
	# load
	bag = proj_perceptron.Perceptron(X_train_std, X_test_std,
					y_train, y_test,
					epochs_=args.epochs,
					eta_=args.eta,
					random_state_=args.random_state)

	# *** TRAIN ***
	begin_t = time.time()
	pcpn.fit()
	end_t = time.time()
	train_t = end_t - begin_t

	print('pcpn training time: {0:.2f}s'
		''.format(train_t))

	# *** TEST ***
	# transform test data
	begin_t = time.time()
	# actually test and get result
	y_pred = pcpn.predict()
	end_t = time.time()
	test_t = end_t - begin_t

	acc = 100 * accuracy_score(y_pred, pcpn.y_test)

	print('pcpn:\t\t\t{0:.2f}%\t{1:.2f}s\n'
		''.format(acc, test_t))


