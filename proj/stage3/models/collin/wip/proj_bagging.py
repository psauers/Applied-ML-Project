# collin gros
# 03-28-2021
# cs-487
# team project stage 3
#
#
# proj_bagging.py
#
# this code acts as the bagging module for our team's bank
# ML project.
#
# it can train and test on data from a dataset.
#
# ************************************************
# NOTE: if things look weird, make your tabsize=8
# ************************************************
#

# BaggingClassifier
from sklearn.ensemble import BaggingClassifier as BC
# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as DTC


# class Bagging
#
# the class for the BaggingClassifier from
# scikit learn.
#
# is initialized with training data and training
# labels with the constructor()
#
class Bagging:
	# __init__()
	#
	# input: Training data,
	#	testing data,
	#	training labels,
	#	testing labels,
	#	criterion, max_depth, random_state, number of estimators,
	#	number of jobs
	#
	def __init__(self, X_train_, X_test_, y_train_, y_test_,
			criterion_, max_depth_, random_state_,
			n_estimators_, n_jobs_):
		# initialize DTC
		self.tree = DTC(criterion=criterion_, max_depth=max_depth_,
				random_state=random_state_)

		# initialize BaggingClassifier, with
		# n_estimators being the only adjustable value
		self.bag = BC(base_estimator=self.tree,
				n_estimators=n_estimators_,
				n_jobs=n_jobs_,
				random_state=random_state_)

		self.X_train = X_train_
		self.X_test = X_test_
		self.y_train = y_train_
		self.y_test = y_test_

	# fit()
	#
	# input: (optional) the training data (if not given, will use
	#					givens from __init__()),
	#	(optional) the training labels (if not given, will use
	#					givens from __init__())
	# output: the model is trained on the X_train data and y_train
	#	targets (from __init__() if nothign is given)
	#
	#
	# trains the model on the data provided
	#
	def fit(self, X_train_=None, y_train_=None):
		# if the user wants to use their set __init__ values
		if X_train_ is None:
			X_train_ = self.X_train
		if y_train_ is None:
			y_train_ = self.y_train

		# train DTC tree
		self.tree.fit(X_train_, y_train_)
		# train BC
		self.bag.fit(X_train_, y_train_)


	# predict_tree_score()
	# input: X_test (the testing data from the dataset),
	#		y_test (the testing labels from the dataset)
	# output: the accuracy of the predictions (in percents)
	#
	# this function returns the accuracy of testing the DT model
	# against the test data
	#
	def predict_tree_score(self, X_test, y_test):
		return self.tree.score(X_test, y_test)


	# predict_bag_score()
	# input: X_test (the testing data from the dataset),
	#		y_test (the testing labels from the dataset)
	# output: the accuracy of the predictions (in percents)
	#
	# this function returns the accuracy of testing the bagging model
	# against the test data
	#
	def predict_bag_score(self, X_test, y_test):
		return self.bag.score(X_test, y_test)


	# predict()
	#
	# input: (optional) the testing data from the dataset (if not
	#					given, will use givens
	#					from __init()__),
	# output: the predicted labels
	#
	def predict(self, X_test_=None):
		# if the user wants to use their set __init__ values
		if X_test_ is None:
			X_test_ = self.X_test

		y_pred = self.bag.predict(X_test_)
		return y_pred





















