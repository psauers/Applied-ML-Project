# collin gros
# 03-28-2021
# cs-487
# team project stage 3
#
#
# proj_perceptron.py
#
# this code acts as the perceptron module for our team's bank
# ML project.
#
# it can train and test on data from a dataset.
#
# ************************************************
# NOTE: if things look weird, make your tabsize=8
# ************************************************
#
# NOTE: you may want to use Precptron with feature scaled
#	data.
#

# for the Perceptron nn
from sklearn.linear_model import Perceptron as skPCPN


# class Perceptron
#
# the class for the Perceptron algorithm fron
# scikit learn.
#
# is initialized with training data and training labels
# with the constructor()
#
class Perceptron:
	# init()
	#
	# input: Training data,
	#	testing data,
	#	training labels,
	#	testing labels,
	#	epochs,
	#	eta (learning rate),
	#	random state,
	#	
	# output: initializes self.nn with the given values,
	#	no output
	#
	#
	# initializes the Perceptron object
	#
	def __init__(self, X_train_, X_test_, y_train_, y_test_,
			epochs_=40, eta_=0.1, random_state_=1):
		# init Perceptron NN object
		self.model = skPCPN(max_iter=epochs_, eta0=eta_,
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

		self.model.fit(X_train_, y_train_)


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

		y_pred = self.model.predict(X_test_)
		return y_pred





