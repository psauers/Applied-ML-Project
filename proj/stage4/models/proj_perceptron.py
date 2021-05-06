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
	# input:
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
	def __init__(self, epochs_, eta_, random_state_):
		# init Perceptron NN object
		self.model = skPCPN(max_iter=epochs_, eta0=eta_,
					random_state=random_state_)


	# fit()
	#
	# input X_train, y_train
	# output: the model is trained on the X_train data and y_train
	#	targets (from __init__() if nothign is given)
	#
	#
	# trains the model on the data provided
	#
	def fit(self, X_train_, y_train_):
		# if the user wants to use their set __init__ values
		self.model.fit(X_train_, y_train_)


	# predict()
	#
	# input: X_test (testing data)
	# output: the predicted labels
	#
	def predict(self, X_test_):
		# if the user wants to use their set __init__ values
		y_pred = self.model.predict(X_test_)
		return y_pred





