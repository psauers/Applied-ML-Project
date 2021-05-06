# collin gros
# 04-26-2021
# cs-487
# project
#
# proj_adaline.py
#
# this file contains the code to use Adaline with training/testing
# data for our team project.
#
# HEAVILY based off of the lecture notes from Tuan Le
# (CS-487 lecture notes (5))
#
# and https://vitalflux.com/adaline-explained-with-python-example/
#
# *********************************************
# NOTE: if viewing this looks off, change your
#	tabsize to 8
# *********************************************
#

import numpy as np


# this class implements the adaline algorithm
class Adaline:
	# eta (float): learning rate [0.0, 1.0]
	# n_iter (int): passes over the training
	#		dataset
	# random_state (int): random number generator
	#			seed for random weight
	#			initialization
	def __init__(self, n_iter_, eta_, random_state_):
		# learning rate
		self.eta = eta_
		# iterations
		self.n_iter = n_iter_
		# seed for random weight
		self.random_state = random_state_
		# weights
		self.w_ = None



	def net_input(self, X):
		# return dot product between data(X) and weights
		return np.dot(X, self.w_[1:]) + self.w_[0]


	def predict(self, X):
		# decide between 1 and -1 if input is >=0
		y_pred = np.where(self.net_input(X) >= 0.0, 1, 0)
		return y_pred


	def fit(self, X, y):
		# init random generator
		rgen = np.random.RandomState(
				self.random_state)
		# init weights (from size of data)
		self.w_ = rgen.normal(loc=0.0,
				scale=0.01, size=1 +
				X.shape[1])
		self.cost_ = []


	def activiation(self, X):
		return X

		# do the specified number of iterations
		for _ in range(self.n_iter):
			output = self.net_input(X)
			errors = (y - output)
			# update weights and add cost
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()

			#cost = (errors**2).sum() / 2.0
			#self.cost_.append(cost)


		return self























