#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

''' implement sa multilayer perceptron classifier 
  also known as deep feedforward networks in deep learning
'''

def relu(X):
	'''implement rectfied linear unit function 
	   Parameters: 
	       X: {array-like, sparse matrix}, shape (n_samples, n_features)
	   Return:
	       X_new, transformed data
	'''
	np.clip(X, 0, np.finfo(X.dtype).max, out=X)
	return X


def relu_derivative(X):
	'''derivatives to the rectfied linear unit function 
	 Parameters: 
	    X: {array-like, sparse matrix}, shape (n_samples, n_features)
	 Return: 
	    X_new , transformed like (0000 1111111) 
	'''
	return (X>0).astype(X.dtype)


class MLPClassifier(object):
	def __init__(self, hidden_layer_size=(100,),
	             learning_rate=0.001, tol=1e-4,
	             activationfunc='relu', max_itr=200):
        # Information about the hidden layers: both size and number of layers				 
		  self.hidden_layer_size = hidden_layer_size 
		  self.learning_rate  = learning_rate
		  self.tol = tol
		  self.activationfunc = activationfunc
		  self.max_itr = max_itr
	
	
	def _initialize(self):
		# I'm having them initialised with random state
		self.weights = []
		self.bias = []
		self.n_itr_ = 0 # tototal number of iterations for the algorithm
		
		ini_bound = np.sqrt(6.0/(fan_in + fan_out)  		
	
	
	def _forwardpop(self):
		
	
	def _bp():
		
	def _validate_inputs(self, X, y):
		'''this weakly private method is try to validate the input data
		'''
	
	def _fit(self, X, y):
	  '''change the self state, i.e. train the MLP model
	     making the function weakly private..
	  '''
	  hidden_layer
	  self._validate_hyperparameters()
	  X, y = self._validate_inputdata(X, y)
	  n_samples, n_features = X.shape
	
	
	
	def fit(self, X, y):
		''' Fit the model to data matrix X and target y
		
		Returns :
		---------
		self: returns a trained MLP model.
		
		'''
		
		return self._fit(X, y)
	
	
	def predict():
