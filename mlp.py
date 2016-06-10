#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import warnings
''' implement sa multilayer perceptron classifier
  also known as deep feedforward networks in deep learning
'''
import pdb
# from IPython.Debugger import Tracer; debug_here = Tracer()


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
    return (X > 0).astype(X.dtype)


class MLPClassifier(object):
    def __init__(self, hidden_layer_sizes=(100,),
                 learning_rate=0.001, tol=1e-4, lam=0.001,
                 activationfunc='relu', max_iter=15000,
                 verbose=True):
        # Information about the hidden layers: both sizes and number of layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.tol = tol
        self.lam = lam
        self.activationfunc = activationfunc
        self.max_iter = max_iter
        self.trainedflag = False
        self.verbose = verbose

    def _initialize(self):
        # I'm having them initialised with random state
        self.wb = self._weights_biases_initial()
        self.n_iter_ = 0  # total number of iterations for the algorithm

    def _weights_biases_initial(self):
        wb = np.zeros(self.n_weights+self.n_biases)
        # randomly initialise the weights and biases
        for i in range(self.n_layers):
            fan_in, fan_out = self.layer_units[i], self.layer_units[i+1]
            ini_bound = np.sqrt(6.0/(fan_in + fan_out))
            w_s, w_end, _ = self._get_weight_pos(i)
            b_s, b_end, _ = self._get_bias_pos(i)
            assert w_end - w_s == self._f_weights_[i]
            assert b_end - b_s == self._f_biases_[i]
            wb[w_s:w_end] = np.random.uniform(-ini_bound,
                                              ini_bound, self._f_weights_[i])
            wb[b_s:b_end] = np.random.uniform(-ini_bound, ini_bound,
                                              self._f_biases_[i])
        return wb

    def _get_weight_pos(self, ith_layer):
        w_start = self._start_end_weights[ith_layer]
        w_end = self._start_end_weights[ith_layer+1]
        w_shape = (self.layer_units[ith_layer], self.layer_units[ith_layer+1])
        return w_start, w_end, w_shape

    def _get_bias_pos(self, ith_layer):
        b_start = self._start_end_biases[ith_layer]
        b_end = self._start_end_biases[ith_layer+1]
        b_start = b_start + self.n_weights
        b_end = b_end+self.n_weights
        b_shape = (self.layer_units[ith_layer+1], 1)
        return b_start, b_end, b_shape

    def _weights_biases_flattened_(self):
        # flattened weights and biases shapes
        units = self.layer_units
        self._f_weights_ = []  # something like [4, 8, 6, 3]
        self._f_biases_ = units[1:]
        for i in range(len(units)-1):
            self._f_weights_ = self._f_weights_+[units[i]*units[i+1]]

        self.n_weights = sum(self._f_weights_)
        self.n_biases = sum(self._f_biases_)
        # Convention: weights frist, then biases
        self._start_end_weights = np.hstack((0, np.cumsum(self._f_weights_)))
        self._start_end_biases = np.hstack((0, np.cumsum(self._f_biases_)))

    def _forwardpass(self, X, packedwb, n_samples):
        ''' forward pass to obtain the
        {a} and {h} needed ..
       '''
        h_last = X.T
        a, h = [], [h_last]
        I = np.ones((1, n_samples))
        # pdb.set_trace()
        for i in range(self.n_layers):
            W, b = self._unpack(packedwb, i)
            a_temp = np.dot(b, I)+np.dot(W.T, h_last)
            if self.activationfunc == 'relu':
                h_last = relu(a_temp)
            else:
                raise ValueError('Activation function not supported!')
            a = a+[a_temp]
            h = h+[h_last]
        return a, h

    def _bp(self, packedwb, X, y, n_samples):
        '''primarily to deal with gradients ..
        '''
        a, h = self._forwardpass(X, packedwb, n_samples)
        I = np.ones((1, n_samples))
        grads = np.zeros(self.n_weights+self.n_biases)
        # loop backward

        W, _ = self._unpack(packedwb, self.n_layers-1)

        for i in np.arange(self.n_layers-1, -1, -1):
            if i == self.n_layers-1:
                g = (h[i+1].T-y)*relu_derivative(a[i]).T
            else:
                g = np.dot(g, W.T)*relu_derivative(a[i]).T

            if i != self.n_layers-1:
                W, _ = self._unpack(packedwb, i)
            dJdW = np.dot(h[i], g)/n_samples + self.lam*W
            dJdb = np.dot(I, g).T/n_samples
            # pdb.set_trace()
            grads = self._pack(grads, dJdW, dJdb, i)
        return grads

    def costfunc(self, packedwb, X, y, n_samples):
        ''' construct cost function

        '''
        _, h = self._forwardpass(X, packedwb, n_samples)
        d_ = (h[-1]-y.T)**2
        cost = 0.5*np.mean(d_)+0.5*self.lam*sum(
            packedwb[0:self.n_weights]**2)
        grads = self._bp(packedwb, X, y, n_samples)
        self.n_iter_ = self.n_iter_+1
        return cost, grads

    def _validate_hyperparameters(self):
        if self.max_iter <= 0:
                raise ValueError("max_iter must be positive!")
        if self.learning_rate <= 0:
                raise ValueError("learning_rate must be positive!")
        if self.activationfunc != 'relu':
                raise ValueError('activationfunc only support relu!')

    def _validate_predictinputX(self, X):
        if not hasattr(X, '__iter__'):
            X = [X]
        if X.ndim == 1:
            m = X.shape[0]
            X = X.reshape(m, 1)
        else:
            m, n = X.shape
            if n != self.n_features:
                raise ValueError('Wrong number of features given!')
            else:
                return X

    def _validate_inputs(self, X, y):
        '''this weakly private method is try to validate the input data
        '''
        if X.shape[0] != y.shape[0]:
            raise ValueError("The training data are inconsistent, check"
                                 " the feature and corresponding labels!")
        if y.ndim == 1:
            m = y.shape[0]
            y = y.reshape(m, 1)
        return X, y

    def _pack(self, destination, W, b, ith_layer):
        # pack the flattened Weight and bias (should be list)
        w_s, w_end, _ = self._get_weight_pos(ith_layer)
        b_s, b_end, _ = self._get_bias_pos(ith_layer)
        # pdb.set_trace()
        assert w_end-w_s == W.shape[0]*W.shape[1]
        assert b_end-b_s == b.shape[0]*b.shape[1]
        destination[w_s:w_end] = W.ravel()
        destination[b_s:b_end] = b.ravel()
        return destination

    def _unpack(self, packedwb, ith_layer):
        w_s, w_end, wshape = self._get_weight_pos(ith_layer)
        b_s, b_end, bshape = self._get_bias_pos(ith_layer)
        assert w_end-w_s == wshape[0]*wshape[1]
        assert b_end-b_s == bshape[0]*bshape[1]
        weights = packedwb[w_s:w_end].reshape(wshape)
        b = packedwb[b_s:b_end].reshape(bshape)
        return weights, b

    def _fit(self, X, y):
        '''change the self state, i.e. train the MLP model
                 making the function weakly private..
        '''

        # FIRST: make sure hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]

        hidden_layer_sizes = list(hidden_layer_sizes)

        # Second: validate other parameters
        self._validate_hyperparameters()

        # then validate INPUT data
        X, y = self._validate_inputs(X, y)
        n_samples, n_features = X.shape
        n_output = y.shape[1]
        self.n_features = n_features

        # Set up layer_units : [n_features, hidden_layer_sizes, noutput]
        self.layer_units = [n_features]+hidden_layer_sizes+[n_output]

        # Set up number of layers
        self.n_layers = len(self.layer_units)-1

        # Set up the weights and biases
        self._weights_biases_flattened_()

        # Initialise weights and biases
        self._initialize()
        
        _init_costs, _ = self.costfunc(self.wb, X, y, n_samples)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1
        # using l_BFGS method to optimize W and b
        wb_star, self.cost_, opinfo = fmin_l_bfgs_b(
            func=self.costfunc,
            x0=self.wb,
            maxfun=self.max_iter,
            iprint=iprint,
            pgtol=self.tol,
            args=(X, y, n_samples))
        if opinfo['warnflag'] == 1:
            self.trainedflag = True
            self.wb = wb_star
            print('initial costs:')
            print(_init_costs)
            print('after training:')
            print(self.costfunc(wb_star, X, y, n_samples)[0])
        else:
            warnings.warn("Training has failed")

    def fit(self, X, y):
        ''' Fit the model to data matrix X and target y

        Returns :
        ---------
        self: returns a trained MLP model.

        '''
        return self._fit(X, y)

    def predict(self, X):
        if not self.trainedflag:
            print('I am refusing to predict! You have not '
                   'trained the model yet!')
            return
        else:
            X = self._validate_predictinputX(X)
            _, hs = self._forwardpass(X, self.wb, X.shape[0])
            return hs[-1]
