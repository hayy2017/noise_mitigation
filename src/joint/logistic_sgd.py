"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, FILTER, BIAS, WEIGHT

class LogisticRegression(Brick):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input_, n_in, n_out, name='logisticRegression_rel', W = None, b = None, **kwargs):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        print '******************no MIML'
        super(LogisticRegression, self).__init__(**kwargs)
        if W == None:
          # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
          W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX), name='W')
       # else:
       #   self.W = W

        if b == None:
          # initialize the baises b as a vector of n_out 0s
          b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX), name='b')
       # else:
        #  self.b = b
        add_role(W, WEIGHT)
        add_role(b, BIAS)
	self.parameters = []
        self.parameters.append(W)
        self.parameters.append(b)
	self.add_auxiliary_variable(W.norm(2), name='W_norm')
	self.add_auxiliary_variable(b.norm(2), name='b_norm')
	self.allocated = True
	self.name = name
        self.p_y_given_x = T.nnet.softmax(T.dot(input_, self.parameters[0]) + self.parameters[1])

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
	
    @property
    def W(self):
        return self.parameters[0]
 
    @property
    def b(self):
        return self.parameters[1]
 
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def neg_log_likelihood_weightPrecision(self, y, alpha):
      '''
      self.y_pred * (1-y) == 1 if we have a FP
      '''
      self.weightPrec = self.y_pred * (1-y) * alpha * T.log(self.p_y_given_x)[T.arange(y.shape[0]), 1-y]
      self.weightedEntropy = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y] + self.weightPrec
      return -T.mean(self.weightedEntropy)

    def riskLoss(self, y, alpha):
      '''
      self.y_pred * (1-y) == 1 if we have a FP
      '''
      self.isFP = self.y_pred * (1-y)
      self.risk = self.isFP * T.log(alpha)
      self.weightedEntropy = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y] + self.risk
      return -T.mean(self.weightedEntropy)

    def hingeLoss(self, y):
      return T.mean(T.maximum(0, T.max(self.p_y_given_x, axis=1) - self.p_y_given_x[T.arange(y.shape[0]), y]))

    def svmLoss(self, y):
      #self.test1 = y
      y_svm = T.switch(T.eq(y,0), y-1, y)
      #self.test2 = y_svm
      #self.test3 = T.max(self.p_y_given_x, axis=1)
      #self.test4 = y_svm * T.max(self.p_y_given_x, axis=1)
      return T.mean(T.maximum(0, 1 - y_svm * T.max(self.p_y_given_x, axis=1)))

    def f1score(self, y, beta):
      """Return the mean of the f_beta score of the preciction of this
      model under a given targe distribution.

      .. math::
      
      """
      self.numerator = T.mean(self.y_pred * y)
      self.denominator = T.mean(self.y_pred + beta * beta * y)
      return 1.0 - 1.0 * (1 + beta * beta) * self.numerator / self.denominator

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def trainerror(self, y):
        """ Return cross entropy error for trainnig
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_pred_clipped = T.clip(self.y_pred, 0.0001, 0.9999)
            return T.mean(T.nnet.binary_crossentropy(y_pred_clipped, y))
        else:
            raise NotImplementedError()

    def results(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return [T.argmax(self.p_y_given_x, axis=1), T.max(self.p_y_given_x, axis=1), self.p_y_given_x]
        else:
            raise NotImplementedError()

    def resultsNoY(self):
      return [T.argmax(self.p_y_given_x, axis=1), T.max(self.p_y_given_x, axis=1), self.p_y_given_x]

