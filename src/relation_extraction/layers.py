#!/usr/bin/python

############
# Description: Implementation of network layers
# Author: Heike Adel
# Year: 2016
# Note: HiddenLayer and LogisticRegression based on theano tutorial
###########

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import theano.sandbox.neighbours as TSN

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    """
    This implementation simplifies the model in the following ways:

    - LeNetConvPool doesn't implement location-specific gain and bias parameters
    - LeNetConvPool doesn't implement pooling by average, it implements pooling
      by max.
    - Digit classification is implemented with a logistic regression rather than
      an RBF network
    - LeNet5 was not fully-connected convolutions at second layer

    References:
    - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
     Gradient-Based Learning Applied to Document
     Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
     http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

    """


    def preparePooling(self, conv_out):
      neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
      self.neighbors = neighborsForPooling
      neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
      neighborsArgSorted = neighborsArgSorted
      return neighborsForPooling, neighborsArgSorted


    def kmaxPooling(self, conv_out, k):
      #neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
      #self.neighbors = neighborsForPooling
      #neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
      #self.neighborsArgSorted = neighborsArgSorted
      neighborsForPooling, neighborsArgSorted = self.preparePooling(conv_out)
      kNeighborsArg = neighborsArgSorted[:,-k:]
      self.neigborsSorted = kNeighborsArg
      kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)
      ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
      jj = kNeighborsArgSorted.flatten()
      self.ii = ii
      self.jj = jj
      pooledkmaxTmp = neighborsForPooling[ii, jj]

      self.pooled = pooledkmaxTmp

      # reshape pooled_out
      new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                         T.as_tensor([conv_out.shape[2]]),
                         T.as_tensor([k])),
                         'int64')
      pooledkmax = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
      return pooledkmax      

    def convStep(self, curInput, curFilter):
      return conv.conv2d(input=curInput, filters=curFilter,
                filter_shape=self.filter_shape,
                image_shape=None)

    def __init__(self, rng, W, b, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type W: theano.matrix
        :param W: the weight matrix used for convolution

        :type b: theano vector
        :param b: the bias used for convolution

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        self.W = W
        self.b = b
        self.filter_shape = filter_shape

        # convolve input feature maps with filters
        conv_out = self.convStep(self.input, self.W)

        self.conv_out = conv_out
        self.conv_out_tanh = T.tanh(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.conv_out_sigm = T.nnet.sigmoid(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        k = poolsize[1]
        self.pooledkmax = self.kmaxPooling(conv_out, k)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(self.pooledkmax + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

#######################################################################

class HiddenLayer(object):
 
    """
    A multilayer perceptron is a logistic regressor where
    instead of feeding the input to the logistic regression you insert a
    intermediate layer, called the hidden layer, that has a nonlinear
    activation function (usually tanh or sigmoid) . One can use many such
    hidden layers making the architecture deep. The tutorial will also tackle
    the problem of MNIST digit classification.

    .. math::

        f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

    References:

        - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

    """

    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, name=""):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        if name != "":
          prefix = name
        else:
          prefix = "mlp_"
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=prefix+'W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=prefix+'b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

##################################################################

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

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

    References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W = None, b = None):
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
        print 'multi instance training ...'
        if W == None:
          # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
          self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='softmax_W', borrow=True)
        else:
          self.W = W

        if b == None:
          # initialize the baises b as a vector of n_out 0s
          self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='softmax_b', borrow=True)
        else:
          self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood_mi(self, y):
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
        maxPerInstance = T.argmax(self.p_y_given_x, axis=1)
        maxInstance = T.argmax(maxPerInstance)
        return -T.log(self.p_y_given_x)[maxInstance,y[maxInstance]]

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
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return [T.argmax(self.p_y_given_x, axis=1), T.max(self.p_y_given_x, axis=1), self.p_y_given_x]
        else:
            raise NotImplementedError()

