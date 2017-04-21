"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


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
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, FILTER, BIAS, WEIGHT

import theano.sandbox.neighbours as TSN

class LeNetConvPoolLayer(Brick):
    """Pool Layer of a convolutional network """

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

    def __init__(self, rng, W, b, filter_shape, image_shape, poolsize=(2, 2), name='ConvRel', **kwargs):
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
        super(LeNetConvPoolLayer, self).__init__(**kwargs)
        assert image_shape[1] == filter_shape[1]
        self.input = input

        add_role(W, WEIGHT)
        add_role(b, BIAS)
        # store parameters of this layer
	self.parameters = []
        self.parameters.append(W)
        self.parameters.append(b)
	self.add_auxiliary_variable(W.norm(2), name='W_norm')
	self.add_auxiliary_variable(b.norm(2), name='b_norm')
	self.allocated = True
	self.name = name
        self.filter_shape = filter_shape
        self.poolsize = poolsize
    
    @property
    def W(self):
        return self.parameters[0]
    
    @property
    def b(self):
        return self.parameters[1]
    
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        # convolve input feature maps with filters
        conv_out = self.convStep(input_, self.W)
        self.conv_out = conv_out
        self.conv_out_tanh = T.tanh(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.conv_out_sigm = T.nnet.sigmoid(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        k = self.poolsize[1]
        self.pooledkmax = self.kmaxPooling(conv_out, k)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        output = T.tanh(self.pooledkmax + self.b.dimshuffle('x', 0, 'x', 'x'))

        return output

