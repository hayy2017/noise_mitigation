'''
Created on Feb 9, 2016

@author: yadollah
'''
import argparse
import os
import sys

import h5py
import numpy
import theano
import theano.tensor as T
import yaml
from blocks import main_loop
from blocks.algorithms import GradientDescent, AdaGrad
from blocks.bricks import MLP, Tanh, Logistic, WEIGHT, Rectifier
from blocks.bricks.base import Parameters, application
from blocks.bricks.conv import (ConvolutionalSequence, Flattener, MaxPooling,
                                ConvolutionalActivation)
from blocks.bricks.cost import BinaryCrossEntropy, Cost
from blocks.bricks.lookup import LookupTable
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks import initialization
from blocks.main_loop import MainLoop
from blocks.model import Model
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer, AgnosticSourcewiseTransformer
from blocks.utils import check_theano_variable, shared_floatx_nans
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent, Bidirectional
from blocks.bricks import Linear, Tanh, Initializable, Feedforward, Sequence
from blocks.bricks.parallel import Fork
from myutils import debug_print
from fuel.utils import do_not_pickle_attributes

SEQ_INPUTS = ['mentions', 'contexts']
REC_MODELS = ['rnn', 'lstm', 'bilstm']

class LettersTransposer(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, **kwargs):
        super(LettersTransposer, self).__init__(data_stream=data_stream, produces_examples=data_stream.produces_examples, **kwargs)
  
    def transform_any_source(self, source, _):
        return source.T

def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.01)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()

def initialize_identity(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Identity()
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()

class MultiMisclassificationRate(Cost):
    """ Cost function that calculates the misclassification rate for a
        multi-label classification output.
    """
    @application(outputs=["error_rate"])
    def apply(self, y, y_hat):
        """ Apply the cost function.

        :param y:       Expected output, must be a binary k-hot vector
        :param y_hat:   Observed output, must be a binary k-hot vector
        """
        mistakes = T.neq(y, y_hat)
        return mistakes.mean(dtype=theano.config.floatX)
    
class WindowTransformer(Transformer):
    """ A Transformer that extracts fixed-width windows around a pivotal index
        from a sequence of varying-width arrays, optionally ignoring the
        value at the pivotal index.
    """
    def __init__(self, data_stream, margin=2, padding_val=1,
                 ignore_pivot=False, **kwargs):
        """ Initialize the WindowTransformer.

        :param data_stream:     Datastream to extract windows from
        :param margin:          Margin on each side of the pivotal index
                                to use for the window
        :param padding_val:     Padding value to use if the margin around
                                the pivotal index is not sufficient in the
                                input data
        :param ignore_pivot:    Do not keep the value at the pivotal index
                                in the window.
        """
        super(WindowTransformer, self).__init__(
            data_stream=data_stream,
            produces_examples=data_stream.produces_examples, **kwargs)
        self.margin = margin
        self.padding_val = padding_val
        self.ignore_pivot = ignore_pivot

    def transform_example(self, example):
        example = list(example)
        if 'contexts' in self.sources and 'entmentions' in self.sources:
            ctx_idx = self.sources.index('contexts')
            ent_idx = self.sources.index('entmentions')
            pivot_idx = example[ent_idx]['position'] + self.margin
            #print 'before', example[ctx_idx], example[ent_idx]
            padding = [self.padding_val]*self.margin
            padded_context = padding + list(example[ctx_idx]) + padding
            lidx = pivot_idx - self.margin
            ridx = pivot_idx + self.margin + 1
            if self.ignore_pivot:
                example[ctx_idx] = tuple(
                    padded_context[lidx:pivot_idx] +
                    padded_context[pivot_idx+1:ridx])
            else:
                example[ctx_idx] = tuple(padded_context[lidx:ridx])
            #print 'after', example[ctx_idx]
        return tuple(example)
    def transform_batch(self, batch):
        if 'contexts' in self.sources and 'entmentions' in self.sources:
            ctx_idx = self.sources.index('contexts')
            batch = list(batch)
            ctx_windows = numpy.zeros((batch[ctx_idx].shape[0], self.margin*2),
                    dtype='int32') #TODO: bugy
            for idx, example in enumerate(zip(*batch)):
                ctx = self.transform_example(example)[ctx_idx]
                ctx_windows[idx, :] = ctx
            batch[ctx_idx] = ctx_windows
            batch = tuple(batch)
        return batch
    
class GenerateNegPosTransformer(Transformer):
    def __init__(self, data_stream, margin=2, padding_val=1,
                 ignore_pivot=False, **kwargs):
        super(GenerateNegPosTransformer, self).__init__(
            data_stream=data_stream,
            produces_examples=data_stream.produces_examples, **kwargs)

    def transform_example(self, example):
        example = list(example)
        target_idx = self.sources.index('targets')
        binvec = example[target_idx]
        
        
        if 'contexts' in self.sources and 'entmentions' in self.sources:
            ctx_idx = self.sources.index('contexts')
            ent_idx = self.sources.index('entmentions')
            pivot_idx = example[ent_idx]['position'] + self.margin
            #print 'before', example[ctx_idx], example[ent_idx]
            padding = [self.padding_val]*self.margin
            padded_context = padding + list(example[ctx_idx]) + padding
            lidx = pivot_idx - self.margin
            ridx = pivot_idx + self.margin + 1
            if self.ignore_pivot:
                example[ctx_idx] = tuple(
                    padded_context[lidx:pivot_idx] +
                    padded_context[pivot_idx+1:ridx])
            else:
                example[ctx_idx] = tuple(padded_context[lidx:ridx])
            #print 'after', example[ctx_idx]
        return tuple(example)
    def transform_batch(self, batch):
        if 'contexts' in self.sources and 'entmentions' in self.sources:
            ctx_idx = self.sources.index('contexts')
            batch = list(batch)
            ctx_windows = numpy.zeros((batch[ctx_idx].shape[0], self.margin*2),
                    dtype='int32') #TODO: bugy
            for idx, example in enumerate(zip(*batch)):
                ctx = self.transform_example(example)[ctx_idx]
                ctx_windows[idx, :] = ctx
            batch[ctx_idx] = ctx_windows
            batch = tuple(batch)
        return batch
    
class StaticLookupTable():
    """ A LookupTable that does not update its parameters during training.

     For all other intents and purposes exactly identical to Block's standard
     LookupTable.
    """
    def __init__(self, length, dim):
        self.dim = dim
        self.length = length

    def allocate(self):
        self.W = shared_floatx_nans((self.length, self.dim),
                                         name='lookup_table')
        self.W.name = 'W'
    def W(self):
        return self.W

    def apply(self, indices):
        check_theano_variable(indices, None, ("int", "uint"))
        output_shape = [indices.shape[i]
                  for i in range(indices.ndim)] + [self.dim]
        return self.W[indices.flatten()].reshape(output_shape)


def rnn_layer(in_dim, h, h_dim, n):
    linear = Linear(input_dim=in_dim, output_dim=h_dim, name='linear' + str(n) + h.name)
    rnn = SimpleRecurrent(dim=h_dim, name='rnn' + str(n))
    initialize([linear, rnn])
    return rnn.apply(linear.apply(h))

def gru_layer(dim, h, n):
    fork = Fork(output_names=['linear' + str(n), 'gates' + str(n)],
                name='fork' + str(n), input_dim=dim, output_dims=[dim, dim * 2])
    gru = GatedRecurrent(dim=dim, name='gru' + str(n))
    initialize([fork, gru])
    linear, gates = fork.apply(h)
    return gru.apply(linear, gates)


def lstm_layer(in_dim, h, h_dim, n):
    linear = Linear(input_dim=in_dim, output_dim=h_dim * 4, name='linear' + str(n) + h.name)
    lstm = LSTM(dim=h_dim, name='lstm' + str(n)+h.name)
    initialize([linear, lstm])
    return lstm.apply(linear.apply(h))[0]

def bilstm_layer(in_dim, inp, h_dim, n):
    linear = Linear(input_dim=in_dim, output_dim=h_dim * 4, name='linear' + str(n)+inp.name)
    lstm = LSTM(dim=h_dim, name='lstm' + str(n)+inp.name)
    bilstm = Bidirectional(prototype=lstm)
    bilstm.name = 'bilstm' + str(n) + inp.name
    initialize([linear, bilstm])
    return bilstm.apply(linear.apply(inp))[0]



def create_cnn_general(embedded_x, mycnf, max_len, embedding_size, inp_conv=False):
    fv_len = 0
    filter_sizes = mycnf['cnn_config']['filter_sizes']
    num_filters = mycnf['cnn_config']['num_filters']
    for i, fw in enumerate(filter_sizes):
        conv = ConvolutionalActivation(
                        activation=Rectifier().apply,
                        filter_size=(fw, embedding_size), 
                        num_filters=num_filters,
                        num_channels=1,
                        image_size=(max_len, embedding_size),
                        name="conv"+str(fw)+embedded_x.name)
        pooling = MaxPooling((max_len-fw+1, 1), name="pool"+str(fw)+embedded_x.name)
        initialize([conv])
        if inp_conv:
            convinp = embedded_x
        else:
            convinp = embedded_x.flatten().reshape((embedded_x.shape[0], 1, max_len, embedding_size))
        onepool = pooling.apply(conv.apply(convinp)).flatten(2)
        if i == 0:
            outpools = onepool
        else:
            outpools = T.concatenate([outpools, onepool], axis=1)
        fv_len += conv.num_filters
    return outpools, fv_len

def create_rec(xemb, mycnf, embedding_size):
    hiddensize = mycnf['rnn_config']['hidden']
    mymodel = mycnf['model']
    assert mymodel in REC_MODELS 
    inpsize = embedding_size
    if 'bilstm' in mymodel:
        for i in range(1):
            xemb = bilstm_layer(inpsize, xemb, hiddensize, i)
            xemb.name = 'bilstm' + str(i) + xemb.name
            inpsize = hiddensize * 2
        fv_len = hiddensize * 2
    elif 'lstm' in mymodel:
        for i in range(1):
            xemb = lstm_layer(embedding_size, xemb, hiddensize, 1)
            embedding_size = hiddensize
            xemb.name = 'lstm' + str(i) + xemb.name
        fv_len = hiddensize
    else:
        xemb = rnn_layer(embedding_size, xemb, hiddensize, 1)
        xemb.name = 'rnn' + str(i) + xemb.name
        fv_len = hiddensize
    fv = xemb
    fv = debug_print(fv[fv.shape[0] - 1], 'outRec', False)
    return fv, fv_len


import sys
import h5py
import yaml
from fuel.datasets import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Mapping, Cast, AgnosticSourcewiseTransformer, FilterSources, Merge
from blocks.extensions import saveload, predicates
from blocks.extensions.training import TrackTheBest
from blocks import main_loop
from blocks.initialization import Uniform
from blocks.roles import add_role, WEIGHT, BIAS
from fuel.utils import do_not_pickle_attributes
import theano.tensor as T
import theano
from blocks.bricks.lookup import LookupTable
import numpy, logging
from myutils import build_ngram_vocab, get_ngram_seq,\
    str_to_bool, debug_print
import os
# from src.classification.nn.blocks.joint.model import initialize,\
#     create_cnn_general, create_lstm, create_ff, create_mean
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('myutils')
rng = numpy.random.RandomState(23455)
from theano.tensor import shared_randomstreams

seq_features = ['letters', 'words', 'ngrams2', 'ngrams3', 'ngrams4', 'ngrams5']

def scan_func(M):
    def loop (row):
        one_indices = T.nonzero(row)[0]
        zero_indices = T.eq(row, 0).nonzero()[0]
        random = shared_randomstreams.RandomStreams(5)
        ind1=random.random_integers(size=(1,), low=0, high=one_indices.shape[0]-1, ndim=None)
        ind2=random.random_integers(size=(50,), low=0, high=zero_indices.shape[0]-1, ndim=None)
        return one_indices[ind1], zero_indices[ind2]
    
    pos_inds, updates = theano.scan(fn=loop,
                                sequences=M,
                                outputs_info=None) 
    return pos_inds[0], pos_inds[1], updates

def ranking_loss(y_hat, y):
    pos_inds, neg_inds, updates = scan_func(y)
    index = T.arange(y.shape[0])
    pos_scores = y_hat[index, pos_inds.T].T
    neg_scores = y_hat[index, neg_inds.T].T
    pos_scores = T.tile(pos_scores, neg_scores.shape[1])
    cost = T.sum(T.maximum(0., 1. - pos_scores + neg_scores), axis=1)
    return T.mean(cost), updates  


def cross_entropy_loss(y_hat, y):
    return T.mean(T.nnet.binary_crossentropy(y_hat, y))
    
#Define this class to skip serialization of extensions
@do_not_pickle_attributes('extensions')
class MainLoop(main_loop.MainLoop):

    def __init__(self, **kwargs):
        super(MainLoop, self).__init__(**kwargs)
        
    def load(self):
        self.extensions = []

class LettersTransposer(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, **kwargs):
        super(LettersTransposer, self).__init__(data_stream=data_stream, produces_examples=data_stream.produces_examples, **kwargs)
  
    def transform_any_source(self, source, _):
        return source.T
    
class CutInput(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, max_len, **kwargs):
        super(CutInput, self).__init__(data_stream=data_stream, produces_examples=data_stream.produces_examples, **kwargs)
        self.max_len = max_len
    def transform_any_source(self, source, _):
        return source[:,0:self.max_len]
    
def sample_transformations(thestream):
    cast_stream = Cast(data_stream=thestream,  dtype='float32', which_sources=('features',))
    return cast_stream
                       
def track_best(channel, save_path):
    tracker = TrackTheBest(channel, choose_best=min, after_epoch=True)
    checkpoint = saveload.Checkpoint(
        save_path, after_training=False, use_cpickle=True)
    checkpoint.add_condition(["after_epoch"], predicate=predicates.OnLogRecord('{0}_best_so_far'.format(channel)))
    return [tracker, checkpoint]

def get_targets_metadata(dsdir):
    with h5py.File(dsdir + '_targets.h5py') as f:
        t_to_ix = yaml.load(f['targets'].attrs['type_to_ix'])
        ix_to_t = yaml.load(f['targets'].attrs['ix_to_type'])
    return t_to_ix, ix_to_t

        
def transpose_stream(data):
    return (data[0].T, data[1])

def get_comb_stream(fea2obj, which_set, batch_size=None, shuffle=True):
    streams = []
    for fea in fea2obj:
        obj = fea2obj[fea]
        dataset = H5PYDataset(obj.fuelfile, which_sets=(which_set,),load_in_memory=True)
        if batch_size == None: batch_size = dataset.num_examples
        if shuffle: 
            iterschema = ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size)
        else: 
            iterschema = SequentialScheme(examples=dataset.num_examples, batch_size=batch_size)
        stream = DataStream(dataset=dataset, iteration_scheme=iterschema)
        if fea in seq_features:
            stream = CutInput(stream, obj.max_len)
            if obj.rec == True:
                logger.info('transforming data for recursive input')
                stream = LettersTransposer(stream, which_sources=fea)# Required because Recurrent bricks receive as input [sequence, batch,# features]
        streams.append(stream)
    stream = Merge(streams, tuple(fea2obj.keys()))
    return stream, dataset.num_examples


from blocks.bricks import Linear, Tanh
from blocks.bricks import MLP, Rectifier, Tanh, Linear, Softmax, Logistic
from blocks import initialization
from blocks.bricks.base import Parameters, application
from blocks.bricks.cost import BinaryCrossEntropy, Cost

class MultiMisclassificationRate(Cost):
    """ Cost function that calculates the misclassification rate for a
        multi-label classification output.
    """
    @application(outputs=["error_rate"])
    def apply(self, y, y_hat):
        """ Apply the cost function.

        :param y:       Expected output, must be a binary k-hot vector
        :param y_hat:   Observed output, must be a binary k-hot vector
        """
        mistakes = T.neq(y, y_hat)
        return mistakes.mean(dtype=theano.config.floatX)

def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()
        
def softmax_layer(h, y, hidden_size, num_targets, cost_fn='cross'):
    hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size, output_dim=num_targets)
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(h)
    linear_output.name = 'linear_output'
    y_pred = T.argmax(linear_output, axis=1)
    label_of_predicted = debug_print(y[T.arange(y.shape[0]), y_pred], 'label_of_predicted', False)
    pat1 = T.mean(label_of_predicted)
    updates = None
    if 'ranking' in cost_fn:
        cost, updates = ranking_loss(linear_output, y)
        print 'using ranking loss function!'
    else:
        y_hat = Logistic().apply(linear_output)
        y_hat.name = 'y_hat'
        cost = cross_entropy_loss(y_hat, y)
    cost.name = 'cost'
    pat1.name = 'precision@1'
    misclassify_rate = MultiMisclassificationRate().apply(y, T.ge(linear_output, 0.5))
    misclassify_rate.name = 'error_rate'
    return cost, pat1, updates, misclassify_rate

