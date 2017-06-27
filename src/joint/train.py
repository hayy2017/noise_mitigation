'''
Created on Feb 9, 2016

@author: yadollah
'''
from _collections import defaultdict
import argparse
import cPickle
from collections import Iterable, Counter
import logging
import os
import shutil
import string
from subprocess import Popen
import subprocess
import sys

from numpy import argmax, mean
import numpy
import six
import theano

from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp, AdaGrad, Scale, AdaDelta, Momentum, Adam
from blocks.bricks import Initializable, Feedforward
from blocks.bricks import MLP, Tanh, Logistic, WEIGHT, Rectifier
from blocks.bricks import WEIGHT, MLP, Rectifier, Tanh, Linear, Softmax, Logistic
from blocks.bricks.base import Parameters, application
from blocks.bricks.conv import (ConvolutionalSequence, Flattener, MaxPooling,
                                ConvolutionalActivation)
from blocks.bricks.cost import BinaryCrossEntropy, Cost
from blocks.bricks.lookup import LookupTable
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing, saveload
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.serialization import load
from blocks.theano_expressions import l2_norm
from blocks.utils import shared_floatx_nans
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme, IterationScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer, Merge, Cast
import h5py
from picklable_itertools import chain, repeat, imap, iter_
from picklable_itertools.base import BaseItertool
from model import cross_entropy_loss, track_best, \
    MainLoop, ranking_loss, softmax_layer
from convWithAttention import LeNetConvPoolLayer
from model import MultiMisclassificationRate, StaticLookupTable, initialize, \
    LettersTransposer, create_cnn_general, SEQ_INPUTS, create_rec, REC_MODELS, \
    GenerateNegPosTransformer, initialize_inout
from blocks.roles import add_role, WEIGHT
from myutils import debug_print, fillt2i, \
    build_type2entmatrix, big2small, write_small, load_lines_info, MyPool, \
    computeFscore, calcPRF, softmax, normalize
import theano.tensor as T
import yaml
import time
from mlp import HiddenLayer
from myExtensions import F1MultiClassesExtension,\
    GetPRcurve, WriteBest, ModelResultsMI, ALLModelResultsMI

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('train.py')
def calc_AP(scores_list):
    mylist = sorted(scores_list, key=lambda tuple: tuple[0], reverse=True)
    rel_docs_up_to_i = 0.0
    prec_list = []
    for i in range(len(mylist)):
        if mylist[i][1] == 1: # found a relevant document
            rel_docs_up_to_i += 1.
            prec = rel_docs_up_to_i / (i + 1)
            prec_list.append(prec)
    if len(prec_list) == 0:
        return 0.
    return mean(numpy.asarray(prec_list))

class WindowTransformer(Transformer):
    """ A Transformer that extracts fixed-width windows around a pivotal index
        from a sequence of varying-width arrays, optionally ignoring the
        value at the pivotal index.
    """
    def __init__(self, data_stream, margin=2, padding_val=1,
                 ignore_pivot=False, ctx_num=1, **kwargs):
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

    def build_window(self, example, ctx_num):
        context_label = 'contexts' + ctx_num 
        entmen_label = 'entmen' + ctx_num 
        if context_label in self.sources and entmen_label in self.sources:
            ctx_idx = self.sources.index(context_label)
            ent_idx = self.sources.index(entmen_label)
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
        return example
    
    def transform_example(self, example):
        example = list(example)
        example = self.build_window(example, ctx_num='1')
        example = self.build_window(example, ctx_num='2')
        return tuple(example)
    
    def transform_batch(self, batch):
        context_label = 'contexts' 
        if True:#context_label in self.sources and entmen_label in self.sources:
            ctx_idx1 = self.sources.index(context_label + '1')
            ctx_idx2 = self.sources.index(context_label + '2')
            batch = list(batch)
            ctx_windows1 = numpy.zeros((batch[ctx_idx1].shape[0], self.margin*2), dtype='int32') #TODO: bugy
            ctx_windows2 = numpy.zeros((batch[ctx_idx2].shape[0], self.margin*2), dtype='int32') #TODO: bugy
            for idx, example in enumerate(zip(*batch)):
                ctx1 = self.transform_example(example)[ctx_idx1]
                ctx2 = self.transform_example(example)[ctx_idx2]
                ctx_windows1[idx, :] = ctx1
                ctx_windows2[idx, :] = ctx2
            batch[ctx_idx1] = ctx_windows1
            batch[ctx_idx2] = ctx_windows2
            batch = tuple(batch)
        return batch
    
class My_partition_all(BaseItertool):
    """Partition all elements of sequence into tuples of length at most n
    The final tuple may be shorter to accommodate extra elements.
    >>> list(partition_all(2, [1, 2, 3, 4]))
    [(1, 2), (3, 4)]
    >>> list(partition_all(2, [1, 2, 3, 4, 5]))
    [(1, 2), (3, 4), (5,)]
    See Also:
        partition
    """
    def __init__(self, n_list, seq):
        self._n_list = iter_(n_list)
        self._seq = iter_(seq)

    def __next__(self):
        items = []
        try:
            for _ in six.moves.xrange(next(self._n_list)):
                items.append(next(self._seq))
        except StopIteration:
            pass
        if len(items) == 0:
            raise StopIteration
        return tuple(items)

class MySequentialScheme(IterationScheme):
    """Sequential batches iterator.
    Iterate over all the examples in a dataset of fixed size sequentially
    in batches of a given size.
    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.
    """
    requests_examples = False

    def __init__(self, examples, batch_size_list):
        if isinstance(examples, Iterable):
            self.indices = examples
        else:
            self.indices = xrange(examples)
        self.batch_size_list = batch_size_list
        self.rng = numpy.random.RandomState(987654)
        
    def get_request_iterator(self):
#         return imap(list, My_partition_all(self.batch_size_list, self.indices))
        tmp = list(My_partition_all(self.batch_size_list, self.indices))
        return imap(list, tmp)

class MyShuffledScheme(IterationScheme):
    """Sequential batches iterator.
    Iterate over all the examples in a dataset of fixed size sequentially
    in batches of a given size.
    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.
    """
    requests_examples = False

    def __init__(self, examples, batch_size_list):
        if isinstance(examples, Iterable):
            self.indices = examples
        else:
            self.indices = xrange(examples)
        self.batch_size_list = batch_size_list
        self.rng = numpy.random.RandomState(987654)
        
    def get_request_iterator(self):
#         return imap(list, My_partition_all(self.batch_size_list, self.indices))
        tmp = list(My_partition_all(self.batch_size_list, self.indices))
        self.rng.shuffle(tmp)
        return imap(list, tmp)
    
def find_best_theta(pred_scores, gold_labels):
    best_theta = 0.; best_f = 0.
    totals = gold_labels.sum()
    possible_theta = numpy.arange(0,10.5,.5)/10
    for p in possible_theta:
        positives = numpy.nonzero(pred_scores >= p)[0]
        if len(positives) == 0: 
            continue
        goods = gold_labels[positives].sum()
        prec = float(goods) / len(positives)
        rec = float(goods) / totals
        f = 0. if goods == 0. else 2. / (1. / prec + 1. / rec)
#         print p, goods, totals, len(positives), prec, rec, f
        if f > best_f:
            best_f = f
            best_theta = p
    print best_f, best_theta
    return best_theta + 0.15 


class JointUnaryBinary(object):
    """ Trains a classifier for an entity typing task.

    Uses either a standard Multi-Layer Perceptron with a single hidden layer
    or in addition a Convolutional Neural Network with one convoluation and
    one max-pooling layer.
    """
    @staticmethod
    def from_config( config_path):
        """ Instantiate the trainer from an existing configuration.

        :param config_path:     Path to YAML file with configuration\
        :returns:               The initialized JointUnaryBinary instance
        """
        with open(config_path) as fp:
            config = yaml.load(fp)
        trainer = JointUnaryBinary(config['dsdir'], config['embeddings_path'])
        trainer._config.update(config)
        trainer.t2i,_ = fillt2i(trainer._config['typefile'])
        trainer.dev_big_matrix = None
        trainer.test_big_matrix = None
        
#         trainer.set2bag_len_list = {}
#         trainer.set2bag_len_list['train'] = trainer.get_bag_sizes('train', 'entmentions')
#         trainer.set2bag_len_list['dev'] = trainer.get_bag_sizes('dev', 'entmentions')
#         trainer.set2bag_len_list['devbig'] = trainer.get_bag_sizes('devbig', 'entmentions')
#         trainer.set2bag_len_list['test'] = trainer.get_bag_sizes('test', 'entmentions')
#         print len(trainer.set2bag_len_list['devbig']), len(trainer.set2bag_len_list['test'])
        trainer.curSeed = 23455
        if "seed" in trainer._config:
            trainer.curSeed = int(trainer._config["seed"])
        return trainer

    def __init__(self, samples_path, embeddings_path):
        """ Initialize the trainer.

        :param samples_path:    Path to HDF file with the samples
        :param embeddings_path: Path to HDF file with the embedding vectors
        """
        self._config = {}
        self._config['dsdir'] = samples_path
        self._config['embeddings_path'] = embeddings_path
        self._config.update({
            #'max_len': 4,
            'batch_sizes': {
                'train': 10000,
                'dev': 10000,
                'test': 2048
            },
            'initial_learning_rate': 0.05,
            'checkpoint_path': '/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov/figertypes/cis3-11May15/may20-2/cm_fuel/datasets/first/net.toload',
            'net_path': '/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov/figertypes/cis3-11May15/may20-2/cm_fuel/datasets/first/net',
            'max_num_epochs': 50,
            'finish_if_no_improvement': 5,
            'hidden_units': 300,
            'l2_regularization': 0.005,
        })

    def fill_rel_params(self):
        config = self._config
        
        datafile = config["file"]
        print "datafile ", datafile
        self.rel_fuelfile = datafile
        self.rel_test_fulefile = self.rel_fuelfile + '.test'
        if 'testfile' in config:
            self.rel_test_fulefile = config['testfile']
            
        self.rel_loss = "entropy"
        self.rel_filtersize = [1,int(config["filtersize"])]
        self.rel_nkerns = [int(config["nkerns"])]
        self.rel_pool = [1, int(config["kmax"])]
        self.rel_contextsize = int(config["contextsize"])
        self.rel_numClasses = int(config["numClasses"]) + 1 # plus 1 negative class ($NA)
        print "number of classes: " + str(self.rel_numClasses)
        self.rel_vectorsize = config['embedding_size']
        self.rel_myLambda1 = 0
        if "lambda1" in config:
            self.rel_myLambda1 = float(config["lambda1"])
        self.rel_myLambda2 = 0
        if "lambda2" in config:
            self.rel_myLambda2 = float(config["lambda2"])
        print "lambda1 ", self.rel_myLambda1
        print "lambda2 ", self.rel_myLambda2
        loss = "entropy"
        print "using loss function: ", loss
        self.rel_useHiddenLayer = True
        if "noHidden" in config:
            self.rel_useHiddenLayer = False
            print "using no hidden layer"
        else:
            self.rel_hiddenunits = int(config["hidden"])
            print "hidden units ", self.rel_hiddenunits
        self.rel_hiddenTypeUnits = 50
        if "hiddentype" in config:
            self.rel_hiddenTypeUnits = int(config["hiddentype"])
        print "hidden type units ", self.rel_hiddenTypeUnits
        self.rel_randomizeTrain = False
        if "randomizeTrain" in config:
            self.rel_randomizeTrain = True
            self.rel_corruption_level = float(config["randomizeTrain"])
            print "apply 0-1-corruption mask with probability " + str(self.rel_corruption_level) + " to input"
        self.rel_normalizeTypes = False
        if "normalizeTypes" in config:
            self.rel_normalizeTypes = True
            print "normalizing types before concatenating them"
        
        self.rel_iterationSeed = -1
        if "iterationSeed" in config:
            self.rel_iterationSeed = int(config["iterationSeed"])
            print "using " + str(self.rel_iterationSeed) + " as seed for iteration scheme"
        
        if self.rel_contextsize < self.rel_filtersize[1]:
            print "INFO: setting filtersize to ", self.rel_contextsize
            self.rel_filtersize[1] = self.rel_contextsize
        print "filtersize ", self.rel_filtersize
        
        self.rel_sizeAfterConv = self.rel_contextsize - self.rel_filtersize[1] + 1
        self.rel_sizeAfterPooling = -1
        if self.rel_sizeAfterConv < self.rel_pool[1]:
            print "INFO: setting poolsize to ", self.rel_sizeAfterConv
            self.rel_pool[1] = self.rel_sizeAfterConv
        if "kmax" in config or "variableLength":
            self.rel_sizeAfterPooling = self.rel_pool[1]
            print "kmax pooling: k = ", self.rel_pool[1]
        else:
            sizeAfterPooling = self.rel_sizeAfterConv / self.rel_pool[1]
            if self.rel_sizeAfterConv % self.rel_pool[1] != 0:
                sizeAfterPooling += 1
                print "traditional pooling: pool = ", self.rel_pool[1]
        
        self.rel_representationsize = self.rel_vectorsize + 1
        self.rel_use_type = True
        if 'use_type' in self._config:
            self.rel_use_type = self._config['use_type'] 
        self.rel_entitysize = 102
        if 'entitysize' in self._config:
            self.rel_entitysize = self._config['entitysize'] 
        
        print self.rel_representationsize
    
    def gen_neg_pos_sources(self, stream):
        stream.sources.append('negatives')
        
    
    def get_bag_sizes(self, which_set='', source='entmentions'):
        with h5py.File(self._config['dsdir'] + '_bag_lenghts.hdf', "r") as fp:
            return list(fp.get(which_set).value.astype('int32'))
        
        logger.info('filling bag size list from %s', which_set)
        dataset = H5PYDataset(self._config['dsdir'] + '_' + source + '.hdf', which_sets=(which_set,), load_in_memory=True)
        data_stream = DataStream(dataset=dataset, iteration_scheme=SequentialScheme(examples=dataset.num_examples, batch_size=1))
        epoch_iter = data_stream.get_epoch_iterator(as_dict=True)
        bag_len_list = []
        old = epoch_iter.next()[source]['id']
        idx = 1
        n = 0
        while idx < dataset.num_examples:
            n += 1
            idx += 1
            src2vals  = epoch_iter.next()
            newe = src2vals[source]['id']
            if newe != old:
                bag_len_list.append(n)
                n = 0
            old = newe
        return bag_len_list
        
    def get_datastreams_joint(self, which_set='', sources_types=('mentions', 'contexts', 'entmentions', 'targets'), max_len=None, shuffling=False,
                        batch_size=None, multi=False, sources_rel=['xa', 'xb', 'xc', 'ent1', 'ent2', 'y'], bag_size_list=None, num_samples=None):
        """ Load all datastreams.
        :param max_len:     The desired window size
        """
        print bag_size_list, multi
        max_len = max_len or self._config['contexts']['max_len']
        batch_size = batch_size or self._config['batch_sizes'][which_set]
        streams = []
        for src in sources_types:
            print src, self._config['dsdir'] + '_' + src + '.hdf'
            dataset = H5PYDataset(self._config['dsdir'] + '_' + src + '.hdf', which_sets=(which_set,), load_in_memory=True)
            nexamples = num_samples or dataset.num_examples
            if shuffling == True:
                if multi:
                    iteration_scheme = MyShuffledScheme(examples=nexamples, batch_size_list=bag_size_list)
                else:
                    iteration_scheme = ShuffledScheme(examples=nexamples, batch_size=batch_size)
            else:
#                     batch_size_list = self.set2bag_len_list[which_set] if bag_size_list == None else bag_size_list
                if multi:
                    iteration_scheme = MySequentialScheme(examples=nexamples, batch_size_list=bag_size_list)
                else:
                    iteration_scheme = SequentialScheme(examples=nexamples, batch_size=batch_size)
            data_stream = DataStream(dataset=dataset, iteration_scheme=iteration_scheme)
            streams.append(data_stream)
        
        stream = Merge(streams, sources_types)
        stream = WindowTransformer(stream, ignore_pivot=True, margin=max_len//2) 
        for src in sources_types:
            if src in SEQ_INPUTS and self._config[src]['model'] in REC_MODELS:
                logger.info('transposing %s ...', src)
                stream = LettersTransposer(stream, which_sources=src)# Required because Recurrent bricks receive as input [sequence, batch,# features]
                
        if which_set == 'dev':
            which_set = 'test'
        data_set_rel = H5PYDataset(self.rel_fuelfile, which_sets = (which_set,), load_in_memory=True)
        
        numSamples = data_set_rel.num_examples
        
        print "got " + str(numSamples) + " rel examples in : ", which_set
        if shuffling == True:
            if multi:
                stream_rel = DataStream(data_set_rel, iteration_scheme=MyShuffledScheme(examples=nexamples, batch_size_list=bag_size_list))
            else:  
                stream_rel = DataStream(data_set_rel, iteration_scheme=ShuffledScheme(examples=nexamples, batch_size=batch_size))
        else:
            if multi:
                stream_rel = DataStream(data_set_rel, iteration_scheme=MySequentialScheme(examples=nexamples, batch_size_list=bag_size_list))
            else:
                stream_rel = DataStream(data_set_rel, iteration_scheme=SequentialScheme(examples=nexamples, batch_size=batch_size))
        cast_stream = Cast(data_stream=stream_rel, dtype='float32', which_sources=('xa', 'xb', 'xc', 'ent1', 'ent2',))
        print "rel sources: ", cast_stream.sources
        stream = Merge([stream, cast_stream], sources_types + list(cast_stream.sources))
        print stream.sources 
        
        return stream, dataset.num_examples

    def get_datastreams(self, which_set='', sources=('mentions', 'contexts', 'entmentions', 'targets'), max_len=None, shuffling=False,
                        batch_size=None, multi=False, sources_rel=['xa', 'xb', 'xc', 'ent1', 'ent2', 'y'], bag_size_list=None):
        """ Load all datastreams.
        :param max_len:     The desired window size
        """
        print bag_size_list, multi
        max_len = max_len or self._config['contexts']['max_len']
        batch_size = batch_size or self._config['batch_sizes'][which_set]
        streams = []
        for src in sources:
            print src, self._config['dsdir'] + '_' + src + '.hdf'
            dataset = H5PYDataset(self._config['dsdir'] + '_' + src + '.hdf', which_sets=(which_set,), load_in_memory=True)
            if shuffling == True:
                if multi:
                    iteration_scheme = MyShuffledScheme(examples=dataset.num_examples, batch_size_list=bag_size_list)
                else:
                    iteration_scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size)
            else:
#                     batch_size_list = self.set2bag_len_list[which_set] if bag_size_list == None else bag_size_list
                if multi:
                    iteration_scheme = MySequentialScheme(examples=dataset.num_examples, batch_size_list=bag_size_list)
                else:
                    iteration_scheme = SequentialScheme(examples=dataset.num_examples, batch_size=batch_size)
            data_stream = DataStream(dataset=dataset, iteration_scheme=iteration_scheme)
            streams.append(data_stream)
        
        stream = Merge(streams, sources)
        stream = WindowTransformer(stream, ignore_pivot=True, margin=max_len//2) 
        for src in sources:
            if src in SEQ_INPUTS and self._config[src]['model'] in REC_MODELS:
                logger.info('transposing %s ...', src)
                stream = LettersTransposer(stream, which_sources=src)# Required because Recurrent bricks receive as input [sequence, batch,# features]
        return stream, dataset.num_examples
    
    def get_embeddings(self):
        with h5py.File(self._config['dsdir'] + '_embeddings.hdf', "r") as fp:
            return (fp.get('vectors').value.astype('float32'),
                    fp.get('words').value)
        logger.info('loading embeddings finished!')

    def build_mymodel(self, embedded_x, max_len, embedding_size, mycnf, mymodel='mean', use_conv_inp=False):
        logger.info('%s length: %d ', mymodel, max_len)
        if 'cnn' in mymodel:
            fv, fv_len = create_cnn_general(embedded_x, mycnf, max_len, embedding_size, inp_conv=use_conv_inp)
        elif mymodel in REC_MODELS:
            fv, fv_len = create_rec(embedded_x, mycnf, embedding_size)
        elif 'ff' in mymodel:
            logger.info('training with feed-forward')
            fv = embedded_x.flatten(2)
            fv_len = max_len * embedding_size
        elif 'mean' in mymodel:
            logger.info('model: average of mention words')
            if use_conv_inp == False:
                fv = T.mean(embedded_x, axis=1)
            else:
                fv = T.mean(embedded_x, axis=2)
                
            fv_len = embedding_size
        return fv, fv_len
    
    def split_inp(self, max_len, embedded_x, mymodel, DEBUG, use_conv_inp=False):
        pivot = max_len // 2
        l_size = max_len /2
        r_size = max_len /2
        if mymodel in REC_MODELS:
            if use_conv_inp == False:
                l_emb = embedded_x[:l_size,:, :]; 
                r_emb = embedded_x[pivot:pivot+r_size, :, :];
            else:
                l_emb = embedded_x[:, l_size,:, :]; #TODO: Probably wrong 
                r_emb = embedded_x[:, pivot:pivot+r_size, :, :]; #TODO: Probably wrong
        else:
            if use_conv_inp == False:
                l_emb = embedded_x[:,:l_size,:]; 
                r_emb = embedded_x[:,pivot:pivot+r_size,:];
            else:
                l_emb = embedded_x[:,:, :l_size,:]; 
                r_emb = embedded_x[:,:, pivot:pivot+r_size,:];
        l_emb = debug_print(l_emb, 'l_emb', DEBUG)
        l_emb.name = 'left_ctx_emb'
        r_emb = debug_print(r_emb, 'r_emb', DEBUG)
        r_emb.name = 'right_ctx_emb'
        return l_emb, l_size, r_emb, r_size

    def apply_ctxwise_cnn(self, l_emb1, l_size1, l_emb2, l_size2, r_emb1, r_size1, r_emb2, r_size2, embedding_size, mycnf):
        assert l_size1 == r_size1
        assert l_size2 == r_size2
        assert l_size1 == l_size1
        max_len = l_size1
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
                            name="conv"+str(fw))
            pooling = MaxPooling((max_len-fw+1, 1), name="pool"+str(fw))
            initialize([conv])
            l_convinp1 = l_emb1.flatten().reshape((l_emb1.shape[0], 1, max_len, embedding_size))
            l_convinp2 = l_emb2.flatten().reshape((l_emb2.shape[0], 1, max_len, embedding_size))
            l_pool1 = pooling.apply(conv.apply(l_convinp1)).flatten(2)
            l_pool2 = pooling.apply(conv.apply(l_convinp2)).flatten(2)
            r_convinp1 = r_emb1.flatten().reshape((r_emb1.shape[0], 1, max_len, embedding_size))
            r_convinp2 = r_emb2.flatten().reshape((r_emb2.shape[0], 1, max_len, embedding_size))
            r_pool1 = pooling.apply(conv.apply(r_convinp1)).flatten(2)
            r_pool2 = pooling.apply(conv.apply(r_convinp2)).flatten(2)
            onepools1 = T.concatenate([l_pool1, r_pool1], axis=1)
            onepools2 = T.concatenate([l_pool2, r_pool2], axis=1)
            fv_len += conv.num_filters * 2
            if i == 0:
                outpools1 = onepools1
                outpools2 = onepools2
            else:
                outpools1 = T.concatenate([outpools1, onepools1], axis=1)
                outpools2 = T.concatenate([outpools2, onepools2], axis=1)
        return outpools1, outpools2, fv_len
        
    def build_feature_vector_mentionAsCtx(self, features, DEBUG=False):
        embeddings, _ = self.get_embeddings()
        embedding_size = embeddings.shape[1]
        print embeddings.shape
        lookup = StaticLookupTable(embeddings.shape[0], embeddings.shape[1])
        lookup.allocate()
        lookup.W.set_value(embeddings)
        mlp_in_dim = 0; mntn_fv_len1 = 0; mntn_fv_len2 = 0
        if 'mentions' in features:
            fea = 'mentions'
            mycnf = self._config[fea]
            mymodel = mycnf['model']
            max_len = mycnf['max_len']
            x_ment1 = T.matrix(fea + '1', dtype='int32')
            x_ment2 = T.matrix(fea + '2', dtype='int32')
            embedded_x_1 = lookup.apply(x_ment1) #embedded_x.shape = (batch_size, len(x), embedding_size)
            embedded_x_2 = lookup.apply(x_ment2) #embedded_x.shape = (batch_size, len(x), embedding_size)
            embedded_x_1.name = fea + '_embed1'
            embedded_x_2.name = fea + '_embed2'
            mntn_fv1, mntn_fv_len1 = self.build_mymodel(embedded_x_1, max_len, embedding_size, mycnf, mymodel='mean')
            mntn_fv2, mntn_fv_len2 = self.build_mymodel(embedded_x_2, max_len, embedding_size, mycnf, mymodel='mean')
            mntn_fv1 = mntn_fv1.reshape((embedded_x_1.shape[0], 1, embedding_size))
            mntn_fv2 = mntn_fv2.reshape((embedded_x_2.shape[0], 1, embedding_size))
        logger.info('length of mention vec: %d', mntn_fv_len1)
        fea = 'contexts'
        mycnf = self._config[fea]
        mymodel = mycnf['model']
        max_len = mycnf['max_len']
        x_ctx1 = T.matrix(fea + '1', dtype='int32')
        x_ctx2 = T.matrix(fea + '2', dtype='int32')
        embedded_ctx1 = lookup.apply(x_ctx1) #embedded_x.shape = (batch_size, len(x), embedding_size)
        embedded_ctx2 = lookup.apply(x_ctx2) #embedded_x.shape = (batch_size, len(x), embedding_size)
        embedded_ctx1.name = fea + '_embed1'
        embedded_ctx2.name = fea + '_embed2'
        l_emb1, l_size1, r_emb1, r_size1 = self.split_inp(max_len, embedded_ctx1, mymodel, DEBUG)
        l_emb2, l_size2, r_emb2, r_size2 = self.split_inp(max_len, embedded_ctx2, mymodel, DEBUG)
        if mntn_fv_len1 != 0:
            l_emb1 = T.concatenate([l_emb1, mntn_fv1], axis=1)
            r_emb1 = T.concatenate([mntn_fv1, r_emb1], axis=1)
            l_size1 += 1; r_size1 += 1
        if mntn_fv_len2 != 0:
            l_emb2 = T.concatenate([l_emb2, mntn_fv2], axis=1)
            r_emb2 = T.concatenate([mntn_fv2, r_emb2], axis=1)
            l_size2 += 1; r_size2 += 1
            
        r_emb1 = debug_print(r_emb1, 'r_emb1', DEBUG)
        logger.info('feature size for each input token: %d', embedding_size)
        fv1, fv2, fvlen = self.apply_ctxwise_cnn(l_emb1, l_size1, l_emb2, l_size2, r_emb1, r_size1, r_emb2, r_size2, embedding_size, mycnf)
        logger.info('feature vector length: %d', fvlen)
        fv1 = debug_print(fv1, 'fv1', DEBUG)
        return fv1, fv2, fvlen    

    def build_network(self, num_labels, features, max_len=None, hidden_units=None,
                      l2=None, use_cnn=None, cnn_filter_size=None,
                      cnn_pool_size=None, cnn_num_filters=None, cnn_filter_sizes=None, embedding_size=None, 
                      DEBUG=False):
        """ Build the neural network used for training.

        :param num_labels:      Number of labels to classify
        :param features:        the input features we use
        :param max_len:     Configured window-size
        :param hidden_units:    Number of units in the MLP's hiddden layer
        :returns:               The cost function, the misclassification rate
                                function, the computation graph of the cost
                                function and the prediction function
        """
        logger.info('building the network')
        hidden_units = hidden_units or self._config['hidden_units']
        logger.info('#hidden units: %d', hidden_units)
        # building the feature vector from input.  
        mlp_in_e1, mlp_in_e2, mlp_in_dim = self.build_feature_vector_mentionAsCtx(features)
        logger.info('feature vector size: %d', mlp_in_dim)
        
        mlp = MLP(activations=[Rectifier()],
            dims=[mlp_in_dim, hidden_units],
        )
        initialize([mlp])
        before_out_e1 = mlp.apply(mlp_in_e1)
        before_out_e2 = mlp.apply(mlp_in_e2)
        hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_units, output_dim=num_labels)
        initialize([hidden_to_output])
        linear_output_e1 = hidden_to_output.apply(before_out_e1)
        linear_output_e2 = hidden_to_output.apply(before_out_e2)
        linear_output_e1.name = 'linear_output_e1'
        linear_output_e2.name = 'linear_output_e2'
        
        y_hat_e1 = Logistic(name='logistic1').apply(linear_output_e1)
        y_hat_e2 = Logistic(name='logistic2').apply(linear_output_e2)
        y_hat_e1.name = 'y_hat_e1'
        y_hat_e2.name = 'y_hat_e2'
        y_hat_e1 = debug_print(y_hat_e1, 'y_1', DEBUG)
        return y_hat_e1, y_hat_e2, before_out_e1, before_out_e2
    
    def compute_cost(self, y_hat, target_label='y_types1', num_labels=102, DEBUG=False):
        logger.info('In: compute_cost')
        y_t = T.matrix(target_label, dtype='uint8')
        y_pred = T.argmax(y_hat, axis=1)
        label_of_predicted = debug_print(y_t[T.arange(y_t.shape[0]), y_pred], 'label_of_predicted', DEBUG)
        pat1 = T.mean(label_of_predicted)
        cost = cross_entropy_loss(y_hat, y_t)
        cost.name = 'cost' + target_label
        pat1.name = 'precision@1'
        misclassify_rate = MultiMisclassificationRate().apply(y_t, T.ge(y_hat, 0.5))
        misclassify_rate.name = 'error_rate'
        return cost, pat1, misclassify_rate
    
    
    def build_relation_network(self, ent1=None, ent2=None, entityrepresentationsize=102, multi=True):
        dt = 'float32'  # @UndefinedVariabl
        from logistic_sgd_MIML import LogisticRegressionMIML
        from logistic_sgd import LogisticRegression

        # train network
        rng = numpy.random.RandomState(self.curSeed)
        seed = rng.get_state()[1][0]
        print "seed: " + str(seed)
        # allocate symbolic variables for the data
        xa = T.matrix(u'xa')   # the data is presented as rasterized images
        xb = T.matrix(u'xb')
        xc = T.matrix(u'xc')
        y = T.imatrix(u'y')  # the labels are presented as 1D vector of
                                # [int] labels
        if ent1 == None and ent2 == None:
            ent1 = T.matrix('ent1') # feature for entity 1 (e.g. embedding or type prediction)
            ent2 = T.matrix('ent2') # feature for entity 2
        ishape = [self.rel_representationsize, self.rel_contextsize]  # this is the size of context matrizes
        
        rel_parameters = []
        print '... building the model'
        time1 = time.time()
        layer0a_input = xa.reshape((xa.shape[0], 1, ishape[0], ishape[1]))
        layer0b_input = xb.reshape((xb.shape[0], 1, ishape[0], ishape[1]))
        layer0c_input = xc.reshape((xc.shape[0], 1, ishape[0], ishape[1]))
        
        y = y.reshape((xa.shape[0], ))
        
        # Construct the first convolutional pooling layer:
        filter_shape = (self.rel_nkerns[0], 1, self.rel_representationsize, self.rel_filtersize[1])
        poolsize=(self.rel_pool[0], self.rel_pool[1])
        
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                      numpy.prod(poolsize))
        
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        # the convolution weight matrix
        convW = theano.shared(numpy.asarray(
                   rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                   dtype=dt), name='conv_W')
        
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=dt)
        convB = theano.shared(value=b_values, name='conv_b')
        
        myconv = LeNetConvPoolLayer(rng, W=convW, b=convB,
                    image_shape=(xa.shape[0], 1, ishape[0], ishape[1]),
                    filter_shape=filter_shape, poolsize=poolsize)
        output0a = myconv.apply(input_=layer0a_input)
        output0b = myconv.apply(input_=layer0b_input)
        output0c = myconv.apply(input_=layer0c_input)
        
        layer0flattened = T.concatenate([output0a.flatten(2), output0b.flatten(2), output0c.flatten(2)], axis = 1)
        #.reshape((batch_size, nkerns[0] * sizeAfterPooling))
        rel_parameters += myconv.parameters
        
        if self.rel_randomizeTrain:
            srng = T.shared_randomstreams.RandomStreams(
                      rng.randint(999999))
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask1 = srng.binomial(n=1, p=1-self.rel_corruption_level, size=ent1.shape)
            # The cast is important because
            # int * float32 = float64 which pulls things off the gpu
            ent1Train = ent1 * T.cast(mask1, self.rel_dt)
        
            mask2 = srng.binomial(n = 1, p = 1-self.rel_corruption_level, size = ent2.shape)
            ent2Train = ent2 * T.cast(mask2, dt)
        else:
            ent1Train = ent1
            ent2Train = ent2
        
        if self.rel_normalizeTypes:
            ent1 = ent1 / T.sqrt((ent1**2).sum(axis=1)).dimshuffle(0, 'x')
            ent1Train = ent1Train / T.sqrt((ent1Train**2).sum(axis=1)).dimshuffle(0, 'x')
            ent2 = ent2 / T.sqrt((ent2**2).sum(axis=1)).dimshuffle(0, 'x')
            ent2Train = ent2Train / T.sqrt((ent2Train**2).sum(axis=1)).dimshuffle(0, 'x')
        
        if self.rel_use_type:
            if self.rel_hiddenTypeUnits > 0:
                mlp = MLP(activations=[Tanh()],
                    dims=[entityrepresentationsize, self.rel_hiddenTypeUnits], name='type_',seed=self.curSeed)
                initialize_inout(mlp, entityrepresentationsize, self.rel_hiddenTypeUnits, seed=self.curSeed)
                layer1a_train_output = mlp.apply(ent1Train)
                layer1b_train_output = mlp.apply(ent2Train)
                layer2_inputSize = self.rel_nkerns[0] * self.rel_sizeAfterPooling * 3 + 2 * self.rel_hiddenTypeUnits
                layer2_input_train = T.concatenate([layer0flattened, layer1a_train_output, layer1b_train_output], axis = 1)
                rel_parameters += mlp.parameters
            else:
                layer2_inputSize = self.rel_nkerns[0] * self.rel_sizeAfterPooling * 3 + 2 * entityrepresentationsize
                layer2_input_train = T.concatenate([layer0flattened, ent1Train, ent2Train], axis = 1)
        else:
            print 'NO TYPE features is used in relation'
            layer2_inputSize = self.rel_nkerns[0] * self.rel_sizeAfterPooling * 3
            layer2_input_train = layer0flattened
        
        if self.rel_useHiddenLayer:
            # construct a fully-connected sigmoidal layer
            mlp = MLP(activations=[Tanh()],
                      dims=[layer2_inputSize, self.rel_hiddenunits], name='hidden_rel', seed=self.curSeed)
            initialize_inout(mlp, layer2_inputSize, self.rel_hiddenunits)
            layer2train_output = mlp.apply(layer2_input_train)
            rel_parameters += mlp.parameters
            if multi:
                layer3train = LogisticRegressionMIML(input_=layer2train_output, n_in=self.rel_hiddenunits, n_out=self.rel_numClasses)
            else:
                layer3train = LogisticRegression(input_=layer2train_output, n_in=self.rel_hiddenunits, n_out=self.rel_numClasses)
            rel_parameters += layer3train.parameters
        else:
            # classify the values of the fully-connected sigmoidal layer
            if multi:
                layer3train = LogisticRegressionMIML(input_=layer2_input_train, n_in=layer2_inputSize, n_out=self.rel_numClasses)
            else:
                layer3train = LogisticRegression(input_=layer2_input_train, n_in=layer2_inputSize, n_out=self.rel_numClasses)
            rel_parameters += layer3train.parameters
        
        
        # the cost we minimize during training is the NLL of the model
        cost = layer3train.negative_log_likelihood(y)
        errors = layer3train.errors(y)
        if self.rel_loss != "entropy":
            print "WARNING: so far only cross entropy loss supported"
        cost.name = 'cost_rel'
        return cost, layer3train, y, rel_parameters, errors
    
    def train_joint(self, num_epochs=None, stop_after=None,  batch_sizes=None,
              initial_learning_rate=None, use_bokeh=False,
              checkpoint_path=None, init_lr=None, step_rule=None, DEBUG=False, shuffling=False, multi=False, devset='dev', 
              pre_trained_params=None, wrel=None, l2weight=None):
        """ Train the model and report on the performance.
        :param use_bokeh:       Activate live-plotting to a running bokeh server. Make sure a server has been launched with the `bokeh-server` command!
        :param checkpoint_path:     Save (partially) trained model to this file. If the file already exists, training will resume from the checkpointed state.
        """
        wrel = 1. if wrel is None and 'wrel' not in self._config else wrel if wrel else self._config['wrel']
        logger.info('weight for relation cost is: %f', wrel)
        l2 = l2weight or self._config['l2_regularization']
        step_rule = step_rule or self._config['step_rule']
        init_lr = (init_lr or self._config['init_lr'])
        print l2, step_rule, init_lr
        stop_after = stop_after or self._config['finish_if_no_improvement']
        checkpoint_path = checkpoint_path or self._config['checkpoint_path']
        net_path = self._config['net_path']
        features = self._config['features'] # contexts, mentions
        sources_types = ['entmen1','entmen2', 'y_types1', 'y_types2'] # the sources for data streams
        if 'contexts' in features:
            sources_types.extend(['contexts1', 'contexts2']) 
        if 'mentions' in features:
            sources_types.extend(['mentions1', 'mentions2']) 
        logger.info('sources are %s', sources_types)
        self.conv_inp = False
        
        self.fill_rel_params()
        bagF = open(self.rel_fuelfile + ".entities")
        bag_size_list_train  = cPickle.load(bagF)
        bag_size_list_test = cPickle.load(bagF)
        bagF.close()
        logger.info('loading streams with bags')
        train_stream, num_train = self.get_datastreams_joint(which_set='train', sources_types=sources_types, shuffling=True, multi=multi, bag_size_list=bag_size_list_train)
        dev_stream, num_dev = self.get_datastreams_joint(which_set=devset, sources_types=sources_types, shuffling=False, multi=multi, bag_size_list=bag_size_list_test)
        logger.info('#train: %d, #dev: %d', num_train, num_dev)
        
        ################################
        ### build typing network and computing type vector of ent1 and ent2        
        y_hat_e1, y_hat_e2, hid_e1, hid_e2 = self.build_network(102, features, DEBUG=DEBUG)
        ### calculating cost of typing
        cost1, pat1_1, misclassify_rate1 = self.compute_cost(y_hat_e1, target_label='y_types1', DEBUG=DEBUG)
        cost2, pat1_2, misclassify_rate2 = self.compute_cost(y_hat_e2, target_label='y_types2', DEBUG=DEBUG)
        cost_type = (cost1 + cost2) / 2.
        cost_type.name = 'cost_type'
        pat1 = (pat1_1 + pat1_2) / 2.
        pat1.name = 'p@1'
        misclassify_rate = (misclassify_rate1 + misclassify_rate2) / 2.
        misclassify_rate.name = 'error_rate'
        ### build relation network and computing relation output
        if 'hidden4rel' in self._config and self._config['hidden4rel']:
            logger.info('Using shared hidden layer')
            cost_rel, y_hat_rel_dev, y_rel, rel_params, rel_errors = self.build_relation_network(hid_e1, hid_e2, entityrepresentationsize=self._config['hidden_units'], multi=multi)
        else:
            cost_rel, y_hat_rel_dev, y_rel, rel_params, rel_errors = self.build_relation_network(y_hat_e1, y_hat_e2, multi=multi)
        #Add the two costs
        cost = cost_type + wrel * cost_rel
        rel_errors.name = 'avg_rel_errors'
        
        cg = ComputationGraph(cost)
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        print cg.parameters
        logger.info('weights: %s', weights)
        cost += l2 * l2_norm(weights)
        cost.name = 'cost'
        logger.info('number of parameters in the model: %d', T.sum([p.size for p in cg.parameters]).eval())
        pat1.name = 'prec@1'
        if 'adagrad' in step_rule:
            cnf_step_rule = AdaGrad(init_lr)
        elif 'rms' in step_rule:
            cnf_step_rule = RMSProp(learning_rate=init_lr, decay_rate=0.90)
            cnf_step_rule = CompositeRule([cnf_step_rule, StepClipping(1.0)])
        elif 'sgd' in step_rule:
            cnf_step_rule=Scale(learning_rate=init_lr)
        
        logger.info('net path: %s', net_path)
        theanomap = {'allow_input_downcast': True}
        algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=cnf_step_rule, on_unused_sources='warn',theano_func_kwargs=theanomap)
        gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
        step_norm = aggregation.mean(algorithm.total_step_norm)
        monitored_vars = [cost, cost_type, cost_rel, pat1, misclassify_rate, gradient_norm, step_norm]
        train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_batch=True, before_first_epoch=False, prefix='tra')
        dev_monitor = DataStreamMonitoring(variables=[cost, cost_type, cost_rel, misclassify_rate, pat1, rel_errors], after_batch=False, 
                before_first_epoch=True, data_stream=dev_stream, prefix="dev")

        model = Model(cost)
        
        extensions = [dev_monitor, train_monitor,
 #                       ProgressBar(), 
                        Timing(),
                        TrackTheBest(record_name='dev_cost'),
                        WriteBest(record_name='dev_cost'),
                        WriteBest(record_name='dev_cost_rel'),
                        FinishIfNoImprovementAfter('dev_cost_best_so_far', epochs=stop_after),
                        FinishAfter(after_n_epochs=num_epochs), Printing(),
                        saveload.Load(net_path+'.toload.pkl'),
#                         saveload.Checkpoint(net_path+'.cp.pkl'),
                        ] + track_best('dev_cost', net_path+'.best.pkl')
                        
#         if multi:
#             extensions.append(F1MultiClassesExtension(layer3=y_hat_rel_dev, y=y_rel, model=model, data_stream=dev_stream, num_samples=len(bag_size_list_test), batch_size=bag_size_list_test, every_n_epochs=1,
#                                                       before_first_epoch=True))        
        shapes = [param.get_value().shape for param in cg.parameters]
        logger.info("Parameter shapes: ")
        for shape, count in Counter(shapes).most_common():
            logger.info('    {:15}: {}'.format(shape, count))
        main_loop = MainLoop(model=model,data_stream=train_stream, algorithm=algorithm, extensions=extensions)
        main_loop.run()
    
    def train_rel_yy(self, num_epochs=None, stop_after=None,  batch_sizes=None,
              initial_learning_rate=None, use_bokeh=False,
              checkpoint_path=None, init_lr=None, step_rule=None, DEBUG=False, shuffling=False, multi=False, devset='dev', pre_trained_params=None):
        """ Train the model and report on the performance.
        :param use_bokeh:       Activate live-plotting to a running bokeh server. Make sure a server has been launched with the `bokeh-server` command!
        :param checkpoint_path:     Save (partially) trained model to this file. If the file already exists, training will resume from the checkpointed state.
        """
        l2 = self._config['l2_regularization']
        step_rule = step_rule or self._config['step_rule']
        init_lr = (init_lr or self._config['init_lr'])
        stop_after = stop_after or self._config['finish_if_no_improvement']
        checkpoint_path = checkpoint_path or self._config['checkpoint_path']
        net_path = self._config['net_path']
        features = self._config['features'] # contexts, mentions
        sources_types = ['entmen1','entmen2', 'y_types1', 'y_types2'] # the sources for data streams
        if 'contexts' in features:
            sources_types.extend(['contexts1', 'contexts2']) 
        if 'mentions' in features:
            sources_types.extend(['mentions1', 'mentions2']) 
        logger.info('sources are %s', sources_types)
        self.conv_inp = False
        
        self.fill_rel_params()
        bagF = open(self.rel_fuelfile + ".entities")
        bag_size_list_train  = cPickle.load(bagF)
        bag_size_list_test = cPickle.load(bagF)
        bagF.close()
 
        train_stream, num_train = self.get_datastreams_joint(which_set='train', sources_types=sources_types, shuffling=shuffling, multi=multi, bag_size_list=bag_size_list_train)
        dev_stream, num_dev = self.get_datastreams_joint(which_set='dev', sources_types=sources_types, shuffling=False, multi=multi, bag_size_list=bag_size_list_test)
        logger.info('#train: %d, #dev: %d', num_train, num_dev)
        
        ################################
        ### build typing network and computing type vector of ent1 and ent2        
        y_hat_e1, y_hat_e2, hid_e1, hid_e2 = self.build_network(102, features, DEBUG=DEBUG)
        
        ### build relation network and computing relation output
        if 'hidden4rel' in self._config and self._config['hidden4rel']:
            logger.info('Using shared hidden layer')
            cost_rel, y_hat_rel_dev, y_rel, rel_params, rel_errors = self.build_relation_network(hid_e1, hid_e2, entityrepresentationsize=self._config['hidden_units'])
        else:
            cost_rel, y_hat_rel_dev, y_rel, rel_params, rel_errors = self.build_relation_network(y_hat_e1, y_hat_e2)
        #Add the two costs
        cost = cost_rel
        
        cg = ComputationGraph(cost)
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        print cg.parameters
        logger.info('weights: %s', weights)
        cost += l2 * l2_norm(weights)
        cost.name = 'cost'
        logger.info('number of parameters in the model: %d', T.sum([p.size for p in cg.parameters]).eval())
        if 'adagrad' in step_rule:
            cnf_step_rule = AdaGrad(init_lr)
        elif 'rms' in step_rule:
            cnf_step_rule = RMSProp(learning_rate=init_lr, decay_rate=0.90)
            cnf_step_rule = CompositeRule([cnf_step_rule, StepClipping(0.5)])
        
        logger.info('net path: %s', net_path)
        theanomap = {'allow_input_downcast': True}
        algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=cnf_step_rule, on_unused_sources='warn',theano_func_kwargs=theanomap)
        gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
        step_norm = aggregation.mean(algorithm.total_step_norm)
        monitored_vars = [cost, cost_rel, gradient_norm, step_norm]
        train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_batch=True, before_first_epoch=False, prefix='tra')
        dev_monitor = DataStreamMonitoring(variables=[cost, cost_rel], after_batch=False, 
                before_first_epoch=True, data_stream=dev_stream, prefix="dev")

        model = Model(cost)
        
        extensions = [dev_monitor, train_monitor,
 #                       ProgressBar(), 
                        Timing(),
                        TrackTheBest(record_name='dev_cost'),
                        FinishIfNoImprovementAfter('dev_cost_best_so_far', epochs=stop_after),
                        FinishAfter(after_n_epochs=num_epochs), Printing(),
                        saveload.Load(net_path+'.toload.pkl'),
#                         saveload.Checkpoint(net_path+'.cp.pkl'),
                        ] + track_best('dev_cost', net_path+'.best.pkl')
                        
        if multi:
            extensions.append(F1MultiClassesExtension(layer3=y_hat_rel_dev, y=y_rel, model=model, data_stream=dev_stream, num_samples=len(bag_size_list_test), batch_size=bag_size_list_test, every_n_epochs=1,before_first_epoch=True))
        main_loop = MainLoop(model=model,data_stream=train_stream, algorithm=algorithm, extensions=extensions)
        main_loop.run()
            
    def train_rel_heike(self, num_epochs=None, stop_after=None,  batch_sizes=None,
              initial_learning_rate=None, use_bokeh=False,
              checkpoint_path=None, init_lr=None, step_rule=None, DEBUG=False, shuffling=False, multi=False, devset='dev', pre_trained_params=None, l2weight=None):
        """ Train the model and report on the performance.
        :param use_bokeh:       Activate live-plotting to a running bokeh server. Make sure a server has been launched with the `bokeh-server` command!
        :param checkpoint_path:     Save (partially) trained model to this file. If the file already exists, training will resume from the checkpointed state.
        """
        l2 = l2weight or self._config['l2_regularization']
        step_rule = step_rule or self._config['step_rule']
        init_lr = (init_lr or self._config['init_lr'])
        print l2, step_rule, init_lr
	stop_after = stop_after or self._config['finish_if_no_improvement']
        checkpoint_path = checkpoint_path or self._config['checkpoint_path']
        net_path = self._config['net_path']
        features = self._config['features'] # contexts, mentions
        sources_types = ['entmen1','entmen2', 'y_types1', 'y_types2'] # the sources for data streams
        if 'contexts' in features:
            sources_types.extend(['contexts1', 'contexts2']) 
        if 'mentions' in features:
            sources_types.extend(['mentions1', 'mentions2']) 
        logger.info('sources are %s', sources_types)
        self.conv_inp = False
        
        self.fill_rel_params()
        bagF = open(self.rel_fuelfile + ".entities")
        bag_size_list_train  = cPickle.load(bagF)
        bag_size_list_test = cPickle.load(bagF)
        bagF.close()
 
        train_stream, num_train = self.get_datastreams_joint(which_set='train', sources_types=sources_types, shuffling=shuffling, multi=multi, bag_size_list=bag_size_list_train)
        dev_stream, num_dev = self.get_datastreams_joint(which_set='dev', sources_types=sources_types, shuffling=False, multi=multi, bag_size_list=bag_size_list_test)
        logger.info('#train: %d, #dev: %d', num_train, num_dev)
        
        ################################
        
        ### build relation network and computing relation output
        cost_rel, y_hat_rel_dev, y_rel, rel_params, rel_errors = self.build_relation_network(entityrepresentationsize=self.rel_entitysize, multi=multi)
        cost = cost_rel
        rel_errors.name = 'avg_rel_errors'
        
        cg = ComputationGraph(cost)
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        print cg.parameters
        logger.info('weights: %s', weights)
        cost += l2 * l2_norm(weights)
        cost.name = 'cost'
        logger.info('number of parameters in the model: %d', T.sum([p.size for p in cg.parameters]).eval())
        if 'adagrad' in step_rule:
            cnf_step_rule = AdaGrad(init_lr)
        elif 'rms' in step_rule:
            cnf_step_rule = RMSProp(learning_rate=init_lr, decay_rate=0.90)
            cnf_step_rule = CompositeRule([cnf_step_rule, StepClipping(0.5)])
        elif 'sgd' in step_rule:
            cnf_step_rule=Scale(learning_rate=init_lr)
            cnf_step_rule = CompositeRule([cnf_step_rule, StepClipping(1.0)])
        
        logger.info('net path: %s', net_path)
        theanomap = {'allow_input_downcast': True}
        algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=cnf_step_rule, on_unused_sources='warn',theano_func_kwargs=theanomap)
        gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
        step_norm = aggregation.mean(algorithm.total_step_norm)
        monitored_vars = [cost, gradient_norm, step_norm]
        train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_batch=True, before_first_epoch=False, prefix='tra')
        dev_monitor = DataStreamMonitoring(variables=[cost, rel_errors], after_batch=False, 
                before_first_epoch=True, data_stream=dev_stream, prefix="dev")

        model = Model(cost)
        
        extensions = [dev_monitor, train_monitor,
 #                       ProgressBar(), 
                        Timing(),
                        TrackTheBest(record_name='dev_cost'),
                        FinishIfNoImprovementAfter('dev_cost_best_so_far', epochs=stop_after),
                        FinishAfter(after_n_epochs=num_epochs), Printing(),
                        saveload.Load(net_path+'.toload.pkl'),
#                         saveload.Checkpoint(net_path+'.cp.pkl'),
                        ] + track_best('dev_cost', net_path+'.best.pkl')
#         if multi:
#             extensions.append(F1MultiClassesExtension(layer3=y_hat_rel_dev, y=y_rel, model=model, 
#                                  data_stream=dev_stream, num_samples=len(bag_size_list_test), batch_size=bag_size_list_test, every_n_epochs=1
#                                  , before_first_epoch=True))
        shapes = [param.get_value().shape for param in cg.parameters]
        logger.info("Parameter shapes: ")
        for shape, count in Counter(shapes).most_common():
            logger.info('    {:15}: {}'.format(shape, count))    
        main_loop = MainLoop(model=model,data_stream=train_stream, algorithm=algorithm, extensions=extensions)
        main_loop.run()
    
    def test_rel_heike(self, shuffling=False, multi=False, DEBUG=False, outPRfile=None):
        net_path = self._config['net_path']
        features = self._config['features'] # contexts, mentions
        sources_types = ['entmen1','entmen2', 'y_types1', 'y_types2'] # the sources for data streams
        if 'contexts' in features:
            sources_types.extend(['contexts1', 'contexts2']) 
        if 'mentions' in features:
            sources_types.extend(['mentions1', 'mentions2']) 
        logger.info('sources are %s', sources_types)
        self.conv_inp = False
        
        self.fill_rel_params()
        self.rel_fuelfile = self.rel_test_fulefile
        print self.rel_fuelfile
        bagF = open(self.rel_fuelfile + ".entities")
        bag_size_list_train = cPickle.load(bagF)
        bag_size_list_test = cPickle.load(bagF)
        bagF.close()
        print bag_size_list_test
        test_stream, num_test = self.get_datastreams_joint(which_set='test', sources_types=sources_types, shuffling=False, multi=multi, bag_size_list=bag_size_list_test, num_samples=None) 
#         test_stream, num_test = self.get_datastreams_joint(which_set='train', sources=sources, shuffling=False, multi=multi, bag_size_list=bag_size_list_train)
        logger.info('#test: %d', num_test)
        
        ################################
        ### calculating cost of typing
        cost_rel, y_hat_rel_test, y_rel, rel_params, rel_errors = self.build_relation_network(entityrepresentationsize=self.rel_entitysize, multi=multi)
        cost = cost_rel
        model = Model([cost])
        f = open(net_path + ".best.pkl")
        old_main_loop = load(f)
        f.close()
        old_model = old_main_loop.model
        model.set_parameter_values(old_model.get_parameter_values())
        
        extensions = []
        algorithm = None
	extensions.append(GetPRcurve(layer3=y_hat_rel_test, y=y_rel, model=model, data_stream=test_stream, 
                                      num_samples=len(bag_size_list_test), batch_size=bag_size_list_test, before_training=True, pr_out_file=outPRfile))
#         extensions.append(ALLModelResultsMI(layer3=y_hat_rel_test, y=y_rel, model=model, 
#                                 data_stream=test_stream, num_samples=len(bag_size_list_test), batch_size=bag_size_list_test, before_training=True, 
#                                 out_prob_file=outPRfile+'.test_probs', out_truth_file=outPRfile+'.test_truth'))
#       extensions.append(ModelResultsMI(layer3=y_hat_rel_test, y=y_rel, model=model, 
#                                data_stream=test_stream, num_samples=len(bag_size_list_test), batch_size=bag_size_list_test, before_training=True))
        my_loop = MainLoop(model=model,
                           data_stream=test_stream,
                           algorithm=algorithm,
                           extensions=extensions)
        for extension in my_loop.extensions:
            extension.main_loop = my_loop
        my_loop._run_extensions('before_training')
        
    def test_rel(self, shuffling=False, multi=False, DEBUG=False, outPRfile=None):
        net_path = self._config['net_path']
        features = self._config['features'] # contexts, mentions
        sources_types = ['entmen1','entmen2', 'y_types1', 'y_types2'] # the sources for data streams
        if 'contexts' in features:
            sources_types.extend(['contexts1', 'contexts2']) 
        if 'mentions' in features:
            sources_types.extend(['mentions1', 'mentions2']) 
        logger.info('sources are %s', sources_types)
        self.conv_inp = False
        
        self.fill_rel_params()
        self.rel_fuelfile += '.test'
        print self.rel_fuelfile
        bagF = open(self.rel_fuelfile + ".entities")
        _ = cPickle.load(bagF)
        bag_size_list_test = cPickle.load(bagF)
        bagF.close()
        print bag_size_list_test
        test_stream, num_test = self.get_datastreams_joint(which_set='test', sources_types=sources_types, shuffling=False, multi=multi, bag_size_list=bag_size_list_test) 
        
        ################################
        ### build typing network and computing type vector of ent1 and ent2        
        y_hat_e1, y_hat_e2, hid_e1, hid_e2 = self.build_network(102, features, DEBUG=DEBUG)
        ### calculating cost of typing
        cost1, pat1_1, misclassify_rate1 = self.compute_cost(y_hat_e1, target_label='y_types1')
        cost2, pat1_2, misclassify_rate2 = self.compute_cost(y_hat_e2, target_label='y_types2')
        cost_type = (cost1 + cost2) / 2.
        pat1 = (pat1_1 + pat1_2) / 2.
        pat1.name = 'p@1'
        misclassify_rate = (misclassify_rate1 + misclassify_rate2) / 2.
        misclassify_rate.name = 'error_rate'
        ### build relation network and computing relation output
        if 'hidden4rel' in self._config and self._config['hidden4rel']:
            logger.info('Using shared hidden layer')
            cost_rel, y_hat_rel_test, y_rel, rel_param, rel_errorss = self.build_relation_network(hid_e1, hid_e2, entityrepresentationsize=self._config['hidden_units'], multi=multi)
        else:
            cost_rel, y_hat_rel_test, y_rel, rel_params, rel_errors = self.build_relation_network(y_hat_e1, y_hat_e2, multi=multi)
        cost = cost_type + cost_rel
        model = Model([cost])
        f = open(net_path + ".best.pkl")
        old_main_loop = load(f)
        f.close()
        old_model = old_main_loop.model
        model.set_parameter_values(old_model.get_parameter_values())
        
        extensions = []
        algorithm = None
        extensions.append(GetPRcurve(layer3=y_hat_rel_test, y=y_rel, model=model, data_stream=test_stream, 
                                      num_samples=len(bag_size_list_test), batch_size=bag_size_list_test, before_training=True, pr_out_file=outPRfile))
        #extensions.append(ModelResultsMI(layer3=y_hat_rel_test, y=y_rel, model=model, 
        #                         data_stream=test_stream, num_samples=len(bag_size_list_test), batch_size=bag_size_list_test, before_training=True))
        my_loop = MainLoop(model=model,
                           data_stream=test_stream,
                           algorithm=algorithm,
                           extensions=extensions)
        for extension in my_loop.extensions:
            extension.main_loop = my_loop
        my_loop._run_extensions('before_training')
        
    def aggregate_scores(self):
        devsampledfile = self._config['devcontexts'] 
        testsampledfile = self._config['testcontexts']
        dev_lines, e2freq_dev = load_lines_info(devsampledfile, self.t2i)
        test_lines, e2freq_test = load_lines_info(testsampledfile, self.t2i)
        self.dev_big_matrix = numpy.load(self._config['devscores'] +'.npy') if self.dev_big_matrix == None else self.dev_big_matrix
        self.test_big_matrix = numpy.load(self._config['testscores'] +'.npy') if self.test_big_matrix == None else self.test_big_matrix
        logger.info('Building the big matrix...')
        dev_type2ent_scores = build_type2entmatrix(self.dev_big_matrix, dev_lines, self.t2i)
        test_type2ent_scores = build_type2entmatrix(self.test_big_matrix, test_lines, self.t2i)
        logger.info('aggregating scores for entities')
        dev_type2ent_scores_agg = big2small(dev_type2ent_scores, self.t2i)
        test_type2ent_scores_agg = big2small(test_type2ent_scores, self.t2i)
        write_small(self._config['matrixdev'], dev_type2ent_scores_agg, self.t2i, e2freq_dev)
        write_small(self._config['matrixtest'], test_type2ent_scores_agg, self.t2i, e2freq_test)
    
    def eval(self, configfile, outtype=None):
        self.aggregate_scores()
        cmd = 'python /mounts/Users/student/yadollah/git/cis_kbp/src/classification/eval/matrix2measures_ents.py ' + configfile + ' > ' + configfile + '.meas.ents'
        outtype = outtype or 'multi'
        cmd += '.' + outtype
        p = Popen(cmd, shell=True); 
        p.wait() 
        cmd = 'python /mounts/Users/student/yadollah/git/cis_kbp/src/classification/eval/matrix2measures_types.py ' + configfile + ' > ' + configfile + '.meas.types'
        outtype = outtype or 'multi'
        cmd += '.' + outtype
        p = Popen(cmd, shell=True); 
        p.wait() 
        
    def test_joint(self, net_name=None):
        net_name = net_name or 'best'
        print '***', net_name 
        net_path = self._config['net_path'] + '.' + net_name + '.pkl'
        logger.info('applying the model on the test and dev data, model=%s', net_path)
        features = self._config['features'] # contexts, mentions
        self.conv_inp = False
        sources = ['entmen1','entmen2', 'y_types1', 'y_types2'] # the sources for data streams 
        if 'contexts' in features:
            sources.extend(['contexts1', 'contexts2']) 
        if 'mentions' in features:
            sources.extend(['mentions1', 'mentions2']) 
        logger.info('sources are %s', sources)
        main_loop = load(net_path)
        logger.info('Model loaded. Building prediction function...')
        model = main_loop.model
        logger.info(model.inputs)
        theinputs = [inp for inp in model.inputs if inp.name != 'y_types1' and inp.name != 'y_types2' and inp.name not in ['xa', 'xb', 'xc', 'ent1', 'ent2', 'y']]
        scores_e1 = [v for v in model.variables if v.name == 'y_hat_e1'][0]
        scores_e2 = [v for v in model.variables if v.name == 'y_hat_e2'][0]
        predict = theano.function(theinputs, [scores_e1, scores_e2], on_unused_input='warn')
        test_stream, num_samples_test = self.get_datastreams(which_set='test', shuffling=False, sources=sources, multi=False)
        dev_stream, num_samples_dev = self.get_datastreams(which_set='dev', shuffling=False, sources=sources, multi=False) #TODO: take care of dev big
#         test_stream, num_samples_test = self.get_datastreams_joint(which_set='test', shuffling=False, sources_types=sources, multi=False)
#         dev_stream, num_samples_dev = self.get_datastreams(which_set='dev', shuffling=False, sources_types=sources, multi=False) #TODO: take care of dev big
        logger.info('#size dev: %d, size test: %d', num_samples_dev, num_samples_test)
        self.dev_big_matrix,_ = self.get_scores(dev_stream, num_samples_dev, predict, theinputs)
        numpy.save(self._config['devscores'], self.dev_big_matrix)
        self.test_big_matrix,_ = self.get_scores(test_stream, num_samples_test, predict, theinputs)
        numpy.save(self._config['testscores'], self.test_big_matrix)
        
    def get_scores(self, data_stream, num_samples, predict, theinputs, num_labels=102):
        epoch_iter = data_stream.get_epoch_iterator(as_dict=True)
        goods = 0.
        idx = 0
        scores_matrix = numpy.zeros(shape=(num_samples*2, num_labels), dtype='float32')
        true_y_matrix = numpy.zeros(shape=(num_samples*2, num_labels), dtype='int8')
        while idx < num_samples:
            src2vals  = epoch_iter.next()
            inp = [src2vals[src.name] for src in theinputs]
            scores1, scores2 = predict(*inp)
            y_curr1 = src2vals['y_types1']
            y_curr2 = src2vals['y_types2']
            for j in range(len(scores1)):
                maxtype_ix1 = argmax(scores1[j])
                maxtype_ix2 = argmax(scores2[j])
                if y_curr1[j][maxtype_ix1] == 1:
                    goods += 1
                if y_curr2[j][maxtype_ix2] == 1:
                    goods += 1
                true_y_matrix[idx] = y_curr1[j]
                true_y_matrix[idx+1] = y_curr2[j]
                scores_matrix[idx] = scores1[j]
                scores_matrix[idx+1] = scores2[j]
                idx += 2    
        logger.info('P@1 is = %f ', goods / (len(scores_matrix)))
        return scores_matrix, true_y_matrix
    
    def test_evaluate(self, configfile):
        cmd = 'nice -n 2 python /mounts/Users/student/yadollah/git/cis_kbp/src/classification/nn/cm/multi_instance/train2level.py --config ' + configfile + \
            ' --test true --eval true --outtype cm --net contextmodel' 
        p = Popen(cmd, shell=True, stdout=subprocess.PIPE);
#         p.wait()
        return 
    

    def compute_f_measure(self, test_big_matrix, true_y_matrix, theta=None):
        theta = theta or [0.2 for _ in range(true_y_matrix.shape[1])]
        ff = "{:10.3f}"
        goods = 0.; bads = 0.; totals = 0. 
        pmacro = 0.; rmacro = 0.; fmacro = 0.
        stricts = 0.            
        for idx in range(len(test_big_matrix)):
            scores = test_big_matrix[idx]
            scores = normalize(scores)
            labels = true_y_matrix[idx]
            score_labels = [(scores[i], labels[i]) for i in range(len(scores))]
            (good, bad, total) = computeFscore(score_labels, theta)
            goods += good; bads += bad; totals += total
            (p,r,f) = calcPRF(good, bad, total)
            pmacro += p; rmacro += r; fmacro += f
            if good == total and bad == 0: 
                stricts += 1
        (pr , re, f ) = calcPRF(goods, bads, totals)
        print goods, bads, totals
        print 'Micro results: '
        print 'Prec: ', ff.format(pr), ' Reca: ', ff.format(re), ' F1: ',ff.format(f)
        pmacro /= len(test_big_matrix);rmacro /= len(test_big_matrix);fmacro /= len(test_big_matrix)
        print 'Macro results: '
        print 'Prec: ', ff.format(pmacro), ' Reca: ', ff.format(rmacro), ' F1: ',ff.format(fmacro)
        print 'Strict result:'
        print 'Strict f1: ', ff.format(stricts/len(test_big_matrix))
    

    def find_label_theresholds(self, pred_matrix, gold_matrix):
        label_theta_list = []
        for i in range(gold_matrix.shape[1]):
            label_theta_list.append(find_best_theta(pred_matrix[:,i], gold_matrix[:,i]))
        print label_theta_list
        return label_theta_list
    
    def eval_rel(self, pr_file):
        cmd = "cat " + pr_file + " | grep \"P = \" | cut -f 2 | cut -d \" \" -f 3,6 | perl -pe 's/\,//g' | uniq > " + pr_file +".prepared"
        p = Popen(cmd, shell=True, stdout=subprocess.PIPE);
        p.wait()
	f = open(pr_file + '.prepared', 'r')
	last_recall = -1
	pList = []
	for line in f:
	    line = line.strip()
	    p,r = line.split()
	    p = float(p)
            r = float(r)
		    # assumption: if there was a positive gold label, the recall value changes
            if last_recall == r or last_recall == -1:
	        if r == 1:
				      # only case where there could be a positive gold label without a recall change
                    print "WARNING: maybe missed a positive gold label"
		last_recall = r
		continue
	    pList.append(p)
	    last_recall = r
	mapval = 0.0
	if len(pList) > 0:
		mapval = numpy.mean(numpy.asarray(pList))
		print numpy.mean(numpy.asarray(pList))
        with open(pr_file+'.prepared.MAPvalue', 'w') as fp:
		fp.write(str(mapval))
	
    def eval_rel_map(self, AP_file, out_ap_vec_file):
        probsfile = AP_file + '.test_probs.npy'
        truthfile = AP_file + '.test_truth.npy'
        probs_matrix = numpy.load(probsfile)
        truth_matrix = numpy.load(truthfile)
        AP_list = []
        for i in range(probs_matrix.shape[1]):
            prob_truth_pairs = []
            for j in range(len(probs_matrix)):
                prob_truth_pairs.append((probs_matrix[j][i], truth_matrix[j][i]))
            AP_list.append(calc_AP(prob_truth_pairs))
        mapvalue = numpy.mean(numpy.asarray(AP_list))
        with open(AP_file + '.MAP', 'w') as fp:
            fp.write(str(mapvalue))
            print mapvalue 
        with open(out_ap_vec_file, 'w') as fp:
            for ap in AP_list:
                fp.write(str(ap) + '\n')
                
            
        
def get_argument_parser():
    """ Construct a parser for the command-line arguments. """
    """ Construct a parser for the command-line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", help="Path to configuration file")
    
    parser.add_argument(
        "--test", "-t", type=bool, help="Applying the model on the test data, or not")

    parser.add_argument(
        "--test_rel", "-tere", type=bool, help="Applying the model on the test data, or not")
    parser.add_argument(
        "--test_rel_heike", "-tsreh", type=bool, help="Applying the model on the test data, or not")
    
    parser.add_argument(
        "--eval_rel", "-evre", type=bool, help="evaluation for relation")
    
    parser.add_argument(
        "--testeval", "-te", type=bool, help="Applying the model on the test data, or not")
    
    parser.add_argument(
        "--train", "-tr", type=bool, help="Training the model on the test data, or not")

    parser.add_argument(
        "--train_rel", "-trrel", type=bool, help="Training the model on the test data, or not")
 
    parser.add_argument(
        "--eval", "-ev", type=bool, help="Evaluating the model on classification task")
    parser.add_argument(
        "--apply", "-ap", type=bool, help="Applying the trained model on a test sample.")
 
    parser.add_argument(
        "--multi", "-mu", type=bool, help="Applying the trained model on a test sample.")
 
    parser.add_argument(
        "--dsname", "-ds", help="Name of the evaluation dataset to be applied.e.g., figer")
 
    parser.add_argument(
        "--debug", "-de", type=bool, help="Evaluating the model on classification task")
 
    parser.add_argument(
        "--outtype", "-outtype", help="type of the model training")
    
    parser.add_argument(
        "--net", "-net", help="path to the trained model")
    

    parser.add_argument(
        "--samples", "-s", help="Path to HDF file with datasets")
    
    parser.add_argument(
        "--embeddings", "-e", help="Path to HDF files with embeddings.")
    
    parser.add_argument(
        "--checkpoint", "-cp",
        help="Path for checkpointing, training will resume from the "
             "checkpointed state if the file already exists")
    parser.add_argument(
        "--max-len", "-w", type=int,
        help="Width of the window over the contexts")
    parser.add_argument(
        "--hidden-units", "-hi", type=int,
        help="Number of hidden units for MLP")
    parser.add_argument(
        "--use-cnn",  "-cnn", action="store_true",
        help="Use a CNN in addition to the MLP")
    parser.add_argument(
        "--max-epochs",  "-me", type=int,
        help="Stop the training process after this many epochs")
    parser.add_argument(
        "--wrel",  "-wr", type=float,
        help="weight of relation cost.")
    parser.add_argument(
        "--live-plotting",  "-lp", action="store_true",
        help="Enable live-plotting to a running Bokeh server (start with "
             "`bokeh-server`)")
    return parser


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    logger.info('args: %s', args)
    if not args.config and (not args.samples or not args.embeddings):
        print("Please provide either a config file (--config) or the paths "
              "to the samples and embeddings files. "
              "(--samples, --embeddings).")
        sys.exit(1)
    trainer = JointUnaryBinary.from_config(args.config)
    trainer._config['hidden_units'] = args.hidden_units if args.hidden_units else trainer._config['hidden_units']
    if args.max_len:
        trainer._config['max_len'] = args.max_len
    if args.apply:
        trainer.apply(net_name=args.net, dsname=args.dsname)
        sys.exit()
    multi = False
    print args.multi
    if args.multi:
        from logistic_sgd_MIML import LogisticRegression
        multi = True
        print 'multi'
    else:
        from logistic_sgd import LogisticRegression
    if args.train:
        trainer.train_joint(num_epochs=args.max_epochs, use_bokeh=args.live_plotting,
                    checkpoint_path=args.checkpoint, DEBUG=False, shuffling=True, multi=args.multi, devset='dev')
        
    if args.train_rel:
        trainer.train_rel_heike(num_epochs=args.max_epochs, use_bokeh=args.live_plotting,
                    checkpoint_path=args.checkpoint, DEBUG=False, shuffling=True, multi=args.multi, devset='dev')
    if args.test:
        trainer.test_joint(net_name=args.net)
    if args.test_rel:
        trainer.test_rel(multi=True, outPRfile=args.config+'.PRrel')
    if args.test_rel_heike:
        trainer.test_rel_heike(multi=True, outPRfile=args.config+'.PRrel')
    if args.testeval:    
        trainer.test_evaluate(args.config)
    if args.eval:
        trainer.eval(args.config, args.outtype)
    if args.eval_rel:
        trainer.eval_rel_map(args.config+'.PRrel', out_ap_vec_file=args.config+'.average_precisions_vec')
        
#     if args.retrain:
#         old_main_loop = load(trainer._config['net'])
#         old_model = old_main_loop.model
#         old_parameter_values = old_model.get_parameter_values()
#         trainer.train(num_epochs=args.max_epochs, use_bokeh=args.live_plotting,
#                     checkpoint_path=args.checkpoint, DEBUG=False, shuffling=True, multi=False, pre_trained_params=old_parameter_values)
#         logger.info('Copying net for re-retraining...')
#   
