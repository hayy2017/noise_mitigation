'''
Created on Feb 9, 2016

@author: yadollah
'''
import argparse
import logging
import six
import subprocess
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('train.py')
import os
import sys
from subprocess import Popen
import h5py
import numpy
import theano
import theano.tensor as T
import yaml
from blocks.serialization import load
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing, saveload
from blocks.extensions.saveload import Checkpoint, Load
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp, AdaGrad, Scale, AdaDelta, Momentum, Adam
from blocks.bricks import MLP, Tanh, Logistic, WEIGHT, Rectifier
from blocks.bricks.base import Parameters, application
from blocks.bricks.conv import (ConvolutionalSequence, Flattener, MaxPooling,
                                ConvolutionalActivation)
from blocks.bricks.cost import BinaryCrossEntropy, Cost
from blocks.bricks.lookup import LookupTable
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.model import Model
from blocks.utils import shared_floatx_nans
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme, IterationScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer, Merge
from model import WindowTransformer,\
    MultiMisclassificationRate, StaticLookupTable, initialize, \
    LettersTransposer, create_cnn_general, SEQ_INPUTS, create_rec, REC_MODELS,\
    GenerateNegPosTransformer, initialize_identity
from blocks.theano_expressions import l2_norm
from myutils import debug_print, fillt2i,\
    build_type2entmatrix, big2small, write_small, load_lines_info, MyPool,\
    computeFscore, calcPRF, softmax, normalize, write_small_multi,\
    big2small_avgs
from model import cross_entropy_loss, track_best,\
    MainLoop, ranking_loss, softmax_layer
from numpy import argmax
from _collections import defaultdict
import string
from blocks.bricks import WEIGHT, MLP, Rectifier, Tanh, Linear, Softmax, Logistic
from picklable_itertools import chain, repeat, imap, iter_
from picklable_itertools.base import BaseItertool
from collections import Iterable

dirname, _ = os.path.split(os.path.abspath(__file__))

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

    def __init__(self, examples, batch_size_list, shuffling=False):
        if isinstance(examples, Iterable):
            self.indices = examples
        else:
            self.indices = xrange(examples)
        self.batch_size_list = batch_size_list
        self.rng = numpy.random.RandomState(987654)
        self.shuffling = shuffling
        
    def get_request_iterator(self):
#         return imap(list, My_partition_all(self.batch_size_list, self.indices))
        tmp = list(My_partition_all(self.batch_size_list, self.indices))
        if self.shuffling:
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

class EntityTypingTrainer(object):
    """ Trains a classifier for an entity typing task.

    Uses either a standard Multi-Layer Perceptron with a single hidden layer
    or in addition a Convolutional Neural Network with one convoluation and
    one max-pooling layer.
    """
    @staticmethod
    def from_config( config_path):
        """ Instantiate the trainer from an existing configuration.

        :param config_path:     Path to YAML file with configuration\
        :returns:               The initialized EntityTypingTrainer instance
        """
        with open(config_path) as fp:
            config = yaml.load(fp)
        trainer = EntityTypingTrainer(config['dsdir'])
        trainer._config.update(config)
        trainer.t2i,_ = fillt2i(trainer._config['typefile'])
        trainer.dev_big_matrix = None
        trainer.test_big_matrix = None
        
        trainer.set2bag_len_list = {}
        trainer.set2bag_len_list['train'] = trainer.get_bag_sizes('train', 'entmentions')
        trainer.set2bag_len_list['dev'] = trainer.get_bag_sizes('dev', 'entmentions')
        trainer.set2bag_len_list['devbig'] = trainer.get_bag_sizes('devbig', 'entmentions')
#         print trainer.set2bag_len_list['devbig'][0:10]
        trainer.set2bag_len_list['test'] = trainer.get_bag_sizes('test', 'entmentions')
        print len(trainer.set2bag_len_list['devbig']), len(trainer.set2bag_len_list['test'])
        return trainer

    def __init__(self, samples_path, embeddings_path=""):
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
        
    def get_datastreams(self, which_set='', sources=('mentions', 'contexts', 'entmentions', 'targets'), max_len=None, shuffling=False,
                        batch_size=None, multi=False, num_examples=None):
        """ Load all datastreams.
        :param max_len:     The desired window size
        """
        max_len = max_len or self._config['contexts']['max_len']
        batch_size = batch_size or self._config['batch_sizes'][which_set]
        streams = []
        for src in sources:
            print src, self._config['dsdir'] + '_' + src + '.hdf'
            dataset = H5PYDataset(self._config['dsdir'] + '_' + src + '.hdf', which_sets=(which_set,), load_in_memory=True)
            num_examples = num_examples or dataset.num_examples
            if shuffling == True:
#                 iteration_scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size)
                iteration_scheme = ShuffledScheme(examples=num_examples, batch_size=batch_size)
            else:
                if multi:
                    batch_size_list = self.set2bag_len_list[which_set] #[2, 98] + [100] * (int((dataset.num_examples / 100)) - 1)
                    iteration_scheme = MySequentialScheme(examples=num_examples, batch_size_list=batch_size_list)
                else:
                    iteration_scheme = SequentialScheme(examples=num_examples, batch_size=batch_size)
            data_stream = DataStream(dataset=dataset, iteration_scheme=iteration_scheme)
            streams.append(data_stream)
        stream = Merge(streams, sources)
        stream = WindowTransformer(stream, ignore_pivot=True, margin=max_len//2) 
        for src in sources:
            if src in SEQ_INPUTS and self._config[src]['model'] in REC_MODELS:
                logger.info('transposing %s ...', src)
                stream = LettersTransposer(stream, which_sources=src)# Required because Recurrent bricks receive as input [sequence, batch,# features]
        return stream, num_examples

    def get_datastreams_convinp(self, which_set='train', sources=('contexts', 'entmentions', 'targets'), 
                        batch_size=None, shuffling=False, multi=False, num_examples=None):
        batch_size = batch_size or self._config['batch_sizes'][which_set] 
        streams = []
        for src in sources:
            hdffile = self._config['dsdir'] + '_' + src + '.hdf'
            logger.info('src %s, shuffling = %s, isMulti = %s', src, shuffling, multi)
            dataset = H5PYDataset(hdffile, which_sets=(which_set,), load_in_memory=True)
            num_examples = num_examples or dataset.num_examples
            if shuffling == True:
                iteration_scheme = ShuffledScheme(examples=num_examples, batch_size=batch_size)
            else:
                if multi:
                    batch_size_list = self.set2bag_len_list[which_set] #[2, 98] + [100] * (int((dataset.num_examples / 100)) - 1)
                    iteration_scheme = MySequentialScheme(examples=num_examples, batch_size_list=batch_size_list, shuffling=shuffling)
                else:
                    iteration_scheme = SequentialScheme(examples=num_examples, batch_size=batch_size)
            data_stream = DataStream(dataset=dataset, iteration_scheme=iteration_scheme)
            streams.append(data_stream)
        stream = Merge(streams, sources)
        return stream, num_examples

    def get_embeddings(self):
        """ Load the word embedding vectors and their words.

        :returns:   The embedding vectors and the corresponding words
        :rtype:     (np.array, np.array) tuple
        """
        with h5py.File(self._config['dsdir'] + '_embeddings.hdf', "r") as fp:
            return (fp.get('vectors').value.astype('float32'),
                    fp.get('words').value)
        logger.info('loading embeddings finished!')


    def build_mymodel(self, mymodel, embedded_x, max_len, embedding_size, mycnf, use_conv_inp=False):
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

    def build_feature_vector(self, features, DEBUG=False):
        embeddings, _ = self.get_embeddings()
        embedding_size = embeddings.shape[1]
        print embeddings.shape
        lookup = StaticLookupTable(embeddings.shape[0], embeddings.shape[1])
        lookup.allocate()
        lookup.W.set_value(embeddings)
        mlp_in = None
        mlp_in_dim = 0
        for fea in features:
            logger.info('building network for features: %s', fea)
            mycnf = self._config[fea]
            mymodel = mycnf['model']
            max_len = mycnf['max_len']
            x = T.matrix(fea, dtype='int32')
            embedded_x = lookup.apply(x) #embedded_x.shape = (batch_size, len(x), embedding_size)
            embedded_x.name = fea + '_embed'
            if 'contexts' in fea:
                l_emb, l_size, r_emb, r_size = self.split_inp(max_len, embedded_x, mymodel, DEBUG)
                l_fv, l_fvlen = self.build_mymodel(mymodel, l_emb, l_size, embedding_size, mycnf)
                r_fv, r_fvlen = self.build_mymodel(mymodel, r_emb, r_size, embedding_size, mycnf)
                fv = T.concatenate([l_fv, r_fv], axis=1)
                fv_len = l_fvlen + r_fvlen
            else:
                fv, fv_len = self.build_mymodel(mymodel, embedded_x, max_len, embedding_size, mycnf)
            if mlp_in != None:
                mlp_in = T.concatenate([mlp_in, fv], axis=1)
            else:
                mlp_in = fv
            mlp_in_dim += fv_len
        return mlp_in, mlp_in_dim


    def build_feature_vector_convinp(self, features, DEBUG=False):
        embedding_size = self._config['embedding_size']
        mlp_in = None
        mlp_in_dim = 0
        for fea in features:
            logger.info('building network for features: %s', fea)
            mycnf = self._config[fea]
            mymodel = mycnf['model']
            max_len = mycnf['max_len']
            if 'contexts' in fea:
                x = T.tensor4(fea, dtype='float32')
                l_emb, l_size, r_emb, r_size = self.split_inp(max_len, x, mymodel, DEBUG, use_conv_inp=True)
                l_fv, l_fvlen = self.build_mymodel(mymodel, l_emb, l_size, embedding_size, mycnf, use_conv_inp=True)
                r_fv, r_fvlen = self.build_mymodel(mymodel, r_emb, r_size, embedding_size, mycnf, use_conv_inp=True)
                fv = T.concatenate([l_fv, r_fv], axis=1)
                fv_len = l_fvlen + r_fvlen
            else:
                x = T.matrix(fea, dtype='int32')
                embeddings, _ = self.get_embeddings()
                print embeddings.shape
                lookup = StaticLookupTable(embeddings.shape[0], embeddings.shape[1])
                lookup.allocate()
                lookup.W.set_value(embeddings)
                embedded_x = lookup.apply(x) #embedded_x.shape = (batch_size, len(x), embedding_size)
                embedded_x = debug_print(embedded_x, 'embedded_x', DEBUG)
                embedded_x.name = fea + '_embed'
                fv, fv_len = self.build_mymodel(mymodel, embedded_x, max_len, embedding_size, mycnf)
            if mlp_in != None:
                mlp_in = T.concatenate([mlp_in, fv], axis=1)
            else:
                mlp_in = fv
            mlp_in_dim += fv_len
        return mlp_in, mlp_in_dim
    
    
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
        if self.conv_inp:
            mlp_in, mlp_in_dim = self.build_feature_vector_convinp(features)
        else:
            mlp_in, mlp_in_dim = self.build_feature_vector(features)

        logger.info('feature vector size: %d', mlp_in_dim)
        mlp = MLP(activations=[Tanh()],
            dims=[mlp_in_dim, hidden_units],
        )
        initialize([mlp])
        before_out = debug_print(mlp.apply(mlp_in), 'before_out', DEBUG) #phi(c) is the before_out
        hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_units, output_dim=num_labels)
        initialize([hidden_to_output])
        linear_output = hidden_to_output.apply(before_out)
        linear_output.name = 'linear_output'
        y_hat = Logistic().apply(linear_output)
        y_hat.name = 'y_hat'
        
        return y_hat, before_out, hidden_units, hidden_to_output
    


    def compute_cost_multi_instance_max(self, y_hat, num_labels=102, DEBUG=False):
        logger.info('In: compute_cost_multi_instance_max')
        y = T.matrix('targets', dtype='int32')
        jj = T.argmax(y_hat, axis=0) # jj should be the indices of contexts with highest score (x, y_i)
        maxpredvec = debug_print(y_hat[jj, T.arange(num_labels)], 'max_y_hat', DEBUG)
#         cost = -T.sum(T.log(maxpredvec) * y[0][0:])
        entity_y = debug_print(y[0][0:], 'entity_y', DEBUG)
        cost = cross_entropy_loss(maxpredvec, entity_y)
        cost = debug_print(cost, 'cost', DEBUG)
        pred = T.argmax(maxpredvec)
        label_of_top = entity_y[pred]
        pat1 = T.mean(label_of_top)
        pat1 = debug_print(pat1, 'pat1', DEBUG)
        pat1.name = 'p@1_multi'
        misclassify_rate = MultiMisclassificationRate().apply(entity_y, T.ge(maxpredvec, 0.5))
        misclassify_rate = debug_print(misclassify_rate, 'error_rate', DEBUG)
        misclassify_rate.name = 'error_rate'
        return cost, pat1, misclassify_rate, maxpredvec
    
    def compute_cost_multi_instance_atten_meanalpha(self, ctx_rep, ctx_rep_len, M2, num_labels=102, DEBUG=False):
        logger.info('In: compute_cost_multi_instance_attent_meanalpha')
        y = T.matrix('targets', dtype='int32')
        label_emb_dim = ctx_rep_len / 2 if 'attent_emblabel' not in self._config else self._config['attent_emblabel']
        M = Linear(name='aggr_ctx_to_output', input_dim=label_emb_dim, output_dim=num_labels, use_bias=False) #lable embedding matrix
        initialize([M])
        A = Linear(name='bilinear_matrix_A', input_dim=ctx_rep_len, output_dim=label_emb_dim, use_bias=False) # bilinear matrix
        initialize([A])
        #when A is identity:         e = T.dot(ctx_rep, M.W).T
        e = T.dot(T.dot(ctx_rep, A.W), M.W).T
        alpha = T.nnet.softmax(e) #e is R^num_labels * #ctx
        alpha = debug_print(alpha, 'alpha', DEBUG)
        s = T.dot(T.mean(alpha, axis=0), ctx_rep) #alph is R^|#labels * #ctx| --> s is R^|len_of_ctx_rep|
        s = debug_print(s, 's', DEBUG)
        y_hat = Logistic().apply(M2.apply(s)) 
        
        entity_y = y[0][0:] #bag output
        cost = cross_entropy_loss(y_hat, entity_y)
        pred = T.argmax(y_hat)
        label_of_top = entity_y[pred]
        pat1 = T.mean(label_of_top)
        pat1.name = 'p@1_multi'
        misclassify_rate = MultiMisclassificationRate().apply(entity_y, T.ge(y_hat, 0.5))
        misclassify_rate.name = 'error_rate'
        return cost, pat1, misclassify_rate, y_hat


    def compute_cost_multi_instance_atten(self, ctx_rep, ctx_rep_len, M2, num_labels=102, DEBUG=False):
        logger.info('In: compute_cost_multi_instance_atten')
        y = T.matrix('targets', dtype='int32')
        label_emb_dim = ctx_rep_len / 2 if 'attent_emblabel' not in self._config else self._config['attent_emblabel']
        M = Linear(name='aggr_ctx_to_output', input_dim=label_emb_dim, output_dim=num_labels) #lable embedding matrix
        initialize([M])
        A = Linear(name='bilinear_matrix_A', input_dim=ctx_rep_len, output_dim=label_emb_dim, use_bias=False) # bilinear matrix
        initialize([A])
        #when A is identity:         e = T.dot(ctx_rep, M.W).T
        e = T.dot(T.dot(ctx_rep, A.W), M.W).T
        e = debug_print(e, 'e', DEBUG)
        alpha = T.nnet.softmax(e) #e is R^num_labels * #ctx
        alpha = debug_print(alpha, 'alpha', DEBUG)
        s = T.dot(alpha, ctx_rep) #alph is R^|#labels * #ctx| --> s is R^|#labels * len_of_ctx_rep|
        s = debug_print(s, 's', DEBUG)
        y_hat = Logistic().apply(T.nlinalg.diag(M2.apply(s))) 
        
        entity_y = y[0][0:] #bag output
        cost = cross_entropy_loss(y_hat, entity_y)
        pred = T.argmax(y_hat)
        label_of_top = entity_y[pred]
        pat1 = T.mean(label_of_top)
        pat1.name = 'p@1_multi'
        misclassify_rate = MultiMisclassificationRate().apply(entity_y, T.ge(y_hat, 0.5))
        misclassify_rate.name = 'error_rate'
        return cost, pat1, misclassify_rate, y_hat, alpha

    def compute_cost_multi_instance_mean(self, ctx_rep, ctx_rep_len, M, num_labels=102, DEBUG=False):
        logger.info('In: compute_cost_multi_instance_mean')
        y = T.matrix('targets', dtype='int32')
        
        s = T.mean(ctx_rep, axis=0) #alph is R^|ctx_rep.shape[0] * num_labels| --> s is R^|ctx_rep.shape[1]|
        s = debug_print(s, 's', DEBUG)
        ms = M.apply(s)
        ms = debug_print(ms, 'ms', DEBUG)
       
        y_hat_sigmoidOfAvg = Logistic().apply(ms) #sigmoid of average
#         y_hat_instances = Logistic().apply(M.apply(ctx_rep))
#         y_hat_avgOfSigmoid = Logistic().apply(M.apply(ctx_rep)).mean(axis=0)
        
        y_hat = y_hat_sigmoidOfAvg
        
        entity_y = debug_print(y[0][0:], 'entity_y', DEBUG) #bag output
        cost = cross_entropy_loss(y_hat, entity_y)
        
        cost = debug_print(cost, 'cost', DEBUG)
#         cost += 0.0001 * T.sqrt((M.W**2).sum())
        cost = debug_print(cost, 'cost', DEBUG)
        
        pred = T.argmax(y_hat)
        label_of_top = entity_y[pred]
        pat1 = T.mean(label_of_top)
        pat1 = debug_print(pat1, 'pat1', DEBUG)
        pat1.name = 'p@1_multi'
        misclassify_rate = MultiMisclassificationRate().apply(entity_y, T.ge(y_hat, 0.5))
        misclassify_rate = debug_print(misclassify_rate, 'error_rate', DEBUG)
        misclassify_rate.name = 'error_rate'
        return cost, pat1, misclassify_rate, y_hat, y_hat


    def compute_cost_multi_instance_mean2(self, y_hat, num_labels=102, DEBUG=False):
        logger.info('In: compute_cost_multi_instance_mean2')
        l2 = self._config['l2_regularization']
        y = T.matrix('targets', dtype='int32')
        meanpredvec = T.mean(y_hat, axis=0)
        entity_y = debug_print(y[0][0:], 'entity_y', DEBUG)
        cost = cross_entropy_loss(meanpredvec, entity_y)
        cost = debug_print(cost, 'cost', DEBUG)
        pred = T.argmax(meanpredvec)
        label_of_top = entity_y[pred]
        pat1 = T.mean(label_of_top)
        pat1 = debug_print(pat1, 'pat1', DEBUG)
        pat1.name = 'p@1_multi'
        misclassify_rate = MultiMisclassificationRate().apply(entity_y, T.ge(meanpredvec, 0.5))
        misclassify_rate = debug_print(misclassify_rate, 'error_rate', DEBUG)
        misclassify_rate.name = 'error_rate'
        return cost, pat1, misclassify_rate, meanpredvec, y_hat


    def compute_cost(self, y_hat, num_labels=102, DEBUG=False):
        logger.info('In: compute_cost')
        y = T.matrix('targets', dtype='int32')
        y_pred = T.argmax(y_hat, axis=1)
        label_of_predicted = debug_print(y[T.arange(y.shape[0]), y_pred], 'label_of_predicted', False)
        pat1 = T.mean(label_of_predicted)
        cost = cross_entropy_loss(y_hat, y)
        cost.name = 'cost'
        pat1.name = 'precision@1'
        misclassify_rate = MultiMisclassificationRate().apply(y, T.ge(y_hat, 0.5))
        misclassify_rate.name = 'error_rate'
        return cost, pat1, misclassify_rate
    
    
    def train(self, num_epochs=None, stop_after=None,  batch_sizes=None,
              initial_learning_rate=None, use_bokeh=False,
              checkpoint_path=None, init_lr=None, step_rule=None, DEBUG=False, shuffling=False, multi_type=None, devset='dev'):
        """ Train the model and report on the performance.
        :param use_bokeh:       Activate live-plotting to a running bokeh server. Make sure a server has been launched with the `bokeh-server` command!
        :param checkpoint_path:     Save (partially) trained model to this file. If the file already exists, training will resume from the checkpointed state.
        """
        l2 = self._config['l2_regularization']
        step_rule = step_rule or self._config['step_rule']
        init_lr = (init_lr or self._config['init_lr'])
        stop_after = stop_after or self._config['finish_if_no_improvement']
        valid_epochs = self._config['valid_epochs'] if 'valid_epochs' in self._config else 4
        checkpoint_path = checkpoint_path or self._config['checkpoint_path']
        net_path = self._config['net_path']
        features = self._config['features'] # contexts, mentions
        sources = features + ['entmentions', 'targets'] #always in... 
        logger.info('sources are %s', sources)
        self.conv_inp = False
        multi = multi_type is not None
        k = None
        for fea in features:
            if 'conv' in fea: self.conv_inp = True 
        if self.conv_inp:
            train_stream, num_train = self.get_datastreams_convinp(which_set='train', sources=sources, shuffling=shuffling, multi=multi, num_examples=k)#200000)
            dev_stream, num_dev = self.get_datastreams_convinp(which_set=devset, sources=sources, shuffling=False, multi=multi, num_examples=k)#50000)
        else:
            train_stream, num_train = self.get_datastreams(which_set='train', sources=sources, shuffling=shuffling, multi=multi)
            dev_stream, num_dev = self.get_datastreams(which_set=devset, sources=sources, shuffling=shuffling, multi=multi)
        logger.info('#train: %d, #dev: %d', num_train, num_dev)

        y_hat, ctx_rep, ctx_rep_len, hid_to_out_layer = self.build_network(102, features, DEBUG=DEBUG)
        
        if multi_type == 'max':
            cost, p_at_1, misclassify_rate, y_hat = self.compute_cost_multi_instance_max(y_hat)
        elif multi_type == 'attent':
            cost, p_at_1, misclassify_rate, y_hat,_ = self.compute_cost_multi_instance_atten(ctx_rep, ctx_rep_len, hid_to_out_layer)
        elif multi_type == 'attent_meanalpha':
            cost, p_at_1, misclassify_rate, y_hat = self.compute_cost_multi_instance_atten_meanalpha(ctx_rep, ctx_rep_len, hid_to_out_layer)
        elif multi_type == 'mean1':
            cost, p_at_1, misclassify_rate, y_hat,_ = self.compute_cost_multi_instance_mean(ctx_rep, ctx_rep_len, hid_to_out_layer)
            #cost, p_at_1, misclassify_rate, y_hat = self.compute_cost_multi_instance_mean(y_hat)
        elif multi_type == 'mean2':
            cost, p_at_1, misclassify_rate, y_hat,_ = self.compute_cost_multi_instance_mean2(y_hat)
        else:
            cost, p_at_1, misclassify_rate = self.compute_cost(y_hat)
            
        cg = ComputationGraph(cost)
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        cost += l2 * l2_norm(weights)
        cost.name = 'cost'
        logger.info('number of parameters in the model: %d', T.sum([p.size for p in cg.parameters]).eval())
        p_at_1.name = 'prec@1'
        if 'adagrad' in step_rule:
            cnf_step_rule = AdaGrad(init_lr)
        elif 'rms' in step_rule:
            cnf_step_rule = RMSProp(learning_rate=init_lr, decay_rate=0.90)
            cnf_step_rule = CompositeRule([cnf_step_rule, StepClipping(2.0)])
        
        logger.info('net path: %s', net_path)
        algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=cnf_step_rule, on_unused_sources='warn')
        gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
        step_norm = aggregation.mean(algorithm.total_step_norm)
        monitored_vars = [cost, p_at_1, misclassify_rate, gradient_norm, step_norm]
        train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_batch=True,
                                     before_first_epoch=False, prefix='tra')
        dev_monitor = DataStreamMonitoring(variables=[cost, misclassify_rate, p_at_1], after_batch=False, 
                before_first_epoch=True, data_stream=dev_stream, prefix="dev")
        extensions = [dev_monitor, train_monitor,
 #                       ProgressBar(), 
                        Timing(),
                        TrackTheBest(record_name='dev_cost'),
                        FinishIfNoImprovementAfter('dev_cost_best_so_far', epochs=stop_after),
                        FinishAfter(after_n_epochs=num_epochs), Printing(),
                        saveload.Load(net_path+'.toload.pkl'),
#                         saveload.Checkpoint(net_path+'.cp.pkl'),
                        ] + track_best('dev_cost', net_path+'.best.pkl')
        
        model = Model(cost)
        main_loop = MainLoop(model=model,data_stream=train_stream, algorithm=algorithm, extensions=extensions)
        main_loop.run()

        
    def aggregate_scores(self, outtype, topPercAgg=1.):
        devsampledfile = self._config['devcontexts_big'] 
        testsampledfile = self._config['testcontexts']
        dev_lines, e2freq_dev,_,_ = load_lines_info(devsampledfile, self.t2i)
        test_lines, e2freq_test,_,_ = load_lines_info(testsampledfile, self.t2i)
        self.dev_big_matrix = numpy.load(self._config['devscores']+'.'+outtype+'.npy') if self.dev_big_matrix == None else self.dev_big_matrix
        self.test_big_matrix = numpy.load(self._config['testscores']+'.'+outtype +'.npy') if self.test_big_matrix == None else self.test_big_matrix
        logger.info('Building the big matrix...')
        dev_type2ent_scores = build_type2entmatrix(self.dev_big_matrix, dev_lines, self.t2i)
        test_type2ent_scores = build_type2entmatrix(self.test_big_matrix, test_lines, self.t2i)
        logger.info('aggregating bag_scores for entities')
        dev_type2ent_scores_agg = big2small_avgs(dev_type2ent_scores, self.t2i, top_perc=topPercAgg)
        test_type2ent_scores_agg = big2small_avgs(test_type2ent_scores, self.t2i, top_perc=topPercAgg)
        write_small(self._config['matrixdev']+'.'+outtype, dev_type2ent_scores_agg, self.t2i, e2freq_dev)
        shutil.copyfile(self._config['matrixdev']+'.'+outtype, self._config['matrixdev'])
        write_small(self._config['matrixtest']+'.'+outtype, test_type2ent_scores_agg, self.t2i, e2freq_test)
        shutil.copyfile(self._config['matrixtest']+'.'+outtype, self._config['matrixtest'])
    
    def eval(self, configfile, outtype=None, topPercAgg=1.):
        self.aggregate_scores(outtype=outtype, topPercAgg=topPercAgg)
        cmd = 'python ' + dirname + '/matrix2measures_ents.py ' + configfile + ' > ' + configfile + '.meas.ents'
        outtype = outtype or 'multi'
        cmd += '.' + outtype
        p = Popen(cmd, shell=True); 
        p.wait() 
    
    def test(self, net_name=None, outtype="multi"):
        net_name = net_name or 'best'
        print '***', net_name 
        net_path = self._config['net_path'] + '.' + net_name + '.pkl'
        logger.info('applying the model on the test and dev data, model=%s', net_path)
        features = self._config['features'] # contexts, mentions
        self.conv_inp = False
        for fea in features:
            if 'conv' in fea: self.conv_inp = True
        sources = features + ['entmentions', 'targets'] #always in... 
        
        y_hat, ctx_rep, ctx_rep_len, hid_to_out_layer = self.build_network(102, features)
        cost, _, _ = self.compute_cost(y_hat)
        model = Model(cost)
        
        main_loop = load(net_path)
        old_model = main_loop.model
        logger.info(old_model.inputs)
        
        model.set_parameter_values(old_model.get_parameter_values())
        theinputs = [inp for inp in model.inputs if inp.name != 'targets']
        predict = theano.function(theinputs, y_hat)
        if self.conv_inp:
            test_stream, num_samples_test = self.get_datastreams_convinp(which_set='test', sources=sources, shuffling=False, multi=False, num_examples=None)
            dev_stream, num_samples_dev = self.get_datastreams_convinp(which_set='devbig', sources=sources, shuffling=False, multi=False, num_examples=None)
        else:
            test_stream, num_samples_test = self.get_datastreams(which_set='test', shuffling=False, sources=sources, multi=False)
            dev_stream, num_samples_dev = self.get_datastreams(which_set='devbig', shuffling=False, sources=sources, multi=False) #TODO: take care of dev big
        logger.info('#size dev: %d, size test: %d', num_samples_dev, num_samples_test)
        self.dev_big_matrix,_ = self.get_scores(dev_stream, num_samples_dev, predict, theinputs)
        numpy.save(self._config['devscores']+'.'+outtype, self.dev_big_matrix)
        self.test_big_matrix,_ = self.get_scores(test_stream, num_samples_test, predict, theinputs)
        numpy.save(self._config['testscores']+'.'+outtype, self.test_big_matrix)
        
    def test_eval_multi(self, configfile, net_name=None, outtype=None, multi_type=None):
        net_name = net_name or 'best'
        print '***', net_name 
        net_path = self._config['net_path'] + '.' + net_name + '.pkl'
        logger.info('applying the model on the test and dev data, model=%s', net_path)
        features = self._config['features'] # contexts, mentions
        self.conv_inp = False
        for fea in features:
            if 'conv' in fea: self.conv_inp = True
        sources = features + ['entmentions', 'targets'] #always in... 
        
        y_hat, ctx_rep, ctx_rep_len, hid_to_out_layer = self.build_network(102, features)
        
        if multi_type == 'max':
            cost, _, _, y_hat = self.compute_cost_multi_instance_max(y_hat)
        elif multi_type == 'attent':
            cost, _, _, y_hat,_ = self.compute_cost_multi_instance_atten(ctx_rep, ctx_rep_len, hid_to_out_layer)
        elif multi_type == 'attent_meanalpha':
	    cost, p_at_1, misclassify_rate, y_hat = self.compute_cost_multi_instance_atten_meanalpha(ctx_rep, ctx_rep_len, hid_to_out_layer)
        elif multi_type == 'mean1':
            cost, _, _, y_hat,_ = self.compute_cost_multi_instance_mean(ctx_rep, ctx_rep_len, hid_to_out_layer)
#             cost, _, _, y_hat, y_hat_instance = self.compute_cost_multi_instance_mean2(y_hat)
        elif multi_type == 'mean2':
            cost, _, _, y_hat,_ = self.compute_cost_multi_instance_mean2(y_hat)
        else:
            print "For using multi instance testing, you have to pass multitype in the arguments!!!"
            sys.exit(0)

        model = Model(cost)
        
        main_loop = load(net_path)
        old_model = main_loop.model
        logger.info(old_model.inputs)
        
        model.set_parameter_values(old_model.get_parameter_values())
        
        logger.info('Model loaded. Building prediction function...')
        theinputs = [inp for inp in model.inputs if inp.name != 'targets']
        predict = theano.function(theinputs, y_hat)
        k = None
        if self.conv_inp:
            dev_stream, num_samples_dev = self.get_datastreams_convinp(which_set='devbig', sources=sources, shuffling=False, multi=True, num_examples=k)#49999)
            test_stream, num_samples_test = self.get_datastreams_convinp(which_set='test', sources=sources, shuffling=False, multi=True, num_examples=k)#94211)
        else:
            test_stream, num_samples_test = self.get_datastreams(which_set='test', shuffling=False, sources=sources, multi=True)
            dev_stream, num_samples_dev = self.get_datastreams(which_set='devbig', shuffling=False, sources=sources, multi=True) #TODO: take care of dev big
        
        devsampledfile = self._config['devcontexts_big'] 
        testsampledfile = self._config['testcontexts']
        
        _, e2freq_dev, dev_ent_list,_ = load_lines_info(devsampledfile, self.t2i, top=k)
        _, e2freq_test, test_ent_list,_ = load_lines_info(testsampledfile, self.t2i, top=k)
        
        logger.info('#size dev: %d, size test: %d', len(dev_ent_list), len(test_ent_list))

        self.dev_agg_matrix,_ = self.get_scores_multi(dev_stream, len(dev_ent_list), predict, theinputs, dev_ent_list)
        numpy.save(self._config['devscores']+'.agg', self.dev_agg_matrix)
        self.test_agg_matrix,_ = self.get_scores_multi(test_stream, len(test_ent_list), predict, theinputs, test_ent_list)
        numpy.save(self._config['testscores']+'.agg', self.test_agg_matrix)
        
        
        write_small_multi(self._config['matrixdev']+'.'+outtype , self.dev_agg_matrix, e2freq_dev, dev_ent_list)
        shutil.copyfile(self._config['matrixdev']+'.'+outtype, self._config['matrixdev'])
        write_small_multi(self._config['matrixtest']+'.'+outtype, self.test_agg_matrix, e2freq_test, test_ent_list)
        shutil.copyfile(self._config['matrixtest']+'.'+outtype, self._config['matrixtest'])
        
        cmd = 'python ' + dirname + '/matrix2measures_ents.py ' + configfile + ' > ' + configfile + '.meas.ents'
        outtype = outtype or 'multi'
        cmd += '.' + outtype
        p = Popen(cmd, shell=True); 
        p.wait() 
    
    def print_attentions(self, configfile, net_name=None, outtype=None, multi_type=None, num_labels=102):
        net_name = net_name or 'best'
        print '***', net_name 
        net_path = self._config['net_path'] + '.' + net_name + '.pkl'
        logger.info('applying the model on the test and dev data, model=%s', net_path)
        features = self._config['features'] # contexts, mentions
        self.conv_inp = False
        for fea in features:
            if 'conv' in fea: self.conv_inp = True
        
        k = 500000
        i2t = {i: t for t,i in self.t2i.items()}
        testsampledfile = self._config['testcontexts']
        _, _, test_ent_list, test_lines = load_lines_info(testsampledfile, self.t2i, top=k)

        sources = features + ['entmentions', 'targets'] #always in... 
        
        y_hat_instances, ctx_rep, ctx_rep_len, hid_to_out_layer = self.build_network(102, features)
        
        if multi_type == 'max':
            cost, _, _, y_hat = self.compute_cost_multi_instance_max(y_hat_instances)
            alpha = y_hat_instances.T
        else:
            cost, _, _, y_hat, alpha = self.compute_cost_multi_instance_atten(ctx_rep, ctx_rep_len, hid_to_out_layer)
        
        model = Model(cost)
        
        main_loop = load(net_path)
        old_model = main_loop.model
        logger.info(old_model.inputs)
        
        model.set_parameter_values(old_model.get_parameter_values())
        
        logger.info('Model loaded. Building prediction function...')
        theinputs = [inp for inp in model.inputs if inp.name != 'targets']
        predict = theano.function(theinputs, [y_hat, alpha])
        if self.conv_inp:
            test_stream, num_samples_test = self.get_datastreams_convinp(which_set='test', sources=sources, shuffling=False, multi=True, num_examples=k)#94211)
        else:
            test_stream, num_samples_test = self.get_datastreams(which_set='test', shuffling=False, sources=sources, multi=True)
            dev_stream, num_samples_dev = self.get_datastreams(which_set='devbig', shuffling=False, sources=sources, multi=True) #TODO: take care of dev big
        
        
        
        logger.info('#size test: %d', len(test_ent_list))

        epoch_iter = test_stream.get_epoch_iterator(as_dict=True)
        logger.info('starting to iterate...')
        idx = -1
        num_ents = len(test_ent_list)
        ent2bagsize = test_ent_list.items()
        ctx_idx = 0
        
        while idx < num_ents:
            idx += 1
            ent, bsize = ent2bagsize[idx]
            src2vals  = epoch_iter.next()
            inp = [src2vals[src.name] for src in theinputs]
            bag_scores, alpha = predict(*inp)
            y_curr = src2vals['targets'][0]
            assert bsize == len(src2vals['targets'])
            types_ctx_list = test_lines[ctx_idx:ctx_idx+bsize]
            ctx_idx += bsize    
            mytypes = types_ctx_list[0][0]
            if bsize > 50 or len(mytypes) < 3:
                continue
            print "****", ent, bsize, mytypes

            maxtype_ix = argmax(bag_scores)
            print "predicted top type: ", i2t[maxtype_ix], bag_scores[maxtype_ix]
            for myt in mytypes:
                print bag_scores[self.t2i[myt]]
                ctx_scores = alpha[self.t2i[myt]]
                for score, types_ctx in zip(ctx_scores , types_ctx_list):
                    print myt, score, types_ctx[1] 
            

        
        
    def get_scores_multi(self, data_stream, num_samples, predict, theinputs, ent_list, num_labels=102):
        epoch_iter = data_stream.get_epoch_iterator(as_dict=True)
        goods = 0.
        idx = 0
        scores_matrix = numpy.zeros(shape=(num_samples, num_labels), dtype='float32')
        true_y_matrix = numpy.zeros(shape=(num_samples, num_labels), dtype='int8')
        ent2bagsize = ent_list.items()
        while idx < num_samples:
            src2vals  = epoch_iter.next()
            inp = [src2vals[src.name] for src in theinputs]
            scores = predict(*inp)
            y_curr = src2vals['targets'][0]
            ent, bsize = ent2bagsize[idx]
            assert bsize == len(src2vals['targets'])

            maxtype_ix = argmax(scores)
            if y_curr[maxtype_ix] == 1:
                goods += 1
            true_y_matrix[idx] = y_curr
            scores_matrix[idx] = scores
            idx += 1    
        logger.info('P@1 is = %f ', goods / num_samples)
        return scores_matrix, true_y_matrix
     
    def get_scores(self, data_stream, num_samples, predict, theinputs, num_labels=102):
        epoch_iter = data_stream.get_epoch_iterator(as_dict=True)
        goods = 0.
        idx = 0
        scores_matrix = numpy.zeros(shape=(num_samples, num_labels), dtype='float32')
        true_y_matrix = numpy.zeros(shape=(num_samples, num_labels), dtype='int8')
        while idx < num_samples:
            src2vals  = epoch_iter.next()
            inp = [src2vals[src.name] for src in theinputs]
            scores = predict(*inp)
            y_curr = src2vals['targets']
            for j in range(len(scores)):
                maxtype_ix = argmax(scores[j])
                if y_curr[j][maxtype_ix] == 1:
                    goods += 1
                true_y_matrix[idx] = y_curr[j]
                scores_matrix[idx] = scores[j]
                idx += 1    
        logger.info('P@1 is = %f ', goods / num_samples)
        return scores_matrix, true_y_matrix
    
    def test_evaluate(self, configfile):
        cmd = 'nice -n 2 python ' + dirname + '/train2level.py --config ' + configfile + \
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
    
    def apply(self, net_name=None, dsname='figer'):
        """
            Apply the model on a new test set and save the output.
        """
        net_name = net_name or 'best'
        print '***', net_name 
        net_path = self._config['net_path'] + '.' + net_name + '.pkl'
        logger.info('applying the model on the test, model=%s', net_path)
        features = self._config['features'] # contexts, mentions
        self.conv_inp = False
        for fea in features:
            if 'conv' in fea: self.conv_inp = True
        sources = features + ['entmentions', 'targets'] #always in... 
        main_loop = load(net_path)
        logger.info('Model loaded. Building prediction function...')
        model = main_loop.model
        logger.info(model.inputs)
        theinputs = [inp for inp in model.inputs if inp.name != 'targets']
        scores = [v for v in model.variables if v.name == 'y_hat'][0]
        predict = theano.function(theinputs, scores)
        self._config['dsdir'] += '/_' + dsname
        if self.conv_inp:
            test_stream, num_samples_test = self.get_datastreams_convinp(which_set='test', sources=sources, shuffling=False, multi=False)
            dev_stream, num_samples_dev = self.get_datastreams_convinp(which_set='dev', sources=sources, shuffling=False, multi=False)
        else:
            test_stream, num_samples_test = self.get_datastreams(which_set='test', shuffling=False, sources=sources, multi=False)
            dev_stream, num_samples_dev = self.get_datastreams(which_set='dev', shuffling=False, sources=sources, multi=False)
        logger.info('size test: %d', num_samples_test)
        self.test_big_matrix, self.true_y_matrix_test = self.get_scores(test_stream, num_samples_test, predict, theinputs)
        numpy.save(self._config['testscores']+ '_'+dsname, self.test_big_matrix)
        self.dev_big_matrix, self.true_y_matrix_dev = self.get_scores(dev_stream, num_samples_dev, predict, theinputs)
        
        label_theta_list = self.find_label_theresholds(self.dev_big_matrix, self.true_y_matrix_dev)
        self.compute_f_measure(self.test_big_matrix, self.true_y_matrix_test, theta=label_theta_list)
        
    
    

def get_argument_parser():
    """ Construct a parser for the command-line arguments. """
    """ Construct a parser for the command-line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", help="Path to configuration file")
    
    parser.add_argument(
        "--test", "-t", type=bool, help="Applying the model on the test data, or not")
    
    parser.add_argument(
        "--testeval", "-te", type=bool, help="Applying the model on the test data, or not")
    parser.add_argument(
        "--testmulti", "-tm", type=bool, help="Testing using MIML (or attention), and evaluating...")
    
    parser.add_argument(
        "--train", "-tr", type=bool, help="Training the model on the test data, or not")
    
    parser.add_argument(
        "--trainmulti", "-trm", type=bool, help="Training the model using multi instance learning")
    parser.add_argument(
        "--eval", "-ev", type=bool, help="Evaluating the model on classification task")
    parser.add_argument(
        "--printatt", "-pa", type=bool, help="Evaluating the model on classification task")
 
    parser.add_argument(
        "--apply", "-ap", type=bool, help="Applying the trained model on a test sample.")
 
    parser.add_argument(
        "--dsname", "-ds", help="Name of the evaluation dataset to be applied.e.g., figer")
 
    parser.add_argument(
        "--debug", "-de", type=bool, help="Evaluating the model on classification task")
 
    parser.add_argument(
        "--outtype", "-outtype", help="type of the model training")
    
    parser.add_argument(
        "--net", "-net", help="path to the trained model")
    
    parser.add_argument(
        "--multitype", "-mult", help="Type of multi instance: mean, max, attent")
    
    parser.add_argument(
        "--percagg", "-per", type=float, help="the percentage of the top scores to be aggregated to get the summary score for the bag. ")
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
        "--live-plotting",  "-lp", action="store_true",
        help="Enable live-plotting to a running Bokeh server (start with "
             "`bokeh-server`)")
    return parser

import shutil

if __name__ == '__main__':
#     theano.config.dnn.enabled = False 
    parser = get_argument_parser()
    args = parser.parse_args()
    if not args.config and (not args.samples or not args.embeddings):
        print("Please provide either a config file (--config) or the paths "
              "to the samples and embeddings files. "
              "(--samples, --embeddings).")
        sys.exit(1)
    trainer = EntityTypingTrainer.from_config(args.config)
    trainer._config['hidden_units'] = args.hidden_units if args.hidden_units else trainer._config['hidden_units']
    if args.max_len:
        trainer._config['max_len'] = args.max_len
    if args.apply:
        trainer.apply(net_name=args.net, dsname=args.dsname)
        sys.exit()
    if args.train:
#         trainer.train(use_bokeh=args.live_plotting,
#                 checkpoint_path=args.checkpoint, DEBUG=False, shuffling=False, multi_type=args.multitype, 
#                 num_epochs=10, stop_after=2, init_lr=0.005, step_rule='adagrad', devset='devbig') 
#         sys.exit()
#         
        trainer.train(num_epochs=args.max_epochs, use_bokeh=args.live_plotting,
                    checkpoint_path=args.checkpoint, DEBUG=False, shuffling=True)
        logger.info('Copying net for re-retraining...')
        shutil.copyfile(trainer._config['net_path']+'.best.pkl', trainer._config['net_path']+'.toload.pkl')
        trainer._config['batch_sizes']['train'] = 100
        logger.info('Re-retraining for another 10 epochs with low lr and adagrad...')
        trainer.train(num_epochs=10, stop_after=2, init_lr=0.005, step_rule='adagrad', shuffling=True)
        logger.info('Copying net..')
        shutil.copyfile(trainer._config['net_path']+'.best.pkl', trainer._config['net_path']+'.contextmodel.pkl')
        shutil.copyfile(trainer._config['net_path']+'.best.pkl', trainer._config['net_path']+'.toload.pkl')
        trainer.test_evaluate(args.config)
           
        logger.info('The context-model training finished... Now, fine-tuning using multi-instance learning...')
        trainer.train(use_bokeh=args.live_plotting,
                checkpoint_path=args.checkpoint, DEBUG=False, shuffling=False, multi_type=args.multitype, 
                num_epochs=10, stop_after=2, init_lr=0.008, step_rule='adagrad', devset='devbig') 
    if args.trainmulti:
        trainer.train(use_bokeh=args.live_plotting,
                checkpoint_path=args.checkpoint, DEBUG=False, shuffling=False, multi_type=args.multitype, 
                num_epochs=10, stop_after=2, init_lr=0.008, step_rule='adagrad', devset='devbig')
        
    if args.test:
        trainer.test(net_name=args.net, outtype=args.outtype)
    if args.testeval:    
        trainer.test_evaluate(args.config)
    if args.testmulti:
        trainer.test_eval_multi(args.config, net_name=args.net, outtype=args.outtype, multi_type=args.multitype)
    if args.eval:
        trainer.eval(args.config, args.outtype, topPercAgg=args.percagg)
    if args.printatt:
        trainer.print_attentions(args.config, net_name=args.net, multi_type=args.multitype)
