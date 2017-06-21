'''
Created on Feb 9, 2016

@author: yadollah
'''
from _collections import defaultdict
import argparse
import cPickle
from collections import Iterable
import logging
import os
import shutil
import string
from subprocess import Popen
import subprocess
import sys

from numpy import argmax
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
    GenerateNegPosTransformer
from blocks.roles import add_role, WEIGHT
from myutils import debug_print, fillt2i, \
    build_type2entmatrix, big2small, write_small, load_lines_info, MyPool, \
    computeFscore, calcPRF, softmax, normalize
import theano.tensor as T
import yaml
import time
from layers import HiddenLayer
from myExtensions import F1MultiClassesExtension,\
    GetPRcurve, WriteBest
from train import JointUnaryBinary,\
    get_argument_parser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('train.py')

class JointUnaryBinaryOldEntityTyping(JointUnaryBinary):
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
        trainer = JointUnaryBinaryOldEntityTyping(config['dsdir'], config['embeddings_path'])
        trainer._config.update(config)
        trainer.t2i,_ = fillt2i(trainer._config['typefile'])
        trainer.dev_big_matrix = None
        trainer.test_big_matrix = None
        trainer.curSeed = 23455
#         trainer.set2bag_len_list = {}
#         trainer.set2bag_len_list['train'] = trainer.get_bag_sizes('train', 'entmentions')
#         trainer.set2bag_len_list['dev'] = trainer.get_bag_sizes('dev', 'entmentions')
#         trainer.set2bag_len_list['devbig'] = trainer.get_bag_sizes('devbig', 'entmentions')
#         trainer.set2bag_len_list['test'] = trainer.get_bag_sizes('test', 'entmentions')
#         print len(trainer.set2bag_len_list['devbig']), len(trainer.set2bag_len_list['test'])
        return trainer

    def __init__(self, samples_path, embeddings_path):
        """ Initialize the trainer.

        :param samples_path:    Path to HDF file with the samples
        :param embeddings_path: Path to HDF file with the embedding vectors
        """
        super(JointUnaryBinaryOldEntityTyping, self).__init__(samples_path, embeddings_path)

    def apply_cnn(self, l_emb1, l_size1, l_emb2, l_size2, r_emb1, r_size1, r_emb2, r_size2, embedding_size, mycnf):
        assert l_size1 == r_size1
        assert l_size2 == r_size2
        assert l_size1 == l_size1
        max_len = l_size1
        fv_len = 0
        filter_sizes = mycnf['cnn_config']['filter_sizes']
        num_filters = mycnf['cnn_config']['num_filters']
        for i, fw in enumerate(filter_sizes):
            conv_left = ConvolutionalActivation(
                            activation=Rectifier().apply,
                            filter_size=(fw, embedding_size), 
                            num_filters=num_filters,
                            num_channels=1,
                            image_size=(max_len, embedding_size),
                            name="conv"+str(fw)+l_emb1.name, seed=self.curSeed)
            conv_right = ConvolutionalActivation(
                            activation=Rectifier().apply,
                            filter_size=(fw, embedding_size), 
                            num_filters=num_filters,
                            num_channels=1,
                            image_size=(max_len, embedding_size),
                            name="conv"+str(fw)+r_emb1.name, seed=self.curSeed)
            pooling = MaxPooling((max_len-fw+1, 1), name="pool"+str(fw))
            initialize([conv_left, conv_right])
            l_convinp1 = l_emb1.flatten().reshape((l_emb1.shape[0], 1, max_len, embedding_size))
            l_convinp2 = l_emb2.flatten().reshape((l_emb2.shape[0], 1, max_len, embedding_size))
            l_pool1 = pooling.apply(conv_left.apply(l_convinp1)).flatten(2)
            l_pool2 = pooling.apply(conv_left.apply(l_convinp2)).flatten(2)
            r_convinp1 = r_emb1.flatten().reshape((r_emb1.shape[0], 1, max_len, embedding_size))
            r_convinp2 = r_emb2.flatten().reshape((r_emb2.shape[0], 1, max_len, embedding_size))
            r_pool1 = pooling.apply(conv_right.apply(r_convinp1)).flatten(2)
            r_pool2 = pooling.apply(conv_right.apply(r_convinp2)).flatten(2)
            onepools1 = T.concatenate([l_pool1, r_pool1], axis=1)
            onepools2 = T.concatenate([l_pool2, r_pool2], axis=1)
            fv_len += conv_left.num_filters * 2
            if i == 0:
                outpools1 = onepools1
                outpools2 = onepools2
            else:
                outpools1 = T.concatenate([outpools1, onepools1], axis=1)
                outpools2 = T.concatenate([outpools2, onepools2], axis=1)
        return outpools1, outpools2, fv_len
    
    def build_feature_vector_noMention(self, features, DEBUG=False):
        embeddings, _ = self.get_embeddings()
        embedding_size = embeddings.shape[1]
        print embeddings.shape
        lookup = StaticLookupTable(embeddings.shape[0], embeddings.shape[1])
        lookup.allocate()
        lookup.W.set_value(embeddings)
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
        fv1, fv2, fvlen = self.apply_cnn(l_emb1, l_size1, l_emb2, l_size2, r_emb1, r_size1, r_emb2, r_size2, embedding_size, mycnf)
        logger.info('feature size for each input token: %d', embedding_size)
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
        logger.info('building the network, with one CNN for left and one for right')
        hidden_units = hidden_units or self._config['hidden_units']
        logger.info('#hidden units: %d', hidden_units)
        # building the feature vector from input.  
        mlp_in_e1, mlp_in_e2, mlp_in_dim = self.build_feature_vector_noMention(features)
        logger.info('feature vector size: %d', mlp_in_dim)
        
        mlp = MLP(activations=[Rectifier()],
            dims=[mlp_in_dim, hidden_units], seed=self.curSeed
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
    
if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    logger.info('args: %s', args)
    if not args.config and (not args.samples or not args.embeddings):
        print("Please provide either a config file (--config) or the paths "
              "to the samples and embeddings files. "
              "(--samples, --embeddings).")
        sys.exit(1)
    trainer = JointUnaryBinaryOldEntityTyping.from_config(args.config)
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
        
    else:
        from logistic_sgd import LogisticRegression
        
    if args.train:
        trainer.train_joint(num_epochs=args.max_epochs, use_bokeh=args.live_plotting,
                    checkpoint_path=args.checkpoint, DEBUG=False, shuffling=True, multi=False, devset='dev', init_lr=0.001, step_rule='rms', l2weight=0.0000001)
        logger.info('Copying net for re-retraining...')
        shutil.copyfile(trainer._config['net_path']+'.best.pkl', trainer._config['net_path']+'.toload.pkl')
        trainer.train_joint(num_epochs=args.max_epochs, use_bokeh=args.live_plotting,
                    checkpoint_path=args.checkpoint, DEBUG=False, shuffling=True, multi=True, devset='dev', init_lr=0.002, step_rule='adagrad', l2weight=0.00001)
        trainer.test_rel(multi=True, outPRfile=args.config+'.PRrel')
        trainer.eval_rel(args.config+'.PRrel')
        sys.exit(0)
        
    if args.train_rel:
        trainer.train_rel_heike(num_epochs=args.max_epochs, use_bokeh=args.live_plotting,
                    checkpoint_path=args.checkpoint, DEBUG=False, shuffling=True, multi=False, devset='dev', init_lr=0.001, step_rule='rms', l2weight=0.00000001)
        logger.info('Copying net for re-retraining...')
        shutil.copyfile(trainer._config['net_path']+'.best.pkl', trainer._config['net_path']+'.toload.pkl')
        trainer.train_rel_heike(num_epochs=args.max_epochs, use_bokeh=args.live_plotting,
                    checkpoint_path=args.checkpoint, DEBUG=False, shuffling=True, multi=True, devset='dev', init_lr=0.001, step_rule='adagrad', l2weight=0.00001)
        trainer.test_rel_heike(multi=True, outPRfile=args.config+'.PRrel')
        trainer.eval_rel(args.config+'.PRrel')
        sys.exit(0)
       
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
        trainer.eval_rel(args.config+'.PRrel')
        
