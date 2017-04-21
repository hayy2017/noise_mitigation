'''
Created on Feb 9, 2016

@author: yadollah
'''
# from __future__ import print_function

import itertools as it
import os
import re
import sys
from collections import defaultdict, namedtuple
import h5py
import numpy as np
import logging
import argparse
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('makefueldataset')
import numpy
import yaml
from myclasses import Mention
from fuel.datasets import H5PYDataset
import myutils as cmn

ctx_entity_pat = re.compile(r'(/m/.+)/(.*)##(.+)')


Context = namedtuple(
    "Context", ["idx",           # Index of the context
                "entity_id",     # ID of the context's target entity
                "entity_str",    # String representation of the target entity
                "entity_idx",    # Position of the target entity in the context
                "prominent_type",  # The prominent type of target entity
                "all_types",     # All types of the target entity (including
                                 # the prominent type!)
                "context",        # The actual context, usually a list of
                                 # embedding indices
                "mention"
                ])

EntityType = namedtuple(
    "EntityType", ["idx",          # Index of the entity type
                   "name",         # Name of the entity type
                   "num_entities",  # How many entities have this type?
                   "frequency",    # How often does this type appear
                   "parents"       # Parent entity types
                   ])

def read_embeddings(fname, num=None):
    """ Read word embeddings from file

    :param embedding_file:  Path to embedding file
    :type embedding_file:   str/unicode
    :param num:             Restrict number of embeddings to load
    :type num:              int
    :returns:               Mapping from words to their embedding
    :rtype:                 str -> numpy.array dict
    """
    # NOTE: This should be pretty efficient, since we're only reading the file
    #       line by line and are reading directly into memory-efficient numpy
    #       arrays.
    logger.info("loading word2vec vectors from: %s", fname)
    with open(fname) as fp:
        num_vecs, vec_size = (int(x) for x in fp.readline().strip().split())
        num_vecs += 2  # For <UNK>
        embeddings = numpy.zeros((num_vecs, vec_size), dtype='float32')
        embeddings[0,:] = 0.001
        word_to_idx = {'<UNK>': 0, '<PAD>': 1}
        for idx, line in enumerate(fp, start=2):
            if num is not None and idx > num:
                break
            parts = line.strip().split()
            embeddings[idx,:] = [float(v) for v in parts[1:]]
            word_to_idx[parts[0]] = idx
    logger.info("number of words in word2vec: %d", len(word_to_idx))
    return embeddings, word_to_idx

def load_types(types_path, parents_path=None):
    """ Load entity type information from types and type-parents files.

    :param types_path:      Path to types information
    :type types_path:       str
    :param parents_path:    Path to type parents information
    :type parents_path:     str
    :rtype:                 generator that yields :py:class:`EntityType
                            objects
    """
    parents = defaultdict(list)
    if parents_path:
        with open(parents_path) as fp:
            for line in fp:
                parts = line.split()
                # Ignore empty lines
                if parts:
                    parents[parts[0]] = parts[1:]
    with open(types_path) as fp:
        for idx, line in enumerate(fp):
            ent_type, num_targets, freq = line.split("\t")
            yield EntityType(idx, ent_type, int(num_targets), int(freq),
                             parents[ent_type])

def context_to_embedding_idxs(context, word_to_idx, target_pos):
    """ Convert a context of string tokens to a context of their respective
        embedding indices.

    :param context:     The context
    :type context:      list of str/unicode
    :param word_to_idx: Mapping from words to their embedding index
    :type word_to_idx:  str/unicode -> int dict
    :param target_pos:  position of the target entity
    :type target_pos:   int
    :returns:           The same context with embedding indices instead of the
                        words
    :rtype:             list of int
    """
    indices = []
    
    for idx, token in enumerate(context):
        ent_match = ctx_entity_pat.match(token)
        if idx == target_pos:
            # The target entity uses the embedding of its entity ID
            token = ent_match.group(1)
        elif ent_match:
            # Replace other entities with their category
            token = ent_match.group(3)
        if token in word_to_idx:
            indices.append(word_to_idx[token])
        else:
            indices.append(word_to_idx['<UNK>'])
    return indices


def mention_to_embedding_idxs(entmention, word_to_idx, max_len=4):
    indices = []
    for i, token in enumerate(entmention):
        if i >= max_len: 
            return indices
        if token in word_to_idx:
            indices.append(word_to_idx[token])
        else:
            indices.append(word_to_idx['<UNK>'])
    for j in range(len(indices), max_len):
            indices.append(word_to_idx['<PAD>'])
    assert len(indices) == max_len
    return numpy.asarray(indices)


def load_contexts(path, word_to_idx, voc, upto=-1):
    """ Load all contexts from the given file.

    :param path:        Path to the context
    :type path:         str/unicode
    :param word_to_idx: Mapping from words to their embedding index
    :type word_to_idx:  str/unicode -> int dict
    :rtype:             generator that yields :py:class:`Context` objects
    """
    logger.info("loading all contexts from %s", path)
    contexts = []
    with open(path) as fp:
        for c, line in enumerate(fp):
            line = line.strip()
            if upto != -1 and c > upto:
                break
            if len(line.split('\t')) > 6:
                m = Mention.parse_line_new(line)
            else:
                m = Mention.parse_line_old(line)
            if m == None:
                continue
            entity_idx = next(idx for idx, tok in enumerate(m.mycontext)
                              if tok.startswith(m.fb_mid))
            context = context_to_embedding_idxs(m.mycontext, word_to_idx, entity_idx)
            mention_idexes = mention_to_embedding_idxs(m.mentionName.split('_'), word_to_idx)
            for tt in m.mycontext + m.mentionName.split('_'):
                voc.add(tt)
            contexts.append(Context(c, m.fb_mid, m.mentionName, entity_idx,
                          m.types[0], m.types, context, mention_idexes))
    return contexts
def build_targets_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big):
    logger.info("building targets dataset")
    entity_types = list(load_types(config['typefile']))
    (t2idx, _) = cmn.loadtypes(config['typefile'])
    totals = len(all_contexts)
    targets_m = numpy.zeros(shape=(totals, len(t2idx)), dtype='int32') 
    for i, ctx in enumerate(all_contexts):
        types_idx = [t2idx[t] for t in ctx.all_types if t in t2idx] 
        targets_m[i] = cmn.convertTargetsToBinVec(types_idx, len(t2idx))
    dsdir = config['dsdir']
    fp = h5py.File(dsdir + '_targets.hdf', mode='w')
    targets = fp.create_dataset('targets', targets_m.shape, dtype='int32')
    targets.attrs['type_to_ix'] = yaml.dump(t2idx)
    targets[...] = targets_m
    targets.dims[0].label = 'all_types'
    split_dict = {
        'train': {'targets': (0, nsamples_train)},
        'dev': {'targets': (nsamples_train, nsamples_train + nsamples_dev)},
        'test': {'targets': (nsamples_train + nsamples_dev, nsamples_train + nsamples_dev + nsamples_test)},    
        'devbig': {'targets': (nsamples_train + nsamples_dev + nsamples_test, totals)}
    }    
    fp.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    fp.flush()
    fp.close()
    
#     context_prominent_types = fp.create_dataset(
#           'prominent_types', compression='gzip',
#            data=np.asarray([all_typenames.index(ctx.prominent_type)
#            for ctx in all_contexts], dtype=np.uint16))



def build_contexts_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big):
    logger.info('building contexts dataset')
    totals = len(all_contexts)
    ctx_dtype = h5py.special_dtype(vlen=np.dtype('int32'))
    logger.info("#contexts: %d", totals)
    with h5py.File(config['dsdir'] + "_contexts.hdf", mode='w') as fp:
        contexts = fp.create_dataset(
                'contexts', compression='gzip',
                data=[np.asarray(ctx.context) for ctx in all_contexts],
                shape=(totals,), dtype=ctx_dtype)
        split_dict = {
          'train': {'contexts': (0, nsamples_train)},
          'dev': {'contexts': (nsamples_train, nsamples_train + nsamples_dev)},
          'test': {'contexts': (nsamples_train + nsamples_dev, nsamples_train + nsamples_dev + nsamples_test)},    
          'devbig': {'contexts': (nsamples_train + nsamples_dev + nsamples_test, totals)}
        }    
        fp.attrs['split'] = H5PYDataset.create_split_array(split_dict)


def build_entmentions_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big):
    logger.info('building entmentions dataset')
    totals = len(all_contexts)
    ctx_dtype = h5py.special_dtype(vlen=np.dtype('uint32'))
    dsdir = config['dsdir']
    ctx_entity_dtype = np.dtype([
            ("id", np.dtype(str), 64),
            ("token", np.dtype(str), 64),
            ("position", np.dtype('uint8'))])
    with h5py.File(dsdir + "_entmentions.hdf", mode='w') as fp:
        context_entities = fp.create_dataset(
            'entmentions', compression='gzip',
            data=np.asarray([(ctx.entity_id,
                              ctx.entity_str,
                              ctx.entity_idx) for ctx in all_contexts],
                            dtype=ctx_entity_dtype))
        split_dict = {
          'train': {'entmentions': (0, nsamples_train)},
          'dev': {'entmentions': (nsamples_train, nsamples_train + nsamples_dev)},
          'test': {'entmentions': (nsamples_train + nsamples_dev, nsamples_train + nsamples_dev + nsamples_test)},    
          'devbig': {'entmentions': (nsamples_train + nsamples_dev + nsamples_test, totals)}
        }    
        fp.attrs['split'] = H5PYDataset.create_split_array(split_dict)

def build_mentions_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big, max_len_men=4):
    logger.info('building mentions (indices of mention words) dataset')
    totals = len(all_contexts)
    dsdir = config['dsdir']
    mentions_m = numpy.ones(shape=(totals, max_len_men), dtype='int32') 
    for i, ctx in enumerate(all_contexts):
        mentions_m[i] = ctx.mention
    with h5py.File(dsdir + "_mentions.hdf", mode='w') as fp:
        mentions = fp.create_dataset('mentions', mentions_m.shape, dtype='int32')
        mentions[...] = mentions_m
        split_dict = {
          'train': {'mentions': (0, nsamples_train)},
          'dev': {'mentions': (nsamples_train, nsamples_train + nsamples_dev)},
          'test': {'mentions': (nsamples_train + nsamples_dev, nsamples_train + nsamples_dev + nsamples_test)},    
          'devbig': {'mentions': (nsamples_train + nsamples_dev + nsamples_test, totals)}
        }
        fp.attrs['split'] = H5PYDataset.create_split_array(split_dict)


def get_bag_len_list(contexts):
    baglenlist = []
    ents = set()
    old = contexts[0]
    ents.add(old)
    n = 0
    contexts.append('ALAKI')
    for ctx in contexts[1:]:
        n += 1
        newe = ctx
        ents.add(old)
        if newe != old:
            baglenlist.append(n)
            n = 0
        old = newe
    print len(baglenlist), len(ents)
    if len(baglenlist) == len(ents) - 1: #the last one is missed 
        baglenlist.append(n)
        print n, baglenlist[-1]
    assert len(baglenlist) == len(ents)
    return baglenlist

def save_bag_sizes(config):
    def simple_load_contexts(path, upto=-1):
        midlist = []
        with open(path) as fp:
            for c, line in enumerate(fp):
                line = line.strip()
                if upto != -1 and c > upto:
                    break
                if len(line.split('\t')) > 6:
                    m = Mention.parse_line_new(line)
                else:
                    m = Mention.parse_line_old(line)
                midlist.append(m.fb_mid)
        return midlist
    
    devbig = get_bag_len_list(simple_load_contexts(config['devcontexts_big']))
    train = get_bag_len_list(simple_load_contexts(config['traincontexts']))
    dev = get_bag_len_list(simple_load_contexts(config['devcontexts']))
    test = get_bag_len_list(simple_load_contexts(config['testcontexts']))
    with h5py.File(config['dsdir'] + "_bag_lenghts.hdf", mode='w') as fp:
        _ = fp.create_dataset(
                'train', shape=(len(train),), 
                dtype='int32', data=train)
        _ = fp.create_dataset(
                'dev', shape=(len(dev),), 
                dtype='int32', data=dev)
        _ = fp.create_dataset(
                'test', shape=(len(test),),
                dtype='int32', data=test)
        _ = fp.create_dataset(
                'devbig', shape=(len(devbig),),
                dtype='int32', data=devbig)
    
        

def main1(cnfpath):
    config = cmn.loadConfig(cnfpath)
    upto = -1
    embeddings, word_to_idx = read_embeddings(config['embeddings_path'], num=None)
    voc = set()
    contexts_test = list(load_contexts(config['testcontexts'], word_to_idx, voc, upto=upto))
    contexts_train = list(load_contexts(config['traincontexts'], word_to_idx, voc, upto=upto))
    contexts_dev = list(load_contexts(config['devcontexts'], word_to_idx, voc, upto=upto))
    contexts_dev_big = list(load_contexts(config['devcontexts_big'], word_to_idx, voc, upto=upto))
    all_contexts = list(it.chain(contexts_train, contexts_dev, contexts_test, contexts_dev_big))
    nsamples_train = len(contexts_train); nsamples_dev = len(contexts_dev); nsamples_dev_big = len(contexts_dev_big); nsamples_test = len(contexts_test);
    print '*** vocab size: ', len(voc)
    logger.info('#dev: %d, #dev_big: %d, #train: %d, #test: %d', nsamples_dev, nsamples_dev_big, nsamples_train, nsamples_test)
    if True:
        build_targets_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big)
        build_contexts_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big)
        build_entmentions_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big)
        build_mentions_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big)
        save_bag_sizes(config)
        logger.info("Writing embedding HDF file")
        with h5py.File(config['dsdir'] + "_embeddings.hdf", mode='w') as fp:
            vectors = fp.create_dataset('vectors', compression='gzip',
                                        data=embeddings)
            word_list = [w for w, idx in sorted(word_to_idx.items(),
                                                key=lambda x: x[1])]
            words = fp.create_dataset(
                'words', shape=(len(word_list),), compression='gzip',
                dtype=h5py.special_dtype(vlen=unicode), data=word_list)
    else:
        build_contexts_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big)

def main2(cnfpath):
    config = cmn.loadConfig(cnfpath)
    save_bag_sizes(config)

def get_argument_parser():
    """ Construct a parser for the command-line arguments. """
    """ Construct a parser for the command-line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", help="Path to configuration file")
    
    parser.add_argument(
        "--type", "-t", type=int, help="0: full building, 1: only bagsizes, 2: only test file")
    
    parser.add_argument(
        "--test-file", "-tst", help="path to the sampled test file")
    parser.add_argument(
        "--dsname", "-ds", help="Name of the evaluation dataset to be applied.e.g., figer")
    
    return parser


def build_only_test(cnfpath, test_file, ds_name):
    config = cmn.loadConfig(cnfpath)
    test_file = test_file or config['testcontexts']
    logger.info('building test-dataset from file: %s', test_file)
    upto = -1
    _, word_to_idx = read_embeddings(config['embeddings_path'], num=None)
    voc = set()
    contexts_test = list(load_contexts(test_file, word_to_idx, voc, upto=upto))
    contexts_dev = list(load_contexts(config['devcontexts'], word_to_idx, voc, upto=upto))
    all_contexts = list(it.chain(contexts_dev, contexts_test))
    nsamples_train = 0; nsamples_dev_big = 0; 
    nsamples_test = len(contexts_test); nsamples_dev = len(contexts_dev)
    config['dsdir'] = config['dsdir'] + '/_' + ds_name
    build_targets_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big)
    build_contexts_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big)
    build_entmentions_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big)
    build_mentions_ds(config, all_contexts, nsamples_train, nsamples_dev, nsamples_test, nsamples_dev_big)
#     save_bag_sizes(config)
        
if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    if args.type == 0:
        main1(args.config)
    elif args.type == 1:
        main2(args.config)
    elif args.type == 2: ##build for apply feature
        build_only_test(args.config, args.test_file, args.dsname)
    else:
        print 'bad format for args.type: ', args.type
    
    
