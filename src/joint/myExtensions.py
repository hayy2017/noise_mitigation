#!/usr/bin/python

from blocks.extensions import SimpleExtension
import time
import theano
import numpy
import tempfile
import six
import os
from contextlib import closing
import tarfile
from blocks.filter import get_brick
from collections import Iterable
from myExtensions_subroutines import *
from blocks.serialization import secure_dump
from pickle import HIGHEST_PROTOCOL
try:
    from pickle import DEFAULT_PROTOCOL
    from pickle import _Pickler
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL
    from pickle import Pickler as _Pickler
LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"

class WriteBest(SimpleExtension):
    def __init__(self, record_name, **kwargs):
        self.record_name = record_name
        self.best_name = "best_" + record_name
        kwargs.setdefault("after_epoch", True)
        super(WriteBest, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        best_value = self.main_loop.status.get(self.best_name, None)
        print best_value
        
def getF1(tp, numHypo, numRef, file_pointer=None):
    precision = 1
    recall = 0
    f1 = 0
    if numHypo > 0:
      precision = 1.0 * tp / numHypo
    if numRef > 0:
      recall = 1.0 * tp / numRef
    if precision + recall > 0:
      f1 = 2 * precision * recall / (precision + recall)
    outstr = str(time.ctime()) + "\tP = " + str(precision) + ", R = " + str(recall) + ", F1 = " + str(f1)
    if file_pointer:
        file_pointer.write(outstr + '\n')
    else:
        print outstr
    return f1

class DebugExtension(SimpleExtension):
  def __init__(self, layer, model, data_stream, num_samples, batch_size, **kwargs):
    super(DebugExtension, self).__init__(**kwargs)
    y_hat1 = layer.weightedInput
    y_hat0 = layer.attentionWeightsOwn
    y_hat0b = layer.a
    y_hat0c = layer.kmaxSequence
    y_hat0d = layer.outputTmpCorrect
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y'] # name of input variables in graph
    self.predict = theano.function(self.theinputs, [y_hat1, y_hat0, y_hat0b, y_hat0c, y_hat0d])
    self.data_stream = data_stream
    self.num_samples = num_samples
    self.batch_size = batch_size

  def do(self, which_callback, *args):
    epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)
    for i in range(0,1):
      src2vals = epoch_iter.next()
      inp = [src2vals[src.name] for src in self.theinputs]
      results = self.predict(*inp)
      for r in results:
        print r.shape
        print r
      print "---"
      exit()
      
class ModelResultsMI(SimpleExtension):
  def __init__(self, layer3, y, model, data_stream, num_samples, batch_size, **kwargs):
    super(ModelResultsMI, self).__init__(**kwargs)
    y_hat = layer3.results(y)
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y' and inp.name != 'y_types1' and inp.name != 'y_types2'] # name of input variables in graph
    self.predict = theano.function(self.theinputs, y_hat)
    self.data_stream = data_stream
    self.num_samples = num_samples
    self.batch_size = batch_size

  def do(self, which_callback, *args):
    if isinstance(self.batch_size, Iterable):
      num_batches = self.num_samples
      print "multi instance evaluation"
    else:
      num_batches = self.num_samples / self.batch_size
    epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)
    print "ref\thypo\tconfidence for 1"
    for i in range(num_batches):
      src2vals = epoch_iter.next()
      inp = [src2vals[src.name] for src in self.theinputs]
      result = self.predict(*inp)
      y_curr = src2vals['y']

      if isinstance(self.batch_size, Iterable): # we have a bag
        max_pos = -1
        hypo = 0
        maxTotal = -1
        for instance in range(self.batch_size[i]):
          curHypo = result[0][instance]
          curProb = result[2][instance][curHypo]
          if curHypo > 0: # positive instance
            if curProb > max_pos:
              max_pos = curProb
              hypo = curHypo
          if curProb > maxTotal:
            maxTotal = curProb
        if max_pos > -1:
          conf = max_pos
        else:
          conf = maxTotal
        ref = y_curr[0][0]
        print str(ref) + "\t" + str(hypo) + "\t" + str(conf)
      else:
        for j in range(self.batch_size):
          hypo = result[0][j]
          ref = y_curr[j][0]
          conf = result[2][j][hypo]
          print str(ref) + "\t" + str(hypo) + "\t" + str(conf)
          

class F1Extension(SimpleExtension):
  def __init__(self, layer3, y, model, data_stream, num_samples, batch_size, **kwargs):
    super(F1Extension, self).__init__(**kwargs)
    y_hat = layer3.results(y)
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y'] # name of input variables in graph
    self.predict = theano.function(self.theinputs, y_hat)
    self.data_stream = data_stream
    self.num_samples = num_samples
    self.batch_size = batch_size

  def do(self, which_callback, *args):
    num_batches = self.num_samples / self.batch_size
    epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)
    tp = 0
    numHypo = 0
    numRef = 0
    for i in range(num_batches):
      src2vals = epoch_iter.next()
      inp = [src2vals[src.name] for src in self.theinputs]
      probs = self.predict(*inp)
      y_curr = src2vals['y']

      for j in range(self.batch_size):
        index = i * self.batch_size + j
        hypo = probs[0][j]
        ref = y_curr[j][0]
        if hypo == 1:
          numHypo += 1
          if hypo == ref:
            tp += 1
        if ref == 1:
          numRef += 1
    getF1(tp, numHypo, numRef)

class F1MultiClassesExtension(SimpleExtension):
  def __init__(self, layer3, y, model, data_stream, num_samples, batch_size, **kwargs):
    super(F1MultiClassesExtension, self).__init__(**kwargs)
    y_hat = layer3.results(y)
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y' and inp.name != 'y_types1' and inp.name != 'y_types2'] # name of input variables in graph
    self.predict = theano.function(self.theinputs, y_hat)
    self.data_stream = data_stream
    self.num_samples = num_samples
    self.batch_size = batch_size

  def do(self, which_callback, *args):
    if isinstance(self.batch_size, Iterable):
      num_batches = self.num_samples
      print "multi instance evaluation"
    else:
      num_batches = self.num_samples / self.batch_size
    epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)
    tp = 0
    numHypo = 0
    numRef = 0
    for i in range(num_batches):
      src2vals = epoch_iter.next()
      inp = [src2vals[src.name] for src in self.theinputs]
      result = self.predict(*inp)
      y_curr = src2vals['y']
      #print len(y_curr)
      if isinstance(self.batch_size, Iterable): # we have a bag
        max_pos = -1
        hypo = 0
        for instance in range(self.batch_size[i]):
          curHypo = result[0][instance]
          curProb = result[2][instance][curHypo]
	  #print curHypo, curProb
          if curHypo > 0: # positive instance
            if curProb > max_pos:
              max_pos = curProb
              hypo = curHypo
        #if hypo != 0 or sum(result[0]) > 0:
        #  print result
        #  print hypo
        ref = y_curr[0][0]
        if ref > 0:
          numRef += 1
        if hypo > 0:
          numHypo += 1
          if hypo == ref:
            tp += 1
      else:
        for j in range(self.batch_size):
          hypo = result[0][j]
          ref = y_curr[j][0]
          if ref > 0:
            numRef += 1
          if hypo > 0:
            numHypo += 1
            if hypo == ref:
              tp += 1
    getF1(tp, numHypo, numRef)

class GetPRcurve(SimpleExtension):
  def __init__(self, layer3, y, model, data_stream, num_samples, batch_size, pr_out_file, **kwargs):
    super(GetPRcurve, self).__init__(**kwargs)
    y_hat = layer3.results(y)
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y' and inp.name != 'y_types1' and inp.name != 'y_types2'] # name of input variables in graph
    self.predict = theano.function(self.theinputs, y_hat)
    self.data_stream = data_stream
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.outfile = pr_out_file

  def do(self, which_callback, *args):
    if isinstance(self.batch_size, Iterable):
      num_batches = self.num_samples
      print "multi instance evaluation"
    else:
      num_batches = self.num_samples / self.batch_size
    epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)
    tp = 0
    numHypo = 0
    numRef = 0
    refs = []
    hypos = []
    hypoConf = []
    fp = open(self.outfile, 'w')
    for i in range(num_batches):
      src2vals = epoch_iter.next()
      inp = [src2vals[src.name] for src in self.theinputs]
      result = self.predict(*inp)
      y_curr = src2vals['y']

      if isinstance(self.batch_size, Iterable): # we have a bag
        max_pos = -1
        hypo = 0
        maxTotal = -1
        for instance in range(self.batch_size[i]):
          curHypo = result[0][instance]
          curProb = result[2][instance][curHypo]
          if curHypo > 0: # positive instance
            if curProb > max_pos:
              max_pos = curProb
              hypo = curHypo
          if curProb > maxTotal: 
            maxTotal = curProb
        if max_pos > -1:
          conf = max_pos      
        else:
          conf = maxTotal
        ref = y_curr[0][0]
        if ref > 0:
          numRef += 1
        refs.append(ref)
        hypos.append(hypo)
        hypoConf.append(conf)
      else:
        for j in range(self.batch_size):
          hypo = result[0][j]
          ref = y_curr[j][0]
          conf = result[2][j][hypo]
          if ref > 0:
            numRef += 1
          refs.append(ref)
          hypos.append(hypo)
          hypoConf.append(conf)

    indices = numpy.argsort(numpy.array(hypoConf))[::-1]
    for i in indices:
      ref = refs[i]
      hypo = hypos[i]
      if hypo > 0:
        numHypo += 1
        if hypo == ref:
          tp += 1
      getF1(tp, numHypo, numRef, fp)

class ALLModelResultsMI(SimpleExtension):
  def __init__(self, layer3, y, model, data_stream, num_samples, batch_size, out_prob_file, out_truth_file, **kwargs):
    super(ALLModelResultsMI, self).__init__(**kwargs)
    y_hat = layer3.results(y)
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y'] # name of input variables in graph
    self.predict = theano.function(self.theinputs, y_hat)
    self.data_stream = data_stream
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.prob_matrix = numpy.zeros(shape=(num_samples, 11))
    self.truth_matrix = numpy.zeros(shape=(num_samples, 11))
    self.out_prob_file = out_prob_file
    self.out_truth_file = out_truth_file
    
  def do(self, which_callback, *args):
    if isinstance(self.batch_size, Iterable):
      num_batches = self.num_samples
      print "multi instance evaluation"
    else:
      num_batches = self.num_samples / self.batch_size
    epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)
    print "ref\thypo\tconfidence for 1"
    for i in range(num_batches):
      src2vals = epoch_iter.next()
      inp = [src2vals[src.name] for src in self.theinputs]
      result = self.predict(*inp)
      y_curr = src2vals['y']

      if isinstance(self.batch_size, Iterable): # we have a bag
        max_pos = -1
        hypo = 0
        maxTotal = -1
        probs = result[2]
        max_prob_per_rel = numpy.max(probs, axis=0)
        self.prob_matrix[i] = max_prob_per_rel
        y_bin_vec = numpy.zeros(shape=(11));
        y_bin_vec[y_curr[0][0]] = 1
        self.truth_matrix[i] = y_bin_vec
    numpy.save(self.out_prob_file, self.prob_matrix)
    numpy.save(self.out_truth_file, self.truth_matrix)
    
class CheckpointAfterEpoch(SimpleExtension):
    """Saves a pickled version of the main loop to the disk.
    The pickled main loop can be later reloaded and training can be
    resumed.
    Makes a `SAVED_TO` record in the log with the serialization destination
    in the case of success and ``None`` in the case of failure. The
    value of the record is a tuple of paths to which saving was done
    (there can be more than one if the user added a condition
    with an argument, see :meth:`do` docs).
    Parameters
    ----------
    path : str
        The destination path for pickling.
    parameters : list, optional
        The parameters to save separately. If None, the parameters from
        the model (main_loop.model.parameters) are saved.
    save_separately : list of str, optional
        The list of the main loop's attributes to be saved (copied)
        in a separate file in the tar archive. It may be used for example
        to save the log separetely. The name of the attribute will be used
        as name in the tar file.
    save_main_loop : bool
        Choose whether to save the main loop or not. This can be useful
        for example if you are only interested in saving the parameters,
        but not the whole main loop. Defaults to `True`.
    use_cpickle : bool
        See documentation of :func:`~blocks.serialization.dump`.
    Notes
    -----
    Using pickling for saving the whole main loop object comes with
    certain limitations:
    * Theano computation graphs build in the GPU-mode
      (`theano.config.device == "gpu"`) can not be used in the usual mode
      (and vice-versa). Therefore using this extension binds you to using
      only one kind of device.
    """
    def __init__(self, path, parameters=None, save_separately=None,
                 save_main_loop=True, use_cpickle=False, **kwargs):
        kwargs.setdefault("after_training", True)
        super(CheckpointAfterEpoch, self).__init__(**kwargs)
        self.path = path + ".1"
        self.parameters = parameters
        self.save_separately = save_separately
        self.save_main_loop = save_main_loop
        self.use_cpickle = use_cpickle

    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk.
        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.
        """
        _, from_user = self.parse_args(callback_name, args)
        if os.path.isfile(self.path):
          count = int(self.path.split('.')[-1])
          count += 1
          self.path = ".".join(self.path.split('.')[:-1]) + "." + str(count)
        try:
            path = self.path
            if from_user:
                path, = from_user
            to_add = None
            if self.save_separately:
                to_add = {attr: getattr(self.main_loop, attr) for attr in
                          self.save_separately}
            if self.parameters is None:
                if hasattr(self.main_loop, 'model'):
                    self.parameters = self.main_loop.model.parameters
            object_ = None
            if self.save_main_loop:
                object_ = self.main_loop
            secure_dump(object_, path,
                        dump_function=dump_and_add_to_dump,
                        parameters=self.parameters,
                        to_add=to_add,
                        use_cpickle=self.use_cpickle)
        except Exception:
            path = None
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (path,))
