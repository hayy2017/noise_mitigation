#!/usr/bin/python

############
# Description: Blocks extensions and classes for data stream
# Author: Heike Adel
# Year: 2016
# Info: some helper functions based on code from fuel github
###########

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
from blocks.serialization import secure_dump
import pickle
from pickle import HIGHEST_PROTOCOL
try:
    from pickle import DEFAULT_PROTOCOL
    from pickle import _Pickler
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL
    from pickle import Pickler as _Pickler
LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"

from fuel.schemes import IterationScheme
from picklable_itertools import iter_, imap
from picklable_itertools.extras import partition_all
from collections import Iterable
import six
from picklable_itertools.base import BaseItertool


class _PersistentLoad(object):
    """Loads object saved using a PersistentID mechanism."""
    def __init__(self, tar_file):
        self.tar_file = tar_file
        if '_parameters' in tar_file.getnames():
            self.parameters = numpy.load(
                tar_file.extractfile(tar_file.getmember('_parameters')))

    def __call__(self, id_):
        components = _unmangle_parameter_name(id_)
        return components[0](self.parameters[components[1]])

def _unmangle_parameter_name(mangled_name):
    if mangled_name.startswith('#'):
        type_, name = mangled_name[1:].split('.', 1)
        return _INVERSE_ARRAY_TYPE_MAP[type_], name

def load(file_, name='_pkl', use_cpickle=False):
    """Loads an object saved using the `dump` function.
    By default, this function loads the object saved by the `dump`
    function. If some objects have been added to the archive using the
    `add_to_dump` function, then you can load them by passing their name
    to the `name` parameter.
    Parameters
    ----------
    file_ : file
        The file that contains the object to load.
    name : str
        Name of the object to load. Default is `_pkl`, meaning that it is
        the original object which have been dumped that is loaded.
    use_cpickle : bool
        Use cPickle instead of pickle. Default: False.
    Returns
    -------
    The object saved in file_.
    """
    file_.seek(0)  # To be able to read several objects in one file
    if use_cpickle:
        unpickler = cPickle.Unpickler
    else:
        unpickler = pickle.Unpickler
    with tarfile.open(fileobj=file_, mode='r') as tar_file:
        p = unpickler(
            tar_file.extractfile(tar_file.getmember(name)))
        if '_parameters' in tar_file.getnames():
            p.persistent_load = _PersistentLoad(tar_file)
        return p.load()

def _mangle_parameter_name(type_, name):
    return '#{}.{}'.format(_ARRAY_TYPE_MAP[type_], name)

_ARRAY_TYPE_MAP = {numpy.ndarray: 'numpy_ndarray'}
_INVERSE_ARRAY_TYPE_MAP = {'numpy_ndarray': numpy.array}

class _SaveObject(object):
    r"""Saves an object using Persistent ID.
    Parameters
    ----------
    pickler : object
        The pickler to use
    object_ : object
        The object to pickle.
    external_objects : dict of object
        The external objects to save using persistent id.
    protocol : int, optional
        The pickling protocol to use.
    \*\*kwargs
        Keyword arguments to be passed to `pickle.Pickler`.
    """
    def __init__(self, pickler, object_, external_objects, protocol, **kwargs):
        self.pickler = pickler
        self.object_ = object_
        self.external_objects = external_objects
        self.protocol = protocol
        self.kwargs = kwargs

    def __call__(self, f):
        p = self.pickler(f, protocol=self.protocol, **self.kwargs)
        p.persistent_id = _PersistentID(self.external_objects)
        p.dump(self.object_)


class _PersistentID(object):
    """Returns persistent identifiers for objects saved separately."""
    def __init__(self, external_objects):
        self.external_objects = external_objects

    def __call__(self, object_):
        return self.external_objects.get(id(object_))

class _Renamer(object):
    """Returns a new name for the given parameter.
    It maintains a list of names already used to avoid naming
    collisions. It also provides names for variables without
    names.
    Attributes
    ----------
    used_names : set
        The set of names already taken.
    default_name : str
        The name to use if a parameter doesn't have a name. Default:
        'parameter'.
    """
    def __init__(self):
        self.used_names = set()
        self.default_name = 'parameter'

    def __call__(self, parameter):
        # Standard Blocks parameter
        if get_brick(parameter) is not None:
            name = '{}.{}'.format(
                BRICK_DELIMITER.join(
                    [""] + [brick.name for brick in
                            get_brick(parameter).get_unique_path()]),
                parameter.name)
        # Shared variables with tag.name
        elif hasattr(parameter.tag, 'name'):
            name = parameter.tag.name
        # Standard shared variable
        elif parameter.name is not None:
            name = parameter.name
        # Variables without names
        else:
            name = self.default_name
        # Handle naming collisions
        if name in self.used_names:
            i = 2
            new_name = '_'.join([name, str(i)])
            while new_name in self.used_names:
                i += 1
                new_name = '_'.join([name, str(i)])
            name = new_name
        self.used_names.add(name)
        return name

def _taradd(func, tar_file, name):
    """Adds elements dumped by the function `func` to a tar_file.
    This functions first calls the function `func` and add the file that
    `func` dumps to the achive `tar_file`, under the name `name`.
    Parameters
    ----------
    func : function
        The dumping function.
    tar_file : file
        The archive that we are filling.
    name : str
        The name of the dumped file in the archive.
    """
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        func(temp_file)
        temp_file.close()
        tar_file.add(temp_file.name, arcname=name)
    if os.path.isfile(temp_file.name):
        os.remove(temp_file.name)

class _PicklerWithWarning(_Pickler):
    """Pickler that adds a warning message.
    Adds a warning message if we try to save an object referenced in the
    main module.
    """
    dispatch = _Pickler.dispatch.copy()

    def save_global(self, obj, name=None, **kwargs):
        module = getattr(obj, '__module__', None)
        if module == '__main__':
            warnings.warn(
                MAIN_MODULE_WARNING.format(kwargs.get('name', obj.__name__))
            )
        _Pickler.save_global(self, obj, name=name, **kwargs)

    dispatch[six.types.FunctionType] = save_global
    if six.PY2:
        dispatch[six.types.ClassType] = save_global
        dispatch[six.types.BuiltinFunctionType] = save_global
        dispatch[six.types.TypeType] = save_global

def dump(object_, file_, parameters=None, use_cpickle=False,
         protocol=DEFAULT_PROTOCOL, **kwargs):
    r"""Pickles an object, optionally saving its parameters separately.
    Parameters
    ----------
    object_ : object
        The object to pickle. If None, only the parameters passed to the
        `parameters` argument will be saved.
    file_ : file
        The destination for saving.
    parameters : list, optional
        Shared variables whose internal numpy arrays should be saved
        separately in the `_parameters` field of the tar file.
    pickle_object : bool
        If False, `object_` will not be serialized, only its parameters.
        This flag can be used when `object_` is not serializable, but one
        still want to save its parameters. Default: True
    use_cpickle : bool
        Use cPickle instead of pickle. Setting it to true will disable the
        warning message if you try to pickle objects from the main module,
        so be sure that there is no warning before turning this flag
        on. Default: False.
    protocol : int, optional
        The pickling protocol to use. Unlike Python's built-in pickle, the
        default is set to `2` instead of 0 for Python 2. The Python 3
        default (level 3) is maintained.
    \*\*kwargs
        Keyword arguments to be passed to `pickle.Pickler`.
    """
    if use_cpickle:
        pickler = cPickle.Pickler
    else:
        pickler = _PicklerWithWarning
    with closing(tarfile.TarFile(fileobj=file_, mode='w')) as tar_file:
        external_objects = {}

        def _save_parameters(f):
            renamer = _Renamer()
            named_parameters = {renamer(p): p for p in parameters}
            numpy.savez(f, **{n: p.get_value()
                              for n, p in named_parameters.items()})
            for n, p in named_parameters.items():
                array_ = p.container.storage[0]
                external_objects[id(array_)] = _mangle_parameter_name(
                    type(array_), n)
        if parameters:
            _taradd(_save_parameters, tar_file, '_parameters')
        if object_ is not None:
            save_object = _SaveObject(pickler, object_, external_objects,
                                      protocol, **kwargs)
            _taradd(save_object, tar_file, '_pkl')



def dump_and_add_to_dump(object_, file_, parameters=None, to_add=None,
                         use_cpickle=False, protocol=DEFAULT_PROTOCOL,
                         **kwargs):
    r"""Calls both `dump` and `add_to_dump` to serialze several objects.
    This function is used to serialize several at the same time, using
    persistent ID. Its main advantage is that it can be used with
    `secure_dump`.
    Parameters
    ----------
    object_ : object
        The object to pickle. If None, only the parameters passed to the
        `parameters` argument will be saved.
    file_ : file
        The destination for saving.
    parameters : list, optional
        Shared variables whose internal numpy arrays should be saved
        separately in the `_parameters` field of the tar file.
    to_add : dict of objects
        A {'name': object} dictionnary of additional objects to save in
        the tar archive. Its keys will be used as name in the tar file.
    use_cpickle : bool
        Use cPickle instead of pickle. Setting it to true will disable the
        warning message if you try to pickle objects from the main module,
        so be sure that there is no warning before turning this flag
        on. Default: False.
    protocol : int, optional
        The pickling protocol to use. Unlike Python's built-in pickle, the
        default is set to `2` instead of 0 for Python 2. The Python 3
        default (level 3) is maintained.
    \*\*kwargs
        Keyword arguments to be passed to `pickle.Pickler`.
    """
    dump(object_, file_, parameters=parameters, use_cpickle=use_cpickle,
         protocol=protocol, **kwargs)
    if to_add is not None:
        for name, obj in six.iteritems(to_add):
            add_to_dump(obj, file_, name, parameters=parameters,
                        use_cpickle=use_cpickle, protocol=protocol, **kwargs)

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
  def __init__(self, layer3, y, model, data_stream, num_samples, batch_size, name, **kwargs):
    super(F1MultiClassesExtension, self).__init__(**kwargs)
    y_hat = layer3.results(y)
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y'] # name of input variables in graph
    self.predict = theano.function(self.theinputs, y_hat)
    self.data_stream = data_stream
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.name = name

  def do(self, which_callback, *args):
    print "F1 on " + str(self.name)
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
      if isinstance(self.batch_size, Iterable): # we have a bag
        max_pos = -1
        hypo = 0
        for instance in range(self.batch_size[i]):
          curHypo = result[0][instance]
          curProb = result[2][instance][curHypo]
          #print str(curHypo) + " " + str(curProb)
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
  def __init__(self, layer3, y, model, data_stream, num_samples, batch_size, **kwargs):
    super(GetPRcurve, self).__init__(**kwargs)
    y_hat = layer3.results(y)
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y'] # name of input variables in graph
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
    refs = []
    hypos = []
    hypoConf = []
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
      getF1(tp, numHypo, numRef)

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


##################### fuel classes ####################


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


class MultiInstanceScheme(IterationScheme):
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

      def get_request_iterator(self):
          return imap(list, My_partition_all(self.batch_size_list, self.indices))

class MultiInstanceSchemeShuffled(IterationScheme):
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
          tmp = list(My_partition_all(self.batch_size_list, self.indices))
          self.rng.shuffle(tmp)
          return imap(list, tmp)


def getF1(tp, numHypo, numRef):
    precision = 1
    recall = 0
    f1 = 0
    if numHypo > 0:
      precision = 1.0 * tp / numHypo
    if numRef > 0:
      recall = 1.0 * tp / numRef
    if precision + recall > 0:
      f1 = 2 * precision * recall / (precision + recall)
    print str(time.ctime()) + "\tP = " + str(precision) + ", R = " + str(recall) + ", F1 = " + str(f1)
    return f1 
