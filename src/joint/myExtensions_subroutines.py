#!/usr/bin/python

from blocks.extensions import SimpleExtension
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
from pickle import HIGHEST_PROTOCOL
try:
    from pickle import DEFAULT_PROTOCOL
    from pickle import _Pickler
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL
    from pickle import Pickler as _Pickler
LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"

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

