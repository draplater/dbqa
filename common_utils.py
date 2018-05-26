import argparse
import gzip
from io import open
import contextlib
import functools
import warnings
from argparse import ArgumentParser
from optparse import OptionParser

import os
import time

import sys

from itertools import islice


def set_proc_name(newname):
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(len(newname) + 1)
    buff.value = newname.encode("utf-8")
    libc.prctl(15, byref(buff), 0, 0, 0)


def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as err:
        if err.errno != 17:
            raise


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


@deprecated
def parse_dict(parser, dic, prefix=()):
    from training_scheduler import dict_to_commandline
    return parser.parse_args(dict_to_commandline(dic, prefix))


def under_construction(func):
    """This is a decorator which can be used to mark functions
    as under construction. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to under construction function {}.".format(func.__name__), category=UserWarning,
                      stacklevel=2)
        return func(*args, **kwargs)

    return new_func


class Timer(object):
    def __init__(self):
        self.time = time.time()

    def tick(self):
        oldtime = self.time
        self.time = time.time()
        return self.time - oldtime


@contextlib.contextmanager
def smart_open(filename, mode="r", *args, **kwargs):
    if filename != '-':
        fh = open(filename, mode, *args, **kwargs)
    else:
        if mode.startswith("r"):
            fh = sys.stdin
        elif mode.startswith("w") or mode.startswith("a"):
            fh = sys.stdout
        else:
            raise ValueError("invalid mode " + mode)

    try:
        yield fh
    finally:
        if fh is not sys.stdout and fh is not sys.stdin:
            fh.close()


def split_to_batches(iterable, batch_size):
    iterator = iter(iterable)
    sent_id = 0
    batch_id = 0

    while True:
        piece = list(islice(iterator, batch_size))
        if not piece:
            break
        yield sent_id, batch_id, piece
        sent_id += len(piece)
        batch_id += 1


class AttrDict(dict):
    """A dict whose items can also be accessed as member variables.

    >>> d = AttrDict(a=1, b=2)
    >>> d['c'] = 3
    >>> print d.a, d.b, d.c
    1 2 3
    >>> d.b = 10
    >>> print d['b']
    10

    # but be careful, it's easy to hide methods
    >>> print d.get('c')
    3
    >>> d['get'] = 4
    >>> print d.get('a')
    Traceback (most recent call last):
    TypeError: 'int' object is not callable
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    @property
    def __dict__(self):
        return self


class IdentityDict(object):
    """ A dict like IdentityHashMap in java"""

    def __init__(self, seq=None):
        self.dict = dict(seq=((id(key), value) for key, value in seq))

    def __setitem__(self, key, value):
        self.dict[id(key)] = value

    def __getitem__(self, item):
        return self.dict[id(item)]

    def get(self, key, default=None):
        return self.dict.get(id(key), default)

    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return repr(self.dict)

    def __getstate__(self):
        raise NotImplementedError("Cannot pickle this.")


def dict_key_action_factory(choices):
    class DictKeyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, choices[values])

    return DictKeyAction


class DictionarySubParser(argparse._ArgumentGroup):
    def __init__(self, sub_namespace, original_parser, choices,
                 title=None,
                 description=None,
                 default_key="default"):
        super(DictionarySubParser, self).__init__(
            original_parser, title=title, description=description)
        self.sub_namespace = sub_namespace
        self.original_parser = original_parser
        self.original_parser.add_argument("--" + self.sub_namespace,
                                          action=dict_key_action_factory(choices),
                                          choices=choices.keys(),
                                          default=choices[default_key]
                                          )

    def add_argument(self, *args, **kwargs):
        def modify_names(name):
            last_hyphen = -1
            for i, char in enumerate(name):
                if char == "-":
                    last_hyphen = i
                else:
                    break
            last_hyphen += 1
            return name[:last_hyphen] + self.sub_namespace + "." + name[last_hyphen:]

        if "dest" in kwargs:
            kwargs["dest"] = self.sub_namespace + "." + kwargs["dest"]

        original_action_input = kwargs.get("action")
        if original_action_input is None or \
                isinstance(original_action_input, (str, bytes)):
            original_action_class = self._registry_get(
                "action", original_action_input, original_action_input)
        else:
            original_action_class = original_action_input
        kwargs["action"] = group_action_factory(self.sub_namespace, original_action_class)
        kwargs["default"] = argparse.SUPPRESS
        self.original_parser.add_argument(
            *[modify_names(i) for i in args],
            **kwargs)


def group_action_factory(group_name, original_action_class):
    class GroupAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            assert dest.startswith(group_name + ".")
            dest = dest[len(group_name)+1:]
            super(GroupAction, self).__init__(option_strings, dest, **kwargs)
            self.original_action_obj = original_action_class(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            groupspace = getattr(namespace, group_name)
            self.original_action_obj(parser, groupspace, values, option_string)

    return GroupAction


def read_embedding(embedding_filename, encoding):
    if embedding_filename.endswith(".gz"):
        external_embedding_fp = gzip.open(embedding_filename, 'rb')
    else:
        external_embedding_fp = open(embedding_filename, 'rb')

    def embedding_gen():
        for line in external_embedding_fp:
            fields = line.decode(encoding).strip().split(' ')
            if len(fields) <= 2:
                continue
            token = fields[0]
            vector = [float(i) for i in fields[1:]]
            yield token, vector

    external_embedding = list(embedding_gen())
    external_embedding_fp.close()
    return external_embedding