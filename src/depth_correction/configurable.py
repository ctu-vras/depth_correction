from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from copy import deepcopy
import os
import sys
import yaml
from yaml.error import YAMLError

__all__ = [
    'Configurable',
    'ValueEnum',
]


# https://stackoverflow.com/a/10814662
class ValueEnum(type):
    """Simple enumeration type with plain user-defined values."""
    def __iter__(self):
        return iter((f for f in vars(self) if not f.startswith('_')))

    def __contains__(self, item):
        return item in iter(self)


class Format(metaclass=ValueEnum):
    yaml = 'yaml'
    none = 'none'


class Configurable(object):
    """Object configurable from command-line arguments or YAML."""

    DEFAULT = object()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.to_dict().keys())

    def from_dict(self, d):
        old = self.to_dict()
        for k, v in d.items():
            if k in old:
                self[k] = v

    def from_yaml(self, path):
        if isinstance(path, str):
            with open(path, 'r') as f:
                try:
                    d = yaml.safe_load(f)
                    if d:  # Don't raise exception in case of empty yaml.
                        self.from_dict(d)
                except yaml.YAMLError as ex:
                    print(ex)

    def from_args(self, args):
        # Construct argument definitions from current config.
        parser = ArgumentParser()
        for k in self:
            arg = '--%s' % '-'.join(k.split('_'))
            parser.add_argument(arg, type=str, default=Configurable.DEFAULT)

        # Parse arguments and values as YAML.
        parsed_args, remainder = parser.parse_known_args(args)
        new = {}
        for k, v in vars(parsed_args).items():
            if v == Configurable.DEFAULT:
                continue
            try:
                new[k] = yaml.safe_load(v)
            except YAMLError as ex:
                print('Could not parse YAML "%s" from argument %s: %s.' % (v, k, ex), file=sys.stderr)

        self.from_dict(new)

        return remainder

    def from_rosparam(self, prefix='~', format=Format.none):
        assert format in Format
        import rospy
        for k in self:
            name = prefix + k
            if rospy.has_param(name):
                v = rospy.get_param(name, self[k])
                # Allow parsing strings as YAML, if needed.
                if isinstance(v, str) and format == Format.yaml:
                    try:
                        self[k] = yaml.safe_load(v)
                    except YAMLError as ex:
                        print('Could not parse YAML "%s" from rosparam %s: %s.' % (v, name, ex), file=sys.stderr)
                # Note, that YAML is already parsed within rosparam, so values
                # from parameter server shouldn't be parsed (again) in general.
                else:
                    self[k] = v

    def to_dict(self):
        return vars(self)

    def to_roslaunch_args(self, non_default=False, keys=None, format=Format.yaml):
        assert format in Format
        if not keys:
            if non_default:
                keys = self.non_default().keys()
            else:
                keys = self.to_dict().keys()

        args = []
        for k in keys:
            if format == Format.yaml:
                v = yaml.safe_dump(self[k], default_flow_style=True)
                # Remove trailing newlines and document end indicator ("...");
                # this depends on the input value / dict.
                v = v.strip('\n')
                v = v.strip('\n...')
            elif format == Format.none:
                if isinstance(self[k], bool):
                    v = 'true' if self[k] else 'false'
                else:
                    v = str(self[k])
            arg = '%s:=%s' % (k, v)
            args.append(arg)

        return args

    def from_roslaunch_args(self, args, format=Format.yaml):
        assert format in Format
        for arg in args:
            assert isinstance(arg, str)
            k, v = arg.split(':=', maxsplit=1)
            if k not in self:
                continue
            if format == Format.yaml:
                try:
                    v = yaml.safe_load(v)
                    self[k] = v
                except YAMLError as ex:
                    print('Could not parse YAML "%s" from roslaunch argument %s: %s.' % (v, k, ex), file=sys.stderr)
            elif format == Format.none:
                if v == 'true':
                    v = True
                elif v == 'false':
                    v = False
                else:
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                self[k] = v

    def to_yaml(self, path=None):
        if path is None:
            return yaml.safe_dump(self.to_dict())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)

    def diff(self, cfg):
        d = {}
        for k in self:
            if self[k] != cfg[k]:
                d[k] = self[k]
        return d

    def non_default(self):
        cfg = self.__class__()
        d = self.diff(cfg)
        return d

    def copy(self):
        return deepcopy(self)


def test():
    from math import isnan

    cfg = Configurable()
    cfg.bool_var = True
    cfg.int_var = None
    cfg.float_var = 0.2
    cfg.list_var = [0.1, float('inf'), float('nan'), None]
    cfg.dict_var = {}
    cfg.str_var = ''

    cfg.from_dict({'int_var': 5, 'float_var': 0.5})
    assert cfg.int_var == 5
    assert cfg.float_var == 0.5

    cfg.from_args(['--int-var', '10'])
    assert cfg.int_var == 10

    cfg.from_roslaunch_args(['float_var:=.inf'])
    assert cfg.float_var == float('inf')

    args = cfg.to_roslaunch_args(keys=['str_var'])
    assert args[0] == "str_var:=''", args
    cfg.from_roslaunch_args(args)

    list_var = cfg.list_var
    args = cfg.to_roslaunch_args(keys=['list_var'])
    assert args[0] == 'list_var:=[0.1, .inf, .nan, null]'
    cfg.from_roslaunch_args(args)
    assert len(cfg.list_var) == len(list_var)
    assert all(a == b or (isnan(a) and isnan(b)) for a, b in zip(cfg.list_var, list_var))

    cfg.from_roslaunch_args(['bool_var:=false'], format=Format.none)
    assert isinstance(cfg.bool_var, bool)
    assert cfg.bool_var is False
    args = cfg.to_roslaunch_args(keys=['bool_var'], format=Format.none)
    assert args == ['bool_var:=false']

    cfg.from_roslaunch_args(['float_var:=1e-3'], format=Format.none)
    assert isinstance(cfg.float_var, float)
    assert cfg.float_var == 1e-3
    args = cfg.to_roslaunch_args(keys=['float_var'], format=Format.none)
    assert args == ['float_var:=0.001']

    cfg.from_roslaunch_args(['int_var:=10'], format=Format.none)
    assert isinstance(cfg.int_var, int)
    assert cfg.int_var == 10
    args = cfg.to_roslaunch_args(keys=['int_var'], format=Format.none)
    assert args == ['int_var:=10']

    cfg.from_roslaunch_args(['str_var:=1e10 string'], format=Format.none)
    assert isinstance(cfg.str_var, str)
    assert cfg.str_var == '1e10 string'
    args = cfg.to_roslaunch_args(keys=['str_var'], format=Format.none)
    assert args == ['str_var:=1e10 string']


def main():
    test()


if __name__ == '__main__':
    main()
