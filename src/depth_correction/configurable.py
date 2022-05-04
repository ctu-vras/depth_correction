from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from copy import deepcopy
import yaml

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
            new[k] = yaml.safe_load(v)

        self.from_dict(new)

        return remainder

    def from_rosparam(self, prefix='~'):
        import rospy
        for k in self:
            name = prefix + k
            if rospy.has_param(name):
                self[k] = rospy.get_param(name, self[k])
                if isinstance(self[k], str):
                    self[k] = yaml.safe_load(self[k])
                # print('%s: %s (%s)' % (k, self[k], type(self[k]).__name__))

    def to_dict(self):
        return vars(self)

    def to_roslaunch_args(self, non_default=False, keys=None):
        if not keys:
            if non_default:
                keys = self.non_default().keys()
            else:
                keys = self.to_dict().keys()

        args = []
        for k in keys:
            v = yaml.safe_dump(self[k], default_flow_style=True)
            # Remove trailing newlines and document end indicator ("...");
            # this depends on the input value / dict.
            v = v.strip('\n')
            v = v.strip('\n...')
            arg = '%s:=%s' % (k, v)
            args.append(arg)

        return args

    def from_roslaunch_args(self, args):
        for arg in args:
            assert isinstance(arg, str)
            k, v = arg.split(':=', maxsplit=1)
            self[k] = yaml.safe_load(v)

    def to_yaml(self, path=None):
        if path is None:
            return yaml.safe_dump(self.to_dict())
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
    cfg.int_var = None
    cfg.float_var = 0.2
    cfg.list_var = [0.1, float('inf'), float('nan'), None]
    cfg.dict_var = {}

    cfg.from_dict({'int_var': 5, 'float_var': 0.5})
    assert cfg.int_var == 5
    assert cfg.float_var == 0.5

    cfg.from_args(['--int-var', '10'])
    assert cfg.int_var == 10

    cfg.from_roslaunch_args(['float_var:=.inf'])
    assert cfg.float_var == float('inf')

    list_var = cfg.list_var
    args = cfg.to_roslaunch_args(keys=['list_var'])
    assert args[0] == 'list_var:=[0.1, .inf, .nan, null]'
    cfg.from_roslaunch_args(args)
    assert len(cfg.list_var) == len(list_var)
    assert all(a == b or (isnan(a) and isnan(b)) for a, b in zip(cfg.list_var, list_var))


def main():
    test()


if __name__ == '__main__':
    main()
