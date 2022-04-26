from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
# from enum import Enum
import numpy  # needed in eval
import os
import torch  # needed in eval
import yaml

__all__ = [
    'Config',
    'PoseCorrection',
    'PoseProvider',
    'SLAM',
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

    def __getitem__(self, name):
        return getattr(self, name)

    def __iter__(self):
        return iter(self.to_dict().keys())

    def from_dict(self, d):
        old = self.to_dict()
        for k, v in d.items():
            if k in old:
                setattr(self, k, v)

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

    def to_dict(self):
        return vars(self)

    def to_yaml(self, path=None):
        if path is None:
            return yaml.safe_dump(self.to_dict())
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)

    def diff(self, cfg):
        d = {}
        for k in cfg:
            if cfg[k] != self[k]:
                d[k] = self[k]
        return d

    def non_default(self):
        cfg = self.__class__()
        d = self.diff(cfg)
        return d

    def copy(self):
        return deepcopy(self)


class Loss(metaclass=ValueEnum):
    min_eigval_loss = 'min_eigval_loss'
    trace_loss = 'trace_loss'


class Model(metaclass=ValueEnum):
    Polynomial = 'Polynomial'
    ScaledPolynomial = 'ScaledPolynomial'


class PoseCorrection(metaclass=ValueEnum):
    """Pose correction of ground-truth or estimated poses."""

    none = 'none'
    """No pose correction."""

    common = 'common'
    """Common for all sequences, calibration of shared sensor rig.
    Training update can be used in validation."""

    sequence = 'sequence'
    """Common for all poses within sequence, calibration of sensor rig per sequence.
    Validation poses can be optimized separately (for given model)."""

    pose = 'pose'
    """Separate correction of each pose (adjusting localization from SLAM).
    Validation poses can be optimized separately (for given model)."""


class SLAM(metaclass=ValueEnum):
    # ethzasl_icp_mapper = 'ethzasl_icp_mapper'
    norlab_icp_mapper = 'norlab_icp_mapper'


class PoseProvider(metaclass=ValueEnum):
    ground_truth = 'ground_truth'


# Add SLAM pipelines to possible pose providers.
for slam in SLAM:
    setattr(PoseProvider, slam, slam)


def loss_eval_csv(log_dir: str, loss: str):
    path = os.path.join(log_dir, 'loss_eval_%s.csv' % loss)
    return path


def slam_eval_csv(log_dir: str, slam: str, subset: str):
    path = os.path.join(log_dir, 'slam_eval_%s_%s.csv' % (slam, subset))
    return path


def slam_eval_bag(log_dir: str, slam: str):
    path = os.path.join(log_dir, 'slam_eval_%s.bag' % slam)
    return path


def slam_poses_csv(log_dir: str, name: str, slam: str):
    if name:
        path = os.path.join(log_dir, name, 'slam_poses_%s.csv' % slam)
    else:
        path = os.path.join(log_dir, 'slam_poses_%s.csv' % slam)
    return path


class Config(Configurable):
    """Depth correction config.

    Only basic Python types should be used as values."""
    def __init__(self, **kwargs):
        super(Config, self).__init__()

        # Launch and scheduler options.
        self.launch_prefix = None  # Allows setting launch prefix, e.g., for scheduler.
        self.num_jobs = 0  # Allows debugging with fewer jobs.
        self.force = False   # Allow overwriting existing configs, etc.

        self.pkg_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.enable_ros = False
        self.ros_master_port = 11513

        self.slam = SLAM.norlab_icp_mapper
        self.model = None
        self.model_class = Model.ScaledPolynomial
        self.model_state_dict = None
        # self.dtype = np.float64
        self.float_type = 'float64'
        # self.device = torch.device('cpu')
        self.device = 'cpu'

        # Cloud preprocessing
        self.min_depth = 1.0
        self.max_depth = 25.0
        # self.grid_res = 0.05
        self.grid_res = 0.1
        # Neighborhood
        self.nn_k = None
        # self.nn_r = 0.15
        self.nn_r = 0.2

        # Depth correction
        self.eigenvalue_bounds = [[0,               None, (self.nn_r / 8)**2],
                                  [1, (self.nn_r / 4)**2,               None]]

        # Data
        self.dataset = 'asl_laser'
        self.train_names = ['eth']
        self.val_names = ['stairs']
        self.test_names = ['gazebo_winter']
        # print('Training set: %s.' % ', '.join(self.train_names))
        # print('Validation set: %s.' % ', '.join(self.val_names))
        # print('Test set: %s.' % ', '.join(self.val_names))
        self.train_poses_path = []
        self.val_poses_path = []
        self.test_poses_path = []
        self.data_step = 5
        self.world_frame = 'world'

        # Training
        self.loss = Loss.min_eigval_loss
        # self.loss = Loss.trace_loss
        self.loss_kwargs = {}
        self.n_opt_iters = 100

        self.optimizer = 'Adam'
        self.optimizer_args = []
        self.optimizer_kwargs = {}
        self.lr = 1e-4
        # self.optimizer = 'SGD'
        # self.optimizer_args = []
        # self.optimizer_kwargs = {'momentum': 0.9, 'nesterov': True}
        # self.lr = 5e-3

        self.pose_correction = PoseCorrection.none
        self.train_pose_deltas = None
        self.log_dir = os.path.join(self.pkg_dir, 'gen',
                                    datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.loss_eval_csv = None
        self.slam_eval_csv = None
        self.slam_eval_bag = None
        self.slam_poses_csv = None
        # Testing
        self.eval_losses = list(Loss)
        self.eval_slams = [SLAM.norlab_icp_mapper]

        self.log_filters = False
        self.show_results = False
        self.plot_period = 10
        self.plot_size = 6.4, 6.4
        self.rviz = False

        # Override from kwargs
        self.from_dict(kwargs)

    def numpy_float_type(self):
        return eval('numpy.%s' % self.float_type)

    def torch_float_type(self):
        return eval('torch.%s' % self.float_type)

    def sanitize(self):
        if isinstance(self.eigenvalue_bounds, str):
            self.eigenvalue_bounds = yaml.safe_load(self.eigenvalue_bounds)

        eigenvalue_bounds = []
        for i, min, max in self.eigenvalue_bounds:
            if not isinstance(i, int) or i < 0:
                continue
            if not (isinstance(min, float) and -float('inf') < min < float('inf')):
                min = float('nan')
            if not (isinstance(max, float) and -float('inf') < max < float('inf')):
                max = float('nan')
            eigenvalue_bounds.append([i, min, max])
        if eigenvalue_bounds != self.eigenvalue_bounds:
            print('eigenvalue_bounds: %s -> %s' % (self.eigenvalue_bounds, eigenvalue_bounds))

        self.eigenvalue_bounds = eigenvalue_bounds

    def get_depth_filter_desc(self):
        desc = 'd%.0f-%.0f' % (self.min_depth, self.max_depth)
        return desc

    def get_grid_filter_desc(self):
        desc = 'g%.2f' % self.grid_res
        return desc

    def get_nn_desc(self):
        desc = ''
        if self.nn_k:
            desc += 'k%i' % self.nn_k
        if self.nn_r:
            if desc:
                desc += '_'
            desc += 'r%.2f' % self.nn_r
        if not desc:
            desc = 'none'
        return desc

    def get_eigval_bounds_desc(self):
        desc = ''
        for i, min, max in self.eigenvalue_bounds:
            if desc:
                desc += '_'
            desc += 'e%i_%.3g-%.3g' % (i, min, max)
        if not desc:
            desc = 'none'
        return desc

    def get_loss_desc(self):
        desc = self.loss
        loss_kwargs = '_'.join('%s_%s' % (k, v) for k, v in self.loss_kwargs.items())
        if loss_kwargs:
            desc += '_' + loss_kwargs
        return desc

    def get_log_dir(self):
        self.sanitize()
        name = '_'.join([self.dataset, self.get_depth_filter_desc(), self.get_grid_filter_desc()])
        dir = os.path.join(self.pkg_dir, 'gen', name)
        return dir


def test():
    cfg = Config()
    cfg.from_dict({'nn_k': 5, 'grid_res': 0.5})
    cfg.from_args(['--nn-k', '10'])
    print(cfg.non_default())


def main():
    test()


if __name__ == '__main__':
    main()
