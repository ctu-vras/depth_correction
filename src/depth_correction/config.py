from __future__ import absolute_import, division, print_function
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
    ethzasl_icp_mapper = 'ethzasl_icp_mapper'


class PoseProvider(metaclass=ValueEnum):
    ground_truth = 'ground_truth'


# Add SLAM pipelines to possible pose providers.
for slam in SLAM:
    setattr(PoseProvider, slam, slam)


class Config(object):
    """Depth correction config.

    Only basic Python types should be used as values."""
    def __init__(self, **kwargs):
        self.pkg_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.enable_ros = False
        self.ros_master_port = 11513

        self.slam = 'ethzasl_icp_mapper'
        self.model = None
        self.model_class = 'ScaledPolynomial'
        self.model_state_dict = None
        # self.dtype = np.float64
        self.float_type = 'float64'
        # self.device = torch.device('cpu')
        self.device = 'cpu'

        # Cloud preprocessing
        self.min_depth = 1.0
        self.max_depth = 15.0
        # self.grid_res = 0.05
        self.grid_res = 0.1
        # Neighborhood
        self.nn_k = None
        # self.nn_r = 0.15
        self.nn_r = 0.2

        # Depth correction
        self.eigenvalue_bounds = [[0,    None, 0.02**2],
                                  [1, 0.05**2,    None]]

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
        self.loss = 'min_eigval_loss'
        # self.loss = 'trace_loss'
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
        self.slam_poses_csv = None
        # Testing
        self.eval_losses = ['min_eigval_loss', 'trace_loss']
        self.eval_slams = ['ethzasl_icp_mapper']

        self.log_filters = False
        self.show_results = False
        self.plot_period = 10
        self.plot_size = 6.4, 6.4
        self.rviz = False

        # Override from kwargs
        self.from_dict(kwargs)

    def __getitem__(self, name):
        return getattr(name)

    def from_dict(self, d):
        for k, v in d.items():
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

    def to_dict(self):
        return vars(self)

    def to_yaml(self, path=None):
        if path is None:
            return yaml.safe_dump(self.to_dict())
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)

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

    def get_log_dir(self):
        self.sanitize()
        # name = ('depth_%.1f-%.1f_grid_%.2f_r%.2f'
        #         % (self.min_depth, self.max_depth, self.grid_res, self.nn_r))
        if self.nn_r:
            nn = 'r%.2f' % self.nn_r
        elif self.nn_k:
            nn = 'k%i' % self.nn_k
        else:
            nn = 'none'
        e = 'e'
        for i, min, max in self.eigenvalue_bounds:
            e += '%i_%.3g-%.3g' % (i, min, max)
        name = ('%s_d%.0f-%.0f_g%.2f_%s_%s'
                % (self.dataset, self.min_depth, self.max_depth, self.grid_res, nn, e))
        dir = os.path.join(self.pkg_dir, 'gen', name)
        return dir

    def copy(self):
        return deepcopy(self)
