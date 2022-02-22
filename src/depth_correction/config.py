from __future__ import absolute_import, division, print_function
from copy import deepcopy
from datetime import datetime
# from enum import Enum
import numpy  # needed in eval
import os
from rospkg import RosPack
import torch  # needed in eval
import yaml

__all__ = [
    'Config',
    'PoseCorrection',
]


# class PoseCorrection(Enum):
# class PoseCorrection(yaml.YAMLObject):
class PoseCorrection(object):
    """Pose correction of ground-truth or estimated poses."""
    none = 'none'
    # Common for all sequences
    common = 'common'
    # Common for all poses within sequence
    sequence = 'sequence'
    # Separate correction of each pose
    pose = 'pose'

    # yaml_tag = u'!PoseCorrection'
    # def __str__(self):
    #     return self.value
    #
    # def __repr__(self):
    #     return str(self)


class Config(object):
    """Depth correction config.

    Only basic Python types should be used as values."""
    def __init__(self, **kwargs):
        self.pkg_dir = RosPack().get_path('depth_correction')
        self.enable_ros = False

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
        self.grid_res = 0.1
        # Neighborhood
        self.nn_k = None
        self.nn_r = 0.2

        # Depth correction
        self.eig_bounds = [[0,    None, 0.02**2],
                           [1, 0.05**2,    None]]

        # Data
        self.dataset = 'asl_laser'
        self.train_names = ['eth']
        self.val_names = ['stairs']
        self.test_names = ['gazebo_winter']
        # print('Training set: %s.' % ', '.join(self.train_names))
        # print('Validation set: %s.' % ', '.join(self.val_names))
        # print('Test set: %s.' % ', '.join(self.val_names))
        self.data_step = 5
        self.world_frame = 'world'

        # Training
        self.loss = 'min_eigval_loss'
        # self.loss = 'trace_loss'
        self.n_opt_iters = 100

        # self.optimizer = 'Adam'
        # self.optimizer_args = []
        # self.optimizer_kwargs = {}
        # self.lr = 1e-4
        self.optimizer = 'SGD'
        self.optimizer_args = []
        self.optimizer_kwargs = {'momentum': 0.9, 'nesterov': True}
        self.lr = 1e-3

        self.pose_correction = PoseCorrection.none
        self.train_pose_deltas = None
        self.log_dir = os.path.join(self.pkg_dir, 'gen',
                                    datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.loss_eval_csv = None
        self.slam_eval_csv = None

        self.log_filters = False
        self.show_results = False
        self.plot_period = 10
        self.plot_size = 6.4, 6.4

        # Override from kwargs
        self.from_dict(kwargs)

    def __getitem__(self, name):
        return getattr(name)

    def from_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def from_yaml(self, cfg):
        if isinstance(cfg, str):
            with open(cfg, 'r') as f:
                try:
                    d = yaml.safe_load(f)
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

    def get_log_dir(self):
        name = ('depth_%.1f-%.1f_grid_%.2f_r%.2f'
                % (self.min_depth, self.max_depth, self.grid_res, self.nn_r))
        dir = os.path.join(self.pkg_dir, 'gen', name)
        return dir

    def copy(self):
        return deepcopy(self)
