from __future__ import absolute_import, division, print_function
from .configurable import Configurable, ValueEnum
from datetime import datetime
from math import radians, isfinite
import os
from subprocess import DEVNULL, PIPE, run
import yaml

__all__ = [
    'Config',
    'fix_bounds',
    'nonempty',
    'PoseCorrection',
    'PoseProvider',
    'SLAM',
]


def cmd_out(cmd, cwd=None, stdout=None, stderr=None):
    # https://stackoverflow.com/questions/89228/calling-an-external-command-in-python
    # out = run(cmd, check=True, stdout=PIPE, stderr=DEVNULL).stdout.decode()
    if stdout is None and stderr is None:
        ret = run(cmd, capture_output=True, check=True, cwd=cwd)
    else:
        ret = run(cmd, check=True, cwd=cwd, stdout=stdout, stderr=stderr)
    out = ret.stdout.decode() if ret.stdout else ''
    err = ret.stderr.decode() if ret.stderr else ''
    return out, err


def fix_bounds(bounds):
    bounds = [float(x) if x is not None and isfinite(x) else float('nan') for x in bounds]
    return bounds


def nonempty(iterable):
    return filter(bool, iterable)


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


def loss_eval_csv(log_dir: str, loss: str, subset: str=None):
    if subset:
        path = 'loss_eval_{loss}_{subset}.csv'.format(loss=loss, subset=subset)
    else:
        path = 'loss_eval_{loss}.csv'.format(loss=loss)
    if log_dir:
        path = os.path.join(log_dir, path)
    return path


def slam_eval_csv(log_dir: str, slam: str, subset: str=None):
    if subset:
        path = 'slam_eval_{slam}_{subset}.csv'.format(slam=slam, subset=subset)
    else:
        path = 'slam_eval_{slam}.csv'.format(slam=slam)
    if log_dir:
        path = os.path.join(log_dir, path)
    return path


def loss_eval_csv(log_dir: str, loss: str, subset: str=None):
    if subset:
        path = 'loss_eval_{loss}_{subset}.csv'.format(loss=loss, subset=subset)
    else:
        path = 'loss_eval_{loss}.csv'.format(loss=loss)
    if log_dir:
        path = os.path.join(log_dir, path)
    return path


def slam_eval_bag(log_dir: str, slam: str):
    path = 'slam_eval_{slam}.bag'.format(slam=slam)
    if log_dir:
        path = os.path.join(log_dir, path)
    return path


def slam_poses_csv(log_dir: str, name: str, slam: str):
    if name:
        path = os.path.join(name, 'slam_poses_{slam}.csv'.format(slam=slam))
    else:
        path = os.path.join('slam_poses_{slam}.csv'.format(slam=slam))
    if log_dir:
        path = os.path.join(log_dir, path)
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
        self.random_seed = 135

        self.pkg_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # self.version = '{git_log}\n{git_diff}'.format(git_log=cmd_out(['git', 'log',  '-1'], cwd=self.pkg_dir)[0],
        #                                               git_diff=cmd_out(['git', 'diff'], cwd=self.pkg_dir)[0])
        self.version = cmd_out(['git', 'rev-parse', 'HEAD'], cwd=self.pkg_dir)[0].strip()
        # print(self.version)
        # Output directory root.
        self.out_dir = os.path.join(self.pkg_dir, 'gen')
        # Output directory for the experiment, defaults to date and time, customizable.
        self.log_dir = os.path.join(self.out_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.enable_ros = False
        self.ros_master_port = 11513

        self.pose_provider = PoseProvider.ground_truth
        self.slam = SLAM.norlab_icp_mapper
        self.model_class = Model.ScaledPolynomial
        self.model_args = []
        self.model_kwargs = {}
        # self.model_kwargs['w'] = [0.0]
        # self.model_kwargs['exponent'] = [4.0]
        # self.model_kwargs['learnable_exponents'] = False
        self.model_state_dict = ''
        self.float_type = 'float64'
        self.device = 'cpu'

        # Cloud preprocessing
        self.min_depth = 1.0
        self.max_depth = 20.0
        self.grid_res = 0.1
        # Neighborhood
        self.nn_k = 0
        self.nn_r = 0.25

        # Depth correction
        self.min_valid_neighbors = 5
        self.shadow_neighborhood_angle = 0.017453  # 1 deg
        # self.shadow_angle_bounds = [radians(5.), float('inf')]
        self.shadow_angle_bounds = []
        # self.shadow_angle_bounds = None
        self.dir_dispersion_bounds = [0.09, float('inf')]
        self.vp_dispersion_bounds = [0.36, float('inf')]
        # self.vp_dispersion_to_depth2_bounds = [0.2, None]
        self.vp_dispersion_to_depth2_bounds = []
        # self.vp_dist_to_depth_bounds = [0.5, None]
        self.vp_dist_to_depth_bounds = []
        self.eigenvalue_bounds = []
        # self.eigenvalue_bounds = [[0,      -float('inf'), (self.nn_r / 4)**2],
        #                           [1, (self.nn_r / 4)**2,       float('inf')]]
        # self.eigenvalue_ratio_bounds = []
        self.eigenvalue_ratio_bounds = [[0, 1, -float('inf'),         0.25],
                                        [1, 2,          0.25, float('inf')]]

        # Data
        self.dataset = 'asl_laser'
        self.dataset_args = []
        self.dataset_kwargs = {}
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
        self.loss_offset = False
        self.loss_kwargs = {}
        self.loss_kwargs['sqrt'] = False
        self.loss_kwargs['normalization'] = True
        self.n_opt_iters = 100

        self.optimizer = 'Adam'
        self.optimizer_args = []
        self.optimizer_kwargs = {}
        self.optimizer_kwargs['betas'] = (0.5, 0.9)
        self.optimizer_kwargs['weight_decay'] = 0.1
        self.lr = 2e-4
        # self.optimizer = 'SGD'
        # self.optimizer_args = []
        # self.optimizer_kwargs = {'momentum': 0.9, 'nesterov': True}
        # self.lr = 5e-3

        self.pose_correction = PoseCorrection.none
        self.train_pose_deltas = None
        self.loss_eval_csv = None
        self.slam_eval_csv = None
        self.slam_eval_bag = None
        self.slam_poses_csv = None
        # Testing
        self.eval_losses = list(Loss)
        self.eval_slams = list(SLAM)

        self.log_filters = False
        self.show_results = False
        self.plot_period = 10
        self.plot_size = 6.4, 6.4
        self.rviz = False

        # Override from kwargs
        self.from_dict(kwargs)

    def numpy_float_type(self):
        import numpy as np
        return getattr(np, self.float_type)

    def torch_float_type(self):
        import torch
        return getattr(torch, self.float_type)

    def sanitize(self):
        if isinstance(self.shadow_angle_bounds, str):
            self.shadow_angle_bounds = yaml.safe_load(self.shadow_angle_bounds)
        self.shadow_angle_bounds = self.shadow_angle_bounds or []
        self.shadow_angle_bounds = fix_bounds(self.shadow_angle_bounds)

        if isinstance(self.eigenvalue_bounds, str):
            self.eigenvalue_bounds = yaml.safe_load(self.eigenvalue_bounds)
        eigenvalue_bounds = []
        for i, min, max in self.eigenvalue_bounds:
            if not isinstance(i, int) or i < 0:
                continue
            min, max = fix_bounds([min, max])
            eigenvalue_bounds.append([i, min, max])
        if eigenvalue_bounds != self.eigenvalue_bounds:
            print('eigenvalue_bounds: %s -> %s' % (self.eigenvalue_bounds, eigenvalue_bounds))
        self.eigenvalue_bounds = eigenvalue_bounds

        if isinstance(self.dir_dispersion_bounds, str):
            self.dir_dispersion_bounds = yaml.safe_load(self.dir_dispersion_bounds)
        self.dir_dispersion_bounds = self.dir_dispersion_bounds or []
        self.dir_dispersion_bounds = fix_bounds(self.dir_dispersion_bounds)

        if isinstance(self.vp_dispersion_bounds, str):
            self.vp_dispersion_bounds = yaml.safe_load(self.vp_dispersion_bounds)
        self.vp_dispersion_bounds = self.vp_dispersion_bounds or []
        self.vp_dispersion_bounds = fix_bounds(self.vp_dispersion_bounds)

        if isinstance(self.vp_dispersion_to_depth2_bounds, str):
            self.vp_dispersion_to_depth2_bounds = yaml.safe_load(self.vp_dispersion_to_depth2_bounds)
        self.vp_dispersion_to_depth2_bounds = self.vp_dispersion_to_depth2_bounds or []
        self.vp_dispersion_to_depth2_bounds = fix_bounds(self.vp_dispersion_to_depth2_bounds)

    def get_depth_filter_desc(self):
        desc = 'd%.0f-%.0f' % (self.min_depth, self.max_depth)
        return desc

    def get_grid_filter_desc(self):
        desc = 'g%.2f' % self.grid_res
        return desc

    def get_shadow_filter_desc(self):
        desc = ''
        if self.shadow_angle_bounds:
            desc = 's%.3g_%.3g-%.3g' % tuple([self.shadow_neighborhood_angle] + self.shadow_angle_bounds)
        return desc

    def get_nn_desc(self):
        desc = ''
        if self.nn_k:
            desc += 'k%i' % self.nn_k
        if self.nn_r:
            if desc:
                desc += '_'
            desc += 'r%.2f' % self.nn_r
        # if not desc:
        #     desc = 'none'
        return desc

    def get_eigval_bounds_desc(self):
        desc = ''
        for i, min, max in self.eigenvalue_bounds:
            if desc:
                desc += '_'
            desc += 'e%i_%.3g-%.3g' % (i, min, max)
        # if not desc:
        #     desc = 'none'
        return desc

    def get_eigval_ratio_bounds_desc(self):
        desc = ''
        for i, j, min, max in self.eigenvalue_ratio_bounds:
            if desc:
                desc += '_'
            desc += 'e%ie%i_%.3g-%.3g' % (i, j, min, max)
        # if not desc:
        #     desc = 'none'
        return desc

    def get_dir_dispersion_desc(self):
        desc = ''
        if self.dir_dispersion_bounds:
            desc = 'dd_%.3g-%.3g' % tuple(fix_bounds(self.dir_dispersion_bounds))
        return desc

    def get_vp_dispersion_desc(self):
        desc = ''
        if self.vp_dispersion_bounds:
            desc = 'vpd_%.3g-%.3g' % tuple(fix_bounds(self.vp_dispersion_bounds))
        return desc

    def get_vp_dispersion_to_depth2_desc(self):
        desc = ''
        if self.vp_dispersion_to_depth2_bounds:
            desc = 'vpdd_%.3g-%.3g' % tuple(fix_bounds(self.vp_dispersion_to_depth2_bounds))
        return desc

    def get_loss_desc(self):
        desc = self.loss
        # loss_kwargs = '_'.join('%s_%s' % (k, v) for k, v in self.loss_kwargs.items())
        # if loss_kwargs:
        #     desc += '_' + loss_kwargs
        for k, v in self.loss_kwargs.items():
            if k == 'normalization':
                desc += '_norm%i' % v
                continue
            if k == 'sqrt':
                desc += '_sqrt%i' % v
                continue
            desc += '%s_%s' % (k, v)
        return desc

    def get_preproc_desc(self):
        self.sanitize()
        parts = [self.dataset,
                 self.get_depth_filter_desc(),
                 self.get_grid_filter_desc(),
                 self.get_shadow_filter_desc()]
        desc = '_'.join(nonempty(parts))
        return desc

    def get_preproc_dir(self):
        dir = os.path.join(self.out_dir, self.get_preproc_desc())
        return dir

    def get_exp_desc(self):
        parts = [self.pose_provider,
                 self.pose_correction,
                 self.model_class,
                 self.get_nn_desc(),
                 self.get_eigval_bounds_desc(),
                 self.get_eigval_ratio_bounds_desc(),
                 self.get_dir_dispersion_desc(),
                 self.get_vp_dispersion_desc(),
                 self.get_vp_dispersion_to_depth2_desc(),
                 self.get_loss_desc()]
        desc = '_'.join(nonempty(parts))
        return desc

    def get_exp_dir(self):
        dir = os.path.join(self.get_preproc_dir(), self.get_exp_desc())
        return dir


def test():
    cfg = Config()

    cfg.from_dict({'nn_k': 5, 'grid_res': 0.5})
    assert cfg.nn_k == 5
    assert cfg.grid_res == 0.5

    cfg.from_args(['--nn-k', '10'])
    assert cfg.nn_k == 10

    cfg.from_roslaunch_args(['nn_r:=.inf'])
    assert cfg.nn_r == float('inf')

    value = [[0, None, 1.0], [1, 1.0, float('inf')]]
    cfg.eigenvalue_bounds = value
    args = cfg.to_roslaunch_args(keys=['eigenvalue_bounds'])
    assert args[0] == 'eigenvalue_bounds:=[[0, null, 1.0], [1, 1.0, .inf]]'

    cfg.from_roslaunch_args(args)
    assert cfg.eigenvalue_bounds == value


def main():
    test()


if __name__ == '__main__':
    main()
