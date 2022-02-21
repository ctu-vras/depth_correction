from __future__ import absolute_import, division, print_function
import importlib
import numpy as np
import os
import torch
from rospkg import RosPack

__all__ = [
    'config',
    'Config',
]


class Config(object):
    def __init__(self, **kwargs):
        self.pkg_dir = RosPack().get_path('depth_correction')
        self.enable_ros = False

        self.model_class = 'ScaledPolynomial'
        self.model_state_dict = None
        self.dtype = np.float64
        self.device = torch.device('cpu')

        # Cloud preprocessing
        self.min_depth = 1.0
        self.max_depth = 15.0
        self.grid_res = 0.1
        # Neighborhood
        self.nn_k = None
        self.nn_r = 0.1

        # Depth correction
        # self.max_eig_0 = 0.02**2
        # self.min_eig_1 = 0.05**2
        self.eig_bounds = [[0,    None, 0.02**2],
                           [1, 0.05**2,    None]]

        # Data
        self.dataset = 'asl_laser'
        self.train_names = ['apartment', 'eth']
        self.val_names = ['stairs', 'gazebo_winter']
        self.test_names = None
        print('Training set: %s.' % ', '.join(self.train_names))
        print('Validation set: %s.' % ', '.join(self.val_names))
        self.data_step = 3
        self.world_frame = 'world'
        self.n_opt_iters = 100
        self.lr = 1e-3

        self.log_filters = False
        self.show_results = False
        self.plot_period = 10
        self.plot_size = 6.4, 6.4

        # Override from kwargs
        for k, v in kwargs:
            setattr(self, k, v)

        # Post process params
        # self.model_class = eval(self.model_class)

        # from eval('data.%s' % dataset) import Dataset, dataset_names
        # eval('from data.%s import Dataset, dataset_names' % dataset)
        imported_module = importlib.import_module("data.%s" % self.dataset)
        self.Dataset = getattr(imported_module, "Dataset")
        self.dataset_names = getattr(imported_module, "dataset_names")
        print('Using %s datasets %s.' % (self.dataset, ', '.join(self.dataset_names)))


# config = Config()
