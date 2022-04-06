#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .filters import filter_depth, filter_eigenvalues, filter_grid
from .model import *
import torch

__all__ = [
    'filtered_cloud',
    'global_cloud',
    'global_clouds',
    'local_feature_cloud',
]


def filtered_cloud(cloud, cfg: Config):
    cloud = filter_depth(cloud, min=cfg.min_depth, max=cfg.max_depth, log=cfg.log_filters)
    cloud = filter_grid(cloud, grid_res=cfg.grid_res, keep='random', log=cfg.log_filters)
    return cloud


def local_feature_cloud(cloud, cfg: Config):
    # Convert to depth cloud and transform.
    # cloud = DepthCloud.from_structured_array(cloud, dtype=cfg.numpy_float_type())
    if cloud.dtype.names:
        cloud = DepthCloud.from_structured_array(cloud, dtype=cfg.numpy_float_type())
    else:
        cloud = DepthCloud.from_points(cloud, dtype=cfg.numpy_float_type())
    cloud = cloud.to(device=cfg.device)
    # Find/update neighbors and estimate all features.
    cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
    # Select planar regions to correct in prediction phase.
    cloud.mask = filter_eigenvalues(cloud, cfg.eigenvalue_bounds, only_mask=True, log=cfg.log_filters)
    return cloud


def global_cloud(clouds: (list, tuple),
                 model: BaseModel,
                 poses: torch.Tensor):
    """Create global cloud with corrected depth.

    :param clouds: Filtered local features clouds.
    :param model: Depth correction model, directly applicable to clouds.
    :param poses: N-by-4-by-4 pose tensor to transform clouds to global frame.
    :return: Global cloud with corrected depth.
    """
    transformed_clouds = []
    for i, cloud in enumerate(clouds):
        if model is not None:
            cloud = model(cloud)
        cloud = cloud.transform(poses[i])
        transformed_clouds.append(cloud)
    cloud = DepthCloud.concatenate(transformed_clouds, True)
    # cloud.visualize(colors='z')
    # cloud.visualize(colors='inc_angles')
    return cloud


def global_clouds(clouds, model, poses):
    ret = []
    for c, p in zip(clouds, poses):
        cloud = global_cloud(c, model, p)
        ret.append(cloud)
    return ret
