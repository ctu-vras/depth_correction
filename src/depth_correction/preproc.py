#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .filters import *
from .model import *
import numpy as np
import torch

__all__ = [
    'filtered_cloud',
    'global_cloud',
    'global_cloud_mask',
    'local_feature_cloud',
    'offset_cloud',
]


def filtered_cloud(cloud, cfg: Config):
    if ((cfg.min_depth is not None and cfg.min_depth > 0.0)
            or (cfg.max_depth is not None and cfg.max_depth < float('inf'))):
        cloud = filter_depth(cloud, min=cfg.min_depth, max=cfg.max_depth, log=cfg.log_filters)
    if cfg.grid_res > 0.0:
        rng = np.random.default_rng(cfg.random_seed)
        cloud = filter_grid(cloud, grid_res=cfg.grid_res, keep='random', log=cfg.log_filters, rng=rng)
    return cloud


def local_feature_cloud(cloud, cfg: Config):
    # Convert to depth cloud if needed.
    if isinstance(cloud, np.ndarray):
        if cloud.dtype.names:
            cloud = DepthCloud.from_structured_array(cloud, dtype=cfg.numpy_float_type(), device=cfg.device)
        else:
            cloud = DepthCloud.from_points(cloud, dtype=cfg.numpy_float_type(), device=cfg.device)
    assert isinstance(cloud, DepthCloud)

    # Remove shadow points.
    if cfg.shadow_angle_bounds:
        cloud.update_dir_neighbors(angle=cfg.shadow_neighborhood_angle)
        cloud = filter_shadow_points(cloud, cfg.shadow_angle_bounds, log=cfg.log_filters)

    # Find/update neighbors and estimate all features.
    cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)

    # Select planar regions to correct in prediction phase.
    if cfg.eigenvalue_bounds:
        if cloud.mask is None:
            cloud.mask = torch.ones((len(cloud),), dtype=torch.bool, device=cloud.device())
        cloud.mask = cloud.mask & filter_eigenvalues(cloud, cfg.eigenvalue_bounds, only_mask=True, log=cfg.log_filters)

    if cfg.eigenvalue_ratio_bounds:
        if cloud.mask is None:
            cloud.mask = torch.ones((len(cloud),), dtype=torch.bool, device=cloud.device())
        cloud.mask = cloud.mask & filter_eigenvalue_ratios(cloud, cfg.eigenvalue_ratio_bounds, only_mask=True,
                                                           log=cfg.log_filters)

    return cloud


def offset_cloud(clouds: (list, tuple),
                 model: BaseModel):
    corrected_clouds = []
    for i, cloud in enumerate(clouds):
        # Model updates the cloud using its mask.
        if model is not None:
            cloud = model(cloud)
        corrected_clouds.append(cloud)
    fields = DepthCloud.source_fields + ['eigvals']
    cloud = DepthCloud.concatenate(corrected_clouds, fields=fields)
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
        # Model updates the cloud using its mask.
        if model is not None:
            cloud = model(cloud)
        cloud = cloud.transform(poses[i])
        transformed_clouds.append(cloud)
    cloud = DepthCloud.concatenate(transformed_clouds, dependent=True)
    return cloud


def global_cloud_mask(cloud: DepthCloud, mask: torch.Tensor, cfg: Config):

    # Construct point mask from global cloud filters.
    if mask is None:
        mask = torch.ones((len(cloud),), dtype=torch.bool)
    else:
        print('%.3f = %i / %i points kept (previous filters).'
              % (mask.double().mean(), mask.sum(), mask.numel()))

    if cfg.min_valid_neighbors:
        mask &= filter_valid_neighbors(cloud, min=cfg.min_valid_neighbors, only_mask=True, log=cfg.log_filters)
    # Enforce bound on eigenvalues (done for local clouds).
    if cfg.eigenvalue_bounds:
        mask &= filter_eigenvalues(cloud, eig_bounds=cfg.eigenvalue_bounds, only_mask=True, log=cfg.log_filters)
    # Enforce minimum direction and viewpoint spread for bias estimation.
    if cfg.dir_dispersion_bounds:
        # cloud.visualize(colors=cloud.dir_dispersion(), window_name='Direction dispersion')
        mask &= within_bounds(cloud.dir_dispersion(), bounds=cfg.dir_dispersion_bounds,
                              log_variable='dir dispersion' if cfg.log_filters else None)
    if cfg.vp_dispersion_bounds:
        # cloud.visualize(colors=cloud.vp_dispersion(), window_name='Viewpoint dispersion')
        mask &= within_bounds(cloud.vp_dispersion(), bounds=cfg.vp_dispersion_bounds,
                              log_variable='vp dispersion' if cfg.log_filters else None)
    if cfg.vp_dispersion_to_depth2_bounds:
        mask &= within_bounds(cloud.vp_dispersion_to_depth2(), bounds=cfg.vp_dispersion_to_depth2_bounds,
                              log_variable='vp dispersion to depth2' if cfg.log_filters else None)
    return mask
