from __future__ import absolute_import, division, print_function
from .config import Config, NeighborhoodType
from .depth_cloud import DepthCloud
from .filters import *
from .model import *
from .point_cloud import PointCloud
from .segmentation import Planes
from .transform import xyz_axis_angle_to_matrix
from .utils import covs, map_colors, timing
from .visualization import visualize_incidence_angles
from matplotlib import cm
import numpy as np
import open3d as o3d
import torch
from typing import Tuple

__all__ = [
    'compute_neighborhood_features',
    'establish_neighborhoods',
    'filtered_cloud',
    'global_cloud',
    'global_cloud_mask',
    'local_feature_cloud',
    'offset_cloud',
]


class Neighborhoods(PointCloud):

    def __init__(self, cloud, indices, weights):
        # Assume cloud is a list of same clouds.
        assert isinstance(cloud, list)
        assert not cloud or all(c is cloud[0] for c in cloud)
        # indices = torch.as_tensor(indices)
        assert isinstance(indices, torch.Tensor)
        # assert isinstance(indices, list)
        assert isinstance(weights, torch.Tensor)
        super(Neighborhoods, self).__init__(cloud=cloud, indices=indices, weights=weights)
    
    def visualize(self, x=None):
        """Visualize neighborhoods using referenced cloud and indices."""
        if x is None:
            x = self.cloud[0]
        pcd = o3d.geometry.PointCloud()
        if isinstance(x, DepthCloud):
            x = x.to_points()
        pcd.points = o3d.utility.Vector3dVector(x)
        num_primitives = self.size()
        num_points = len(x)
        labels = np.full(num_points, -1, dtype=int)
        for i in range(num_primitives):
            labels[self.indices[i]] = i
        max_label = num_primitives - 1
        colors = np.zeros((num_points, 3), dtype=np.float32)
        segmented = labels >= 0
        # colors[segmented] = map_colors(labels[segmented], colormap=cm.viridis, min_value=0.0, max_value=max_label)
        colors[segmented] = map_colors(labels[segmented], colormap=cm.gist_rainbow, min_value=0.0, max_value=max_label)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])


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


def offset_cloud(clouds: Tuple[list, tuple],
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


def global_cloud(clouds: Tuple[list, tuple]=None,
                 model: BaseModel=None,
                 poses: torch.Tensor=None,
                 pose_corrections: torch.Tensor=None,
                 dataset=None):
    """Create global cloud with corrected depth.

    :param clouds: Filtered local features clouds.
    :param model: Depth correction model, directly applicable to clouds.
    :param poses: N-by-4-by-4 pose tensor to transform clouds to global frame.
    :param pose_corrections: N-by-4-by-4 pose correction tensor for poses.
    :param dataset: Dataset to get clouds and poses from.
    :return: Global cloud with corrected depth.
    """
    # Create clouds and poses if dataset is provided.
    if dataset is not None:
        assert clouds is None
        assert poses is None
        clouds, poses = zip(*dataset)
        clouds = [DepthCloud.from_structured_array(cloud, dtype=np.float64) for cloud in clouds]
        poses = torch.as_tensor(np.array(poses))
    assert clouds is not None
    assert poses is not None

    # Apply pose corrections if provided.
    if pose_corrections is not None:
        if pose_corrections.shape[-1] == 6:
            pose_corrections = xyz_axis_angle_to_matrix(pose_corrections)
        poses = poses @ pose_corrections

    # Transform clouds with corrected poses.
    transformed_clouds = []
    for i, cloud in enumerate(clouds):
        # Model updates the cloud using its mask.
        if model is not None:
            # cloud = local_feature_cloud(cloud)
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
    # Enforce bound on eigenvalues (done for local clouds too).
    if cfg.eigenvalue_bounds:
        mask &= filter_eigenvalues(cloud, bounds=cfg.eigenvalue_bounds, only_mask=True, log=cfg.log_filters)
    if cfg.eigenvalue_ratio_bounds:
        mask &= filter_eigenvalue_ratios(cloud, bounds=cfg.eigenvalue_ratio_bounds, only_mask=True, log=cfg.log_filters)
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


# @timing
def establish_neighborhoods(dataset=None, clouds=None, poses=None, cloud=None, cfg: Config=None):
    """Establish local neighborhoods using specified type and config parameters.

    :param dataset: Dataset to create depth cloud from.
    :param cloud: Depth cloud to establish neighborhoods in.
    :param cfg: Config with neighborhood parameters.
    :return:
    """
    # if dataset is not None:
    #     assert cloud is None
    #     cloud = global_cloud(dataset=dataset)

    if cloud is None:
        cloud = global_cloud(clouds=clouds, poses=poses, dataset=dataset)
    assert cloud is not None

    if cfg.nn_type == NeighborhoodType.ball:
        cloud.update_all(k=cfg.nn_k, r=cfg.nn_r, scale=cfg.nn_scale, keep_neighbors=False)
        # return cloud.neighbors, cloud.weights

        # Computed weights are neglected, so we don't allow nn scale.
        assert cfg.nn_scale is None
        # TODO: Filter neighborhoods based on given filters.
        ind = global_cloud_mask(cloud, None, cfg=cfg).nonzero().ravel()
        # Mask to indices.
        # ind = torch.nonzero(mask_1)
        # TODO: Select subset in a grid.
        with torch.no_grad():
            x = cloud.to_points()
        x = x[ind]
        mask = filter_grid(x, cfg.nn_grid_res, only_mask=True)
        ind = ind[mask]

        indices = cloud.neighbors[ind]
        weights = cloud.weights[ind]
        cloud = len(indices) * [cloud]
        nn = Neighborhoods(cloud, indices, weights)
        # nn.visualize()

        return nn

    elif cfg.nn_type == NeighborhoodType.plane:
        planes = Planes.fit(cloud, cfg.ransac_dist_thresh, min_support=cfg.min_valid_neighbors,
                            max_iterations=cfg.num_ransac_iters, max_models=cfg.max_neighborhoods,
                            eps=2.0 * np.sqrt(3) * cfg.grid_res,
                            visualize_progress=False, visualize_final=cfg.log_filters, verbose=0)
        return planes


# @timing
def compute_neighborhood_features(dataset=None, clouds=None, poses=None, model=None, pose_corrections=None, cloud=None,
                                  neighborhoods=None, cfg: Config=None):
    """Compute neighborhood features for provided neighborhoods, using config parameters.

    :param dataset: Dataset to create depth cloud from.
    :param model: Depth cloud correction model.
    :param cloud: Depth cloud to apply neighborhoods at.
    :param neighborhoods: Neighborhoods at which to compute the features.
    :param cfg: Config with feature parameters.
    :return:
    """
    if neighborhoods is None:
        neighborhoods = establish_neighborhoods(dataset=dataset, cloud=cloud, cfg=cfg)
    if cloud is None:
        # For plane neighborhood, model is applied later with plane normals.
        cloud = global_cloud(clouds=clouds,
                             model=model if cfg.nn_type == NeighborhoodType.ball else None,
                             poses=poses, pose_corrections=pose_corrections,
                             dataset=dataset)
    assert neighborhoods is not None
    if cfg.nn_type == NeighborhoodType.ball:
        # cloud.neighbors, cloud.weights = neighborhoods
        # cloud.update_all(scale=cfg.nn_scale, keep_neighbors=True)
        # return cloud

        assert isinstance(neighborhoods, Neighborhoods)
        # TODO: Compute normals and update incidence angles (where?).
        # Incidence angles from source cloud are used,
        # each point has its own corresponding to its neighborhood.

        # TODO: Compute covs and eigvals for all neighborhoods.
        # output.eigvals is used in loss.
        cloud = neighborhoods.cloud[0]
        neighbor_points = cloud.get_points()[neighborhoods.indices]
        cov = covs(neighbor_points, weights=neighborhoods.weights)
        neighborhoods.covs = cov
        # get_neighbor_points()
        eigvals, eigvecs = torch.linalg.eigh(cov)
        neighborhoods.eigvals = eigvals
        neighborhoods.eigvecs = eigvecs
        return neighborhoods


    elif cfg.nn_type == NeighborhoodType.plane:
        planes = neighborhoods.copy()
        plane_clouds = []
        covs_all = []
        eigvals_all = []
        for i in range(len(planes)):
            # TODO: Fix zero input cloud mask (features with small nn_r invalid?).
            #  Don't use the mask until it is fixed.
            plane_cloud = cloud[planes.indices[i]]
            plane_cloud.mask = None
            plane_cloud.normals = planes.params[i:i + 1, :-1].expand((len(planes.indices[i]), -1))
            plane_cloud.update_incidence_angles()
            if model is not None:
                plane_cloud = model(plane_cloud)
            plane_clouds.append(plane_cloud)
            x = plane_cloud.to_points()
            cov = covs(x)
            covs_all.append(cov)
            eigvals_all.append(torch.linalg.eigh(cov)[0])
        if cfg.log_filters:
            visualize_incidence_angles(plane_clouds)
        planes.plane_cloud = plane_clouds
        planes.cov = torch.stack(covs_all)
        planes.eigvals = torch.stack(eigvals_all)
        planes.check()
        return planes


def demo_neighborhood_features():
    # from data.asl_laser import Dataset, dataset_names, prefix
    # from data.newer_college import Dataset, dataset_names, prefix
    from data.depth_correction import dataset_names, prefix
    from .dataset import create_dataset
    from .loss import create_loss
    cfg = Config()
    cfg.grid_res = 0.1  # asl_laser
    # cfg.grid_res = 0.25  # newer_college
    cfg.min_depth = 0.5
    cfg.max_depth = 25.0

    # cfg.nn_type = NeighborhoodType.ball
    # cfg.nn_r = 0.5
    # cfg.min_valid_neighbors = 5

    cfg.nn_type = NeighborhoodType.plane
    cfg.ransac_model_size = 3
    cfg.ransac_model_dist = 0.03
    cfg.min_valid_neighbors = 2000
    cfg.max_neighborhoods = 10

    cfg.ransac_iters = 1000

    loss_fun = create_loss(cfg)

    for name in dataset_names:
        if not name.startswith(prefix):
            name = '%s/%s' % (prefix, name)
        ds = create_dataset(name, cfg=cfg)
        ns = establish_neighborhoods(dataset=ds, cfg=cfg)
        feats = compute_neighborhood_features(dataset=ds, neighborhoods=ns, cfg=cfg)
        l, _ = loss_fun(feats)
        print('%s: loss %.6f.' % (name, l.item()))


def demo_point_to_plane():
    from data.depth_correction import dataset_names, prefix
    # from data.asl_laser import dataset_names, prefix
    from .dataset import create_dataset
    from .loss import point_to_plane_dist
    from numpy.lib.recfunctions import structured_to_unstructured
    import open3d as o3d
    from time import time
    from scipy.spatial import cKDTree

    cfg = Config()
    cfg.data_step = 1
    cfg.grid_res = 0.2
    cfg.min_depth = 0.5
    cfg.max_depth = 25.0

    cfg.nn_type = NeighborhoodType.ball
    cfg.nn_r = 0.5
    cfg.min_valid_neighbors = 5

    for name in dataset_names:
        print('Creating dataset: %s' % name)
        if not name.startswith(prefix):
            name = '%s/%s' % (prefix, name)
        ds = create_dataset(name, cfg=cfg)

        print('Creating featured clouds from dataset sequence: %s...' % name)
        clouds = []
        masks = []
        for id in ds.ids[::cfg.data_step]:
            points = ds.local_cloud(id)
            pose = torch.as_tensor(ds.cloud_pose(id))
            points = structured_to_unstructured(points[['x', 'y', 'z']])
            points = filtered_cloud(points, cfg)
            # cloud = DepthCloud.from_points(points, dtype=cfg.torch_float_type(), device=cfg.device)
            # cloud.update_all(r=cfg.nn_r)
            cloud = local_feature_cloud(points, cfg)
            cloud = cloud.transform(pose)
            clouds.append(cloud)

            if len(clouds) < 2:
                continue
            cloud1 = clouds[-2]
            assert cloud1.normals is not None, "Cloud must have normals computed to estimate point to plane distance"
            cloud2 = clouds[-1]

            points1 = torch.as_tensor(cloud1.to_points(), dtype=torch.float)
            points2 = torch.as_tensor(cloud2.to_points(), dtype=torch.float)

            # find intersections between neighboring point clouds (1 and 2)
            tree = cKDTree(points2)
            dists, ids = tree.query(points1, k=1)
            mask1 = dists <= cfg.loss_kwargs['dist_th']
            mask2 = ids[mask1]
            masks.append((mask1, mask2))

            # # visualization
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(points) + np.array([0, 1, 0]))
            # pcd.normals = o3d.utility.Vector3dVector(cloud.normals)
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        t0 = time()
        loss = point_to_plane_dist(clouds, masks=masks, verbose=True)
        # loss = point_to_plane_dist(clouds, masks=None, differentiable=True, verbose=True)
        print('Point to plane loss for data sequence %s is : %.3f [m] (took %.3f [sec] to compute).\n'
              % (name, loss.item(), time() - t0))


def main():
    # demo_neighborhood_features()
    demo_point_to_plane()


if __name__ == '__main__':
    main()
