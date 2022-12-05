from __future__ import absolute_import, division, print_function
from .config import Config, NeighborhoodType
from .depth_cloud import DepthCloud
from .filters import *
from .model import *
from .segmentation import Planes
from .transform import xyz_axis_angle_to_matrix
from .utils import covs, timing
from .visualization import visualize_incidence_angles
import numpy as np
import torch

__all__ = [
    'compute_neighborhood_features',
    'establish_neighborhoods',
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


def global_cloud(clouds: (list, tuple)=None,
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
        return cloud.neighbors, cloud.weights
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
        cloud.neighbors, cloud.weights = neighborhoods
        cloud.update_all(scale=cfg.nn_scale, keep_neighbors=True)
        return cloud
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
    from data.asl_laser import Dataset, dataset_names
    prefix = 'asl_laser'
    # from data.newer_college import Dataset, dataset_names, prefix
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
    from .dataset import create_dataset
    from numpy.lib.recfunctions import structured_to_unstructured
    import open3d as o3d
    from scipy.spatial import cKDTree

    cfg = Config()
    cfg.grid_res = 0.1
    cfg.min_depth = 0.5
    cfg.max_depth = 25.0

    cfg.nn_type = NeighborhoodType.ball
    cfg.nn_r = 0.5
    cfg.min_valid_neighbors = 5

    def process(cloud, pose, cfg):
        cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
        cloud = filtered_cloud(cloud, cfg)
        cloud = local_feature_cloud(cloud, cfg)
        cloud = cloud.transform(pose)
        return cloud

    for name in dataset_names:
        print('Creating dataset: %s\n' % name)
        if not name.startswith(prefix):
            name = '%s/%s' % (prefix, name)
        ds = create_dataset(name, cfg=cfg)

        losses = []
        for id1, id2 in zip(ds.ids[:-1], ds.ids[1:]):
            cloud1 = ds.local_cloud(id1)
            pose1 = torch.as_tensor(ds.cloud_pose(id1))
            cloud1 = process(cloud1, pose1, cfg)

            cloud2 = ds.local_cloud(id2)
            pose2 = torch.as_tensor(ds.cloud_pose(id2))
            cloud2 = process(cloud2, pose2, cfg)

            points1 = cloud1.to_points()
            points2 = cloud2.to_points()
            vps_dists = torch.linalg.norm(pose1[:3, 3] - pose2[:3, 3])

            # find intersections between neighboring point clouds (1 and 2)
            dist_th = 0.1

            tree = cKDTree(points2)
            dists, idxs = tree.query(points1, k=1)
            common_pts_mask = dists <= dist_th
            assert len(dists) == points1.shape[0]
            assert len(idxs) == points1.shape[0]
            pts1_inters = points1[common_pts_mask]
            normals1_inters = cloud1.normals[common_pts_mask]

            tree = cKDTree(points1)
            dists, idxs = tree.query(points2, k=1)
            common_pts_mask = dists <= dist_th
            # assert len(dists) == points2.shape[0]
            # assert len(idxs) == points2.shape[0]
            pts2_inters = points2[common_pts_mask]

            # find corresponding closest points for intersecting parts of clouds
            tree = cKDTree(pts2_inters)
            dists, idxs = tree.query(pts1_inters, k=1)
            # assert len(dists) == pts1_inters.shape[0]
            # assert len(idxs) == pts1_inters.shape[0]
            pts2_inters = torch.index_select(pts2_inters, 0, torch.as_tensor(idxs))
            # assert pts1_inters.shape == pts2_inters.shape
            # assert normals1_inters.shape == pts1_inters.shape
            # assert np.allclose(np.linalg.norm(normals1_inters, axis=1), np.ones(len(normals1_inters)))

            # point to plane distance
            vectors = pts2_inters - pts1_inters
            normals = normals1_inters
            dists_to_plane = torch.multiply(vectors, normals).sum(dim=1).abs()
            loss12 = dists_to_plane.mean()
            losses.append(loss12.item())

            print('Mean point to plane distance: %.3f [m] for 2 scans located %.3f [m] apart (sequence: %s)' %
                  (loss12.item(), vps_dists.item(), name))

            # # visualization
            # pcd1 = o3d.geometry.PointCloud()
            # # pcd1.points = o3d.utility.Vector3dVector(points1)
            # # pcd1.colors = o3d.utility.Vector3dVector(np.zeros_like(points1) + np.array([0, 1, 0]))
            # # pcd1.normals = o3d.utility.Vector3dVector(cloud1.normals)
            # pcd1.points = o3d.utility.Vector3dVector(pts1_inters)
            # pcd1.colors = o3d.utility.Vector3dVector(np.zeros_like(pts1_inters) + np.array([0, 1, 0]))
            #
            # pcd2 = o3d.geometry.PointCloud()
            # # pcd2.points = o3d.utility.Vector3dVector(points2)
            # # pcd2.colors = o3d.utility.Vector3dVector(np.zeros_like(points2) + np.array([1, 0, 0]))
            # # pcd2.normals = o3d.utility.Vector3dVector(cloud2.normals)
            # pcd2.points = o3d.utility.Vector3dVector(pts2_inters)
            # pcd2.colors = o3d.utility.Vector3dVector(np.zeros_like(pts2_inters) + np.array([1, 0, 0]))
            #
            # o3d.visualization.draw_geometries([pcd1, pcd2])
            # # o3d.visualization.draw_geometries([pcd1], point_show_normal=True)

        print('\nPoint to plane loss for data sequence %s is : %.3f\n' % (name, np.mean(losses)))


def main():
    # demo_neighborhood_features()
    demo_point_to_plane()


if __name__ == '__main__':
    main()
