"""Segmentation of points into geometric primitives (planes, etc.)."""
from .config import Config, ValueEnum
from .depth_cloud import DepthCloud
from .point_cloud import PointCloud
from .utils import map_colors, timing, transform
from matplotlib import cm
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import open3d as o3d
import torch

# TODO: Compute loss using planes (geometric primitives)
#   - construct global cloud with provided poses
#   - segment planes (geometric primitives) in global points
#   - estimate normals and incidence angles
#   - correct depth using incidence angles
#   - update points from corrected depth
#   - compute loss
#   - optimize


class Impl(metaclass=ValueEnum):
    pcl = 'pcl'
    open3d = 'open3d'


# class Primitive(object):
class Primitives(PointCloud):
    """Geometric primitives."""

    def __init__(self, params, cloud=None, indices=None):
        params = torch.as_tensor(params)
        # if isinstance(params, np.ndarray):
        #     params = torch.as_tensor(params)
        assert isinstance(params, torch.Tensor)

        # if not isinstance(cloud, DepthCloud):
        #     cloud = torch.as_tensor(cloud)
        #     assert isinstance(cloud, torch.Tensor)
        # Assume cloud is a list of same clouds.
        assert isinstance(cloud, list)
        assert not cloud or all(c is cloud[0] for c in cloud)

        # indices = torch.as_tensor(indices)
        # assert isinstance(indices, torch.Tensor)
        assert isinstance(indices, list)

        # self.params = params
        # self.cloud = cloud
        # self.indices = indices
        super(Primitives, self).__init__(params=params, cloud=cloud, indices=indices)

    def distance(self, x):
        """Compute point distance to surface."""
        raise NotImplementedError('Distance function not implemented.')

    def visualize(self, x=None):
        """Visualize geometric primitive using referenced cloud and indices."""
        # if x is None:
        #     x = self.cloud
        # if isinstance(x, DepthCloud):
        #     labels = np.zeros((x.size(),), dtype=np.float32)
        #     labels[self.idx] = 1.0
        #     x.visualize(colors=labels)
        # elif isinstance(x, torch.Tensor):
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(x)
        #     labels = np.zeros((len(x),), dtype=np.float32)
        #     labels[self.idx] = 1.0
        #     colors = map_colors(labels, colormap=cm.viridis, min_value=0.0, max_value=1.0)
        #     pcd.colors = o3d.utility.Vector3dVector(colors)
        #     o3d.visualization.draw_geometries([pcd])
        # cloud = self.cloud[0]
        if x is None:
            x = self.cloud[0]
        pcd = o3d.geometry.PointCloud()
        # if isinstance(cloud, DepthCloud):
        #     cloud = cloud.to_points()
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
        colors[segmented] = map_colors(labels[segmented], colormap=cm.viridis, min_value=0.0, max_value=max_label)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])


class Planes(Primitives):

    def __init__(self, params, cloud=None, indices=None):
        # params = torch.as_tensor(params).reshape((4,))
        params = torch.as_tensor(params).reshape((-1, 4))
        super(Planes, self).__init__(params, cloud=cloud, indices=indices)

    def distance(self, x):
        if isinstance(x, DepthCloud):
            x = x.get_points()
        else:
            x = torch.as_tensor(x)
        assert isinstance(x, torch.Tensor)
        # d = torch.matmul(x, self.params[:-1, None]).squeeze()
        d = torch.matmul(x, self.params.t()).squeeze()
        return d

    def orient(self, x):
        if isinstance(x, DepthCloud):
            x = x.vps
        else:
            x = torch.as_tensor(x)
        flip = torch.sign(self.distance(x)).mean() < 0.0
        params = -self.params if flip else self.params
        return Planes(params, cloud=self.cloud, indices=self.indices)

    @staticmethod
    def fit(*args, **kwargs):
        return fit_planes(*args, **kwargs)


# @timing
def fit_plane_pcl(x, distance_threshold, max_iterations=1000):
    import pcl
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    cld = pcl.PointCloud(x.astype(np.float32))
    seg = cld.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    seg.set_MaxIterations(max_iterations)
    indices, model = seg.segment()
    return model, indices


# @timing
def fit_plane_open3d(x, distance_threshold, max_iterations=1000):
    import open3d as o3d
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    model, indices = pcd.segment_plane(distance_threshold, 3, max_iterations)
    return model, indices


def fit_plane(x, distance_threshold, max_iterations=1000, impl=Impl.pcl):
    if impl == Impl.pcl:
        plane, support = fit_plane_pcl(x, distance_threshold, max_iterations=max_iterations)
    elif impl == Impl.open3d:
        plane, support = fit_plane_open3d(x, distance_threshold, max_iterations=max_iterations)
    return plane, support


# @timing
def cluster_open3d(x, eps, min_points=10):
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert eps >= 0.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    # NB: High min_points value causes finding no points, even if clusters
    # with enough support are found when using lower min_points value.
    # clustering = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    clustering = pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=False)
    # Invalid labels < 0.
    clustering = np.asarray(clustering)
    return clustering


def keep_mask(n, indices):
    mask = np.zeros(n, dtype=bool)
    mask[indices] = 1
    return mask


def remove_mask(n, indices):
    mask = np.ones(n, dtype=bool)
    mask[indices] = 0
    return mask


@timing
def fit_planes_iter(x, distance_threshold, min_support=3, max_iterations=1000, max_models=10, eps=None,
                    visualize_progress=False, verbose=0, fit_impl=Impl.pcl, cluster_impl=Impl.open3d):
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert fit_impl in (Impl.pcl, Impl.open3d)
    assert cluster_impl == Impl.open3d
    remaining = x
    indices = np.arange(len(remaining))  # Input point indices of remaining point cloud.
    planes = []
    labels = np.full(len(remaining), -1, dtype=int)
    label = 0
    while True:
        plane, support_tmp = fit_plane(remaining, distance_threshold, max_iterations=max_iterations, impl=fit_impl)

        support_tmp = np.asarray(support_tmp)
        if verbose >= 2:
            print('Found plane %i [%.3f, %.3f, %.3f, %.3f] supported by %i / %i (%i) points.'
                  % (label, *plane, len(support_tmp). len(remaining), len(x)))

        if len(support_tmp) < min_support:
            if verbose >= 0:
                print('Halt due to insufficient plane support.')
            break

        # Extract the largest contiguous cluster and keep the rest for next iteration.
        if eps:
            if cluster_impl == Impl.open3d:
                clustering = cluster_open3d(remaining[support_tmp], eps, min_points=min_support)
            clusters, counts = np.unique(clustering[clustering >= 0], return_counts=True)
            if len(counts) == 0 or counts.max() < min_support:
                # Remove all points if there is no cluster with sufficient support.
                mask = remove_mask(len(remaining), support_tmp)
                remaining = remaining[mask]
                indices = indices[mask]
                if verbose >= 2:
                    print('No cluster from plane %i has sufficient support (largest %i < %i).'
                          % (label, counts.max() if len(counts) else 0, min_support))
                if len(remaining) < min_support:
                    if verbose >= 1:
                        print('Not enough points to continue.')
                    break
                continue
            i_max = counts.argmax()
            assert counts[i_max] >= min_support
            largest = clusters[i_max]
            support_tmp = support_tmp[clustering == largest]
            if verbose >= 1:
                print('Kept largest cluster from plane %i [%.3f, %.3f, %.3f, %.3f] supported by %i / %i (%i) points.'
                      % (label, *plane, len(support_tmp), len(remaining), len(x)))

        support = indices[support_tmp]
        planes.append((plane, support))
        labels[support] = label

        if visualize_progress:
            obj = Planes(torch.as_tensor([p for p, _ in planes]),
                         cloud=len(planes) * [x],
                         indices=[i for _, i in planes])
            obj.visualize()

        if len(planes) == max_models:
            if verbose >= 1:
                print('Target number of planes found.')
            break

        mask = remove_mask(len(remaining), support_tmp)
        remaining = remaining[mask]
        indices = indices[mask]
        if len(remaining) < min_support:
            if verbose >= 1:
                print('Not enough points to continue.')
            break
        label += 1

    print('%i planes (highest label %i) with minimum support of %i points were found.'
          % (len(planes), labels.max(), min_support))

    planes = Planes(torch.as_tensor(np.concatenate([p for p, _ in planes])),
                    cloud=len(planes) * [x],
                    indices=[i for _, i in planes])

    return planes


def fit_planes(x, distance_threshold, visualize_final=False, **kwargs):
    """Segment points into planes."""
    # TODO: Move outer loop from implementation specific part here.
    if isinstance(x, DepthCloud):
        x = x.to_points().numpy()
    assert isinstance(x, np.ndarray)
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])

    planes = fit_planes_iter(x, distance_threshold, **kwargs)

    if visualize_final:
        planes.visualize()

    return planes


def demo_pcl():
    from .dataset import create_dataset
    from .utils import timer
    from data.newer_college import dataset_names, prefix
    from numpy.lib.recfunctions import structured_to_unstructured
    import pcl
    cfg = Config()
    cfg.min_depth = 0.5
    cfg.max_depth = 20.0
    cfg.grid_res = 0.2
    cfg.ransac_model_size = 3
    cfg.ransac_dist_thresh = 0.03
    cfg.num_ransac_iters = 1000
    cfg.min_valid_neighbors = 100  # min plane support
    cfg.max_neighborhoods = 10  # max planes
    ds = create_dataset('%s/%s' % (prefix, dataset_names[0]), cfg)
    for cloud, pose in ds:
        remaining = structured_to_unstructured(cloud[['x', 'y', 'z']], dtype=np.float32)
        t = timer()
        planes = []
        while len(remaining) >= cfg.min_valid_neighbors:
            cld = pcl.PointCloud(remaining)
            seg = cld.make_segmenter()
            seg.set_optimize_coefficients(True)
            seg.set_model_type(pcl.SACMODEL_PLANE)
            seg.set_method_type(pcl.SAC_RANSAC)
            seg.set_distance_threshold(cfg.ransac_dist_thresh)
            seg.set_MaxIterations(cfg.num_ransac_iters)
            indices, model = seg.segment()
            if len(indices) < cfg.min_valid_neighbors:
                print('Plane support too low, %i < %i.' % (len(indices), cfg.min_valid_neighbors))
                break
            print('Plane [%.3f, %.3f, %.3f, %.3f] supported by %i / %i points.'
                  % (*model, len(indices), len(remaining)))
            planes.append((model, indices))
            keep = np.ones(len(remaining), dtype=bool)
            keep[indices] = 0
            remaining = remaining[keep]
        t = timer() - t
        print('%i planes found in %.3f s.' % (len(planes), t))


def demo_fit_planes():
    from .dataset import create_dataset
    cfg = Config()
    # cfg.train_names = ['asl_laser/apartment']
    # cfg.train_names = ['asl_laser/eth']
    # cfg.train_names = ['asl_laser/gazebo_summer']
    cfg.train_names = ['asl_laser/gazebo_winter']
    # cfg.train_names = ['asl_laser/stairs']
    cfg.val_names = []
    cfg.test_names = []

    cfg.data_step = 4
    cfg.min_depth = 1.0
    cfg.max_depth = 20.0
    cfg.grid_res = 0.1
    cfg.ransac_dist_thresh = 0.03
    cfg.min_valid_neighbors = 500
    cfg.num_ransac_iters = 500
    cfg.max_neighborhoods = 10
    cfg.from_args()
    for name in cfg.train_names:
        ds = create_dataset(name, cfg=cfg)

        # Segment local point clouds.
        # for cloud, pose in ds:
        #     planes = fit_planes(cfg.ransac_dist_thresh, min_support=cfg.min_valid_neighbors,
        #                     max_iterations=cfg.num_ransac_iters, max_models=cfg.max_neighborhoods,
        #                     eps=np.sqrt(3.0)*cfg.grid_res, visualize_progress=False, visualize_final=True, verbose=1)

        # Segment global cloud.
        cloud = np.concatenate([transform(pose, cloud) for cloud, pose in ds])
        planes = fit_planes(cloud, cfg.ransac_dist_thresh, min_support=cfg.min_valid_neighbors,
                            max_iterations=cfg.num_ransac_iters, max_models=cfg.max_neighborhoods,
                            eps=np.sqrt(3.0)*cfg.grid_res, visualize_progress=False, visualize_final=True, verbose=1)


def main():
    demo_fit_planes()
    # demo_pcl()


if __name__ == '__main__':
    main()
