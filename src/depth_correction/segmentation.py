"""Segmentation of points into geometric primitives (planes, etc.)."""
from .config import Config
from .dataset import create_dataset
from .depth_cloud import DepthCloud
from .point_cloud import PointCloud
from .utils import map_colors, transform
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

        cloud = self.cloud[0]
        pcd = o3d.geometry.PointCloud()
        if isinstance(cloud, DepthCloud):
            cloud = cloud.to_points()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        num_primitives = self.size()
        num_points = len(cloud)
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


def fit_planes_open3d(x, distance_threshold, min_support=3, num_planes=int(1e9), eps=None,
                      visualize_progress=False, visualize_final=False, verbose=0):
    ransac_n = 3  # Number of points to use to construct model (does LS?).
    num_iters = 1000
    n = x.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    colors = np.zeros((n, 3), dtype=np.float32)
    # pcd.colors = np.zeros((n, 3), dtype=np.float32)
    remaining_pcd = pcd
    indices = np.arange(n)  # Input point indices of remaining point cloud.
    planes = []
    labels = np.full(n, -1, dtype=int)
    label = 0
    # for label in range(num_planes):
    while True:
        plane, support_tmp = remaining_pcd.segment_plane(distance_threshold, ransac_n, num_iters)
        support_tmp = np.asarray(support_tmp)
        if verbose >= 2:
            print('Found plane %i, [%s] supported by %i points.'
                  % (label, ', '.join('%.3f' % x for x in plane), len(support_tmp)))

        if len(support_tmp) < min_support:
            if verbose >= 0:
                print('Halt due to insufficient plane support.')
            break

        # Extract the largest contiguous cluster and keep the rest for next iteration.
        if eps:
            plane_pcd = remaining_pcd.select_by_index(support_tmp)
            # NB: High min_points value causes finding no points, even if clusters
            # with enough support are found when using lower min_points value.
            # clustering = plane_pcd.cluster_dbscan(eps=eps, min_points=min_support, print_progress=True)
            clustering = plane_pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=False)
            clustering = np.asarray(clustering)
            clusters, counts = np.unique(clustering[clustering >= 0], return_counts=True)
            if len(counts) == 0 or counts.max() < min_support:
                # TODO: Remove all points if there is no cluster with sufficient support.
                keep = np.ones(len(indices), dtype=bool)
                keep[support_tmp] = 0
                indices = indices[keep]
                remaining_pcd = remaining_pcd.select_by_index(support_tmp, invert=True)
                if verbose >= 2:
                    print('No cluster from plane %i has sufficient support (largest %i < %i).'
                          % (label, counts.max() if len(counts) else 0, min_support))
                if len(indices) < min_support:
                    if verbose >= 1:
                        print('Not enough points to continue.')
                    break
                continue

            i_max = counts.argmax()
            assert counts[i_max] >= min_support
            largest = clusters[i_max]
            support_tmp = support_tmp[clustering == largest]
            if verbose >= 1:
                print('Kept largest cluster from plane %i, [%s] supported by %i points.'
                      % (label, ', '.join('%.3f' % x for x in plane), len(support_tmp)))

        support = indices[support_tmp]
        planes.append((plane, support))
        # if isinstance(x, np.ndarray):
        #     x = torch.as_tensor(x)
        # plane = Plane(plane, x, support)
        # planes.append(plane)
        labels[support] = label

        if visualize_progress:
            # plane.visualize()

            # segmented = labels >= 0
            # colors[segmented] = map_colors(labels[segmented], min_value=0, max_value=label)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pcd], window_name='Plane Segmentation')
            # obj = Planes(torch.as_tensor(list(zip(planes))[0], len(planes) * [x], list(zip(planes))[1]))
            obj = Planes(torch.as_tensor([p for p, _ in planes]),
                         cloud=len(planes) * [x],
                         indices=[i for _, i in planes])
            obj.visualize()

        if len(planes) == num_planes:
            if verbose >= 1:
                print('Target number of planes found.')
            break

        keep = np.ones(len(indices), dtype=bool)
        keep[support_tmp] = 0
        indices = indices[keep]
        remaining_pcd = remaining_pcd.select_by_index(support_tmp, invert=True)
        if len(indices) < min_support:
            if verbose >= 1:
                print('Not enough points to continue.')
            break
        label += 1

    print('%i planes (highest label %i) with minimum support of %i points were found.'
          % (len(planes), labels.max(), min_support))

    planes = Planes(torch.as_tensor([p for p, _ in planes]), len(planes) * [x], [i for _, i in planes])

    # if visualize_final:
        # label = labels.max()
        # segmented = labels >= 0
        # colors[segmented] = map_colors(labels[segmented], min_value=0, max_value=label)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd], window_name='Plane Segmentation')
        # obj = Planes(torch.as_tensor([p for p, _ in planes]),
        #              len(planes) * [x],
        #              [i for _, i in planes])
        # obj.visualize()
        # planes.visualize()

    return planes


def fit_planes(x, distance_threshold, min_support=3, num_planes=int(1e9), eps=None,
               visualize_progress=False, visualize_final=False, verbose=0):
    """Segment points into planes."""
    # TODO: Move outer loop from implementation specific part here.
    if isinstance(x, DepthCloud):
        x = x.to_points().numpy()
    assert isinstance(x, np.ndarray)
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])

    planes = fit_planes_open3d(x, distance_threshold, min_support=min_support, num_planes=num_planes, eps=eps,
                               visualize_progress=visualize_progress, visualize_final=visualize_final, verbose=verbose)

    if visualize_final:
        planes.visualize()

    return planes


def main():
    cfg = Config()
    # cfg.train_names = ['asl_laser/apartment']
    # cfg.train_names = ['asl_laser/eth']
    # cfg.train_names = ['asl_laser/gazebo_summer']
    # cfg.train_names = ['asl_laser/gazebo_winter']
    cfg.train_names = ['asl_laser/stairs']

    cfg.val_names = []
    cfg.test_names = []
    cfg.data_step = 2
    cfg.grid_res = 0.1
    cfg.from_args()
    dist_thresh = 0.03
    for name in cfg.train_names + cfg.val_names + cfg.test_names:
        ds = create_dataset(name, cfg=cfg)

        # Segment local point clouds.
        # for cloud, pose in ds:
        #     planes = fit_planes(cloud, dist_thresh, min_support=100, num_planes=1000, eps=2*np.sqrt(3.0)*cfg.grid_res,
        #                         visualize_progress=False, visualize_final=True, verbose=1)

        # Segment global cloud.
        cloud = np.concatenate([transform(pose, cloud) for cloud, pose in ds])
        planes = fit_planes(cloud, dist_thresh, min_support=1000, num_planes=10, eps=np.sqrt(3.0)*cfg.grid_res,
                            visualize_progress=False, visualize_final=True, verbose=1)


if __name__ == '__main__':
    main()
