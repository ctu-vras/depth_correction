import os
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from os.path import dirname, join, realpath
from scipy.spatial.transform.rotation import Rotation as R
if not hasattr(R, 'as_matrix'):  # scipy < 1.4.0
    R.as_matrix = R.as_dcm
import re
import yaml
from copy import copy
import matplotlib.pyplot as plt

__all__ = [
    'data_dir',
    'dataset_names',
    'Dataset',
    'prefix',
]

prefix = 'newer_college'
data_dir = realpath(join(dirname(__file__), '..', '..', '..', 'data', prefix))

sequence_names = [
    '01_short_experiment',
]


def read_points(path, dtype=np.float32, beam_origin_correction=True):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    assert points.shape[1] == 3
    vps = np.zeros_like(points)
    if beam_origin_correction:
        # lidar_origin_to_beam_origin_mm:
        # https://data.ouster.io/downloads/software-user-manual/software-user-manual-v2p0.pdf
        vps_offset = 0.015806
        dp = np.linalg.norm(points[:, :2], axis=1)
        vps[:, :2] = points[:, :2] * vps_offset / (dp.reshape([-1, 1]) + 1e-6)
    points = np.hstack([points, vps])
    points = unstructured_to_structured(points.astype(dtype=dtype), names=['x', 'y', 'z', 'vp_x', 'vp_y', 'vp_z'])
    del pcd
    return points


def read_poses(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    sec = data[:, 0].astype(int)
    nsec = data[:, 1].astype(int)
    timestamps = [tuple(t) for t in zip(sec, nsec)]
    # Allow having only a subset of poses.
    xyz = data[:, 2:5]
    qxyzw = data[:, 5:]
    poses = np.zeros((len(xyz), 4, 4))
    poses[:, :3, :3] = R.from_quat(qxyzw).as_matrix()
    poses[:, :3, 3] = xyz
    poses[:, 3, 3] = 1.0
    return timestamps, poses


def read_calibration():
    calibration = {}
    # https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python
    path_lidar_calib = os.path.join(data_dir, '2020-ouster-os1-64-realsense/04_calibration/kalibr_output'
                                              '/ouster_imu_lidar_transforms.yaml')
    with open(path_lidar_calib, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            t = data['os1_lidar_to_os1_imu']['translation']
            q_xyzw = data['os1_lidar_to_os1_imu']['rotation']
            T_lidar2lidar_imu = np.eye(4)
            T_lidar2lidar_imu[:3, :3] = R.from_quat(q_xyzw).as_matrix()
            T_lidar2lidar_imu[:3, 3] = t
        except yaml.YAMLError as exc:
            print(exc)
            return None
    path_cam_calib = os.path.join(data_dir, '2020-ouster-os1-64-realsense/04_calibration/kalibr_output/cam-ouster-imu'
                                            '/camchain-ouster_imu-cam-rooster_2020-03-11-10-05-35_0.yaml')
    with open(path_cam_calib, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            T_lidar_imu2cam_left = np.array(data['cam0']['T_cam_imu'])
            T_lidar_imu2cam_right = np.array(data['cam1']['T_cam_imu'])
        except yaml.YAMLError as exc:
            print(exc)
            return None

    calibration['T_lidar_imu2cam_left'] = T_lidar_imu2cam_left
    calibration['T_lidar_imu2cam_right'] = T_lidar_imu2cam_right
    calibration['T_lidar2lidar_imu'] = T_lidar2lidar_imu

    T_cam_opt2cam = np.array([[ 0.,  0., 1., 0.],
                              [-1.,  0., 0., 0.],
                              [ 0., -1., 0., 0.],
                              [ 0.,  0., 0., 1.]])
    calibration['T_cam_opt2cam'] = T_cam_opt2cam
    calibration['T_cam2cam_opt'] = np.linalg.inv(T_cam_opt2cam)

    T_lidar2cam = T_cam_opt2cam @ T_lidar_imu2cam_left @ T_lidar2lidar_imu

    # T_lidar2cam = np.eye(4)
    # T_lidar2cam[:3, :3] = R.from_quat([0.012, -0.002, 0.923, 0.385]).as_matrix()
    # T_lidar2cam[:3, 3] = np.array([-0.071, -0.003, 0.048])

    # T_lidar2cam = np.eye(4)
    # T_lidar2cam[:3, :3] = R.from_quat([0.000, 0.000, 0.924, 0.383]).as_matrix()
    # T_lidar2cam[:3, 3] = np.array([-0.084, -0.025, 0.050])

    calibration['T_lidar2cam'] = T_lidar2cam

    return calibration


class NewerCollege(object):
    default_poses_csv = 'registered_poses.csv'

    def __init__(self, name='01_short_experiment', path=None, poses_csv=default_poses_csv, poses_path=None,
                 zero_origin=False):
        """Newer College dataset: https://ori-drs.github.io/newer-college-dataset/.

        ├─ <NHCD_ROOT>    # Root of the data
            ├─2020-ouster-os1-64-realsense
                ├─ 01_short_experiment              # Date of acquisition of the sequence
                    └─ ouster_scan                  # The folder containing the ouster scans
                        ├─ cloud_1583836591_182590976.pcd
                            ...
                    └─ ground_truth                 # The sequence ground truth files (in the reference frame of the left camera)
                        ├─ registered_poses.csv

        :param name: Dataset name.
        :param path: Dataset path, takes precedence over name.
        :param poses_csv: Poses CSV file name.
        :param poses_path: Override for poses CSV path.
        """
        assert isinstance(name, str)
        if name.startswith(prefix):
            name = name[len(prefix) + 1:]
        parts = name.split('/')
        name = parts[0]
        if path is None:
            path = join(data_dir, '2020-ouster-os1-64-realsense', name)

        if not poses_csv:
            poses_csv = Dataset.default_poses_csv

        self.name = name
        self.details = parts[1] if len(parts) > 1 else ''
        self.path = path
        self.poses_path = poses_path
        self.poses_csv = poses_csv
        self.calibration = read_calibration()

        if self.poses_path or self.path:
            self.ids = self.read_available_timestamps()
            timestamps, self.poses = read_poses(self.cloud_poses_path())
            self.id_to_pose_index = dict(zip(timestamps, range(len(timestamps))))
            self.poses = self.transform_poses(self.poses, zero_origin=zero_origin)
        else:
            self.ids = None
            self.poses = None

    def read_available_timestamps(self):
        """Read timestamps of available point clouds."""
        ts = []
        for path in os.listdir(os.path.join(self.path, 'ouster_scan')):
            s, ns = path[6:-4].split('_')
            t = int(s), int(ns)
            ts.append(t)
        ts = sorted(ts)
        return ts

    def local_cloud_path(self, id):
        return os.path.join(self.path, 'ouster_scan', 'cloud_%i_%09i.pcd' % id)

    def cloud_poses_path(self):
        if self.poses_path:
            return self.poses_path
        return os.path.join(self.path, 'ground_truth', self.poses_csv)

    def transform_poses(self, poses, zero_origin=False):
        assert isinstance(poses, np.ndarray)
        assert poses.shape[1:] == (4, 4)
        # transform between camera and lidar
        poses = np.matmul(poses, self.calibration['T_lidar2cam'])
        # transform to 0-origin (pose[0] = np.eye(4))
        if zero_origin:
            poses = np.matmul(np.linalg.inv(poses[0]), poses)
        return poses

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            return self.local_cloud(id), self.cloud_pose(id)

        ds = copy(self)
        if isinstance(item, (list, tuple)):
            ds.ids = [self.ids[i] for i in item]
        else:
            assert isinstance(item, slice)
            ds.ids = self.ids[item]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def local_cloud(self, id):
        return read_points(self.local_cloud_path(id))

    def cloud_pose(self, id):
        return self.poses[self.id_to_pose_index[id]]

    def get_ground_truth_cloud(self, voxel_size=0.1):
        import open3d as o3d
        path_gt = os.path.join(self.path.replace(self.name, '03_new_college_prior_map'),
                               'new-college-29-01-2020-1cm-resolution-libpmfiltered.ply')
        pcd_gt = o3d.io.read_point_cloud(path_gt)
        # transform to 0-origin
        pose0 = read_poses(self.cloud_poses_path())[1][0]
        pose0 = self.transform_poses(pose0[None], zero_origin=False)[0]
        pose0_inv = np.linalg.inv(pose0)
        cloud = np.matmul(np.asarray(pcd_gt.points), pose0_inv[:3, :3].T) + pose0_inv[:3, 3:].T
        pcd_gt.points = o3d.utility.Vector3dVector(cloud)
        if voxel_size:
            pcd_gt.voxel_down_sample(voxel_size=voxel_size)
        return pcd_gt

    def __str__(self):
        if self.details:
            return '%s/%s/%s' % (prefix, self.name, self.details)
        return '%s/%s' % (prefix, self.name)

    def show_global_cloud(self, data_step=5):
        import open3d as o3d
        clouds = []
        for id in self.ids[::data_step]:
            cloud, pose = self.local_cloud(id), self.cloud_pose(id)
            cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
            cloud = np.matmul(cloud, pose[:3, :3].T) + pose[:3, 3:].T
            clouds.append(cloud)
        cloud = np.concatenate(clouds)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.2)])
        del pcd
        return cloud

    def show_path(self, title=None):
        fig, axes = plt.subplots(1, 1, figsize=(8.0, 8.0), constrained_layout=True, squeeze=False)
        ax = axes[0, 0]
        pose_ids = [self.id_to_pose_index[id] for id in self.ids]
        ax.plot(self.poses[pose_ids, 0, 3],
                self.poses[pose_ids, 1, 3], '.')
        ax.set_aspect('equal')
        ax.grid()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        if title:
            ax.set_title(title)
        plt.show()


# np.random.seed(135)
# dataset_names = []
# size_seq = 500
# n_subseq = 8
# step = 12
# quad_mid_n = 2400
# assert n_subseq % 4 == 0  # for train (1/2), val (1/4), test (1/4) split
# ds = NewerCollege()
# for seq in range(n_subseq):
#     # start = np.random.choice(range(0, min(quad_mid_n, len(ds)) - size_seq))
#     start = np.random.choice(range(0, len(ds) - size_seq))
#     dataset_names.append("01_short_experiment/start_%d_end_%d_step_%d" % (start, start + size_seq, step))
# dataset_names = sorted(dataset_names)
# for d in dataset_names:
#     print("'%s'," % d)

# subsequences without Parkland area (include only Quad and Mid-Section)
dataset_names = [
    '01_short_experiment/start_0_end_800_step_12',  # quad
    '01_short_experiment/start_800_end_1600_step_12',
    '01_short_experiment/start_1600_end_2400_step_12',  # mid
    '01_short_experiment/start_7000_end_7800_step_12',  # mid-quad
    '01_short_experiment/start_7800_end_8600_step_12',
    '01_short_experiment/start_8600_end_9500_step_12',  # quad-mid
    '01_short_experiment/start_13900_end_14600_step_12',  # mid-quad
    '01_short_experiment/start_14601_end_15301_step_12',  # mid-quad
]


class Dataset(NewerCollege):
    def __init__(self, name='01_short_experiment', path=None, poses_csv=None, poses_path=None, zero_origin=True):
        """Newer College part of the dataset.

        :param name: Dataset name in format NN_start_SS_end_EE_step_ss
        :param path: Dataset path, takes precedence over name.
        :param poses_csv: Poses CSV file name.
        :param poses_path: Override for poses path.
        """
        super(Dataset, self).__init__(name=name, poses_path=poses_path, zero_origin=zero_origin)

        # Use slice specification from name.
        if self.details:
            start = re.search('start_(\d+)', self.details)
            end = re.search('end_(\d+)', self.details)
            step = re.search('step_(\d+)', self.details)
            start = int(start.group(1)) if start else None
            end = int(end.group(1)) if end else None
            step = int(step.group(1)) if step else None
            sub_seq = slice(start, end, step)
            self.ids = self.ids[sub_seq]

        # move poses to origin to 0:
        if zero_origin:
            Tr_inv = np.linalg.inv(self.cloud_pose(self.ids[0]))
            self.poses = np.matmul(Tr_inv, self.poses)


def view_data():
    import matplotlib.pyplot as plt
    import open3d as o3d
    from tqdm import tqdm

    ds = Dataset(name='01_short_experiment/step_1')

    plt.figure(figsize=(10, 10))
    plt.ylabel('Trajectory: XY, [m]')
    plt.axis('equal')
    plt.plot(ds.poses[:, 0, 3], ds.poses[:, 1, 3], '.')
    plt.grid()
    plt.show()

    clouds = []
    for k, data in tqdm(enumerate(ds[::500])):
        cloud, pose = data
        cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
        cloud = np.matmul(cloud, pose[:3, :3].T) + pose[:3, 3:].T

        clouds.append(cloud)

    cloud = np.concatenate(clouds)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.2)])

    # load ground truth cloud
    pcd_gt = ds.get_ground_truth_cloud()
    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.5).paint_uniform_color([0, 0.651, 0.929]),
                                       pcd_gt.voxel_down_sample(voxel_size=0.5).paint_uniform_color([1, 0.706, 0])])
    # compute Chamfer loss
    d1 = pcd.compute_point_cloud_distance(pcd_gt)
    d2 = pcd_gt.compute_point_cloud_distance(pcd)
    d = np.mean(d1) + np.mean(d2)
    print('Chamfer distance: %f' % d)
    del pcd_gt


def demo():
    from depth_correction.dataset import RenderedMeshDataset

    mesh_path = os.path.join(data_dir,
                             '2020-ouster-os1-64-realsense',
                             '03_new_college_prior_map',
                             'mesh.ply')

    for name in ['start_0_end_800_step_12',  # quad
                 'start_800_end_1600_step_12',  # quad-mid
                 'start_14601_end_15301_step_12',  # quad
                 ]:
        # poses = np.stack([pose for _, pose in Dataset(name, zero_origin=False)])
        # nc = RenderedMeshDataset(name=mesh_path, poses=poses, device='cuda')

        nc = RenderedMeshDataset(name=mesh_path, poses_path='newer_college/%s/ouster_lidar_poses.txt' % name)
        cloud = nc.show_global_cloud(data_step=5)

        plt.figure(figsize=(10, 10))
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.plot(cloud[::10, 0], cloud[::10, 1], '.')
        plt.plot(nc.poses[::1, 0, 3], nc.poses[::1, 1, 3], 'o')
        plt.grid()
        plt.show()


def debug():
    import torch
    from depth_correction.preproc import filtered_cloud, local_feature_cloud, global_cloud
    from depth_correction.dataset import create_dataset
    from depth_correction.config import Config
    from depth_correction.utils import map_colors
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import open3d as o3d

    cfg = Config()
    cfg.data_step = 1
    cfg.min_depth = 1.0
    cfg.max_depth = 15.0
    cfg.grid_res = 0.2
    cfg.nn_k = 0
    cfg.nn_r = 0.4
    cfg.shadow_angle_bounds = []
    cfg.log_filters = True

    names = ['newer_college/01_short_experiment/start_1100_end_1150_step_5',
             'asl_laser/eth_step_4',
             'semantic_kitti/04_start_100_end_120_step_5']
    for name in names:
        ds = create_dataset(name=name, cfg=cfg)

        poses = []
        clouds = []
        for i, (cloud, pose) in enumerate(ds):
            cloud = filtered_cloud(cloud, cfg)
            cloud = local_feature_cloud(cloud, cfg)
            clouds.append(cloud)
            pose = torch.as_tensor(pose)
            poses.append(pose)
        cloud = global_cloud(clouds, None, poses)
        cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)

        poses = torch.stack(poses)
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.title('Viewpoints location %s' % name)
        plt.axis('equal')
        plt.plot(poses[:, 0, 3], poses[:, 1, 3], '.')
        plt.grid()

        plt.subplot(1, 2, 2)
        # https://stackoverflow.com/questions/33203645/how-to-plot-a-histogram-using-matplotlib-in-python-with-a-list-of-data
        angles = cloud.inc_angles.squeeze().numpy()
        q25, q75 = np.percentile(angles, [25, 75])
        bin_width = 2 * (q75 - q25) * len(angles) ** (-1 / 3)
        bins = round((angles.max() - angles.min()) / bin_width)
        plt.hist(angles, density=False, bins=bins)  # density=False would make counts
        plt.ylabel('Number of incidence angles')
        plt.xlabel('Angles')
        plt.grid()

        plt.savefig('/home/ruslan/Desktop/%s_%s.png' % (name.split('/')[0], name.split('/')[1]))
        plt.show()

        # pcd = o3d.geometry.PointCloud()
        # colors = map_colors(cloud.inc_angles, colormap=cm.viridis, min_value=np.deg2rad(60), max_value=np.deg2rad(90))
        # pcd.points = o3d.utility.Vector3dVector(cloud.points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.5)])


def main():
    # view_data()
    demo()
    # debug()


if __name__ == '__main__':
    main()
