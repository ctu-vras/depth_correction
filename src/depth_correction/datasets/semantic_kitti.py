import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from os.path import dirname, join, normpath, realpath
import re
import torch

__all__ = [
    'data_dir',
    'dataset_names',
    'Dataset',
]

prefix = 'semantic_kitti'
data_dir = normpath(join(dirname(__file__), '..', '..', '..', 'data', prefix))

sequence_names = [
    '%02d' % i for i in range(22)
]

def read_poses(path):
    poses = np.genfromtxt(path, delimiter=', ', skip_header=1)
    ids = poses[:, 0].astype(int).tolist()
    # Allow having only a subset of poses.
    # assert ids == list(range(len(ids)))
    poses = poses[:, 2:]
    poses = poses.reshape((-1, 4, 4))
    # poses = dict(zip(ids, poses))
    poses = list(poses)
    return ids, poses
    # return dict(zip(ids, poses))


class Sequence:
    def __init__(self, seq, path=None, poses_path=None, pose_provider='odom'):
        assert isinstance(seq, (int, str))
        if isinstance(seq, str):
            parts = seq.split('/')
            assert 1 <= len(parts) <= 2
            if len(parts) == 2:
                assert parts[0] == prefix
                seq = parts[1]
        if path is None:
            path = join(data_dir, 'sequences')
        self.path = path
        self.poses_path = poses_path
        self.calibrations = None
        self.times = None
        self.poses = None
        if isinstance(seq, int):
            assert seq in range(22)
            seq = "%02d" % seq
        self.sequence = seq[:2]

        self.load_calib()
        self.ids, self.poses, self.times = self.load_poses(pose_provider=pose_provider)

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return len(self.ids)

    def __str__(self):
        return prefix + '/' + self.sequence

    def load_calib(self):
        """
        load calib poses and times.
        """
        seq_folder = join(self.path, str(self.sequence).zfill(2))
        # Read Calib
        self.calibrations = self.parse_calibration(join(seq_folder, "calib.txt"))

    def load_poses(self, pose_provider):
        assert pose_provider == 'odom' or pose_provider == 'surf_slam'
        seq_folder = join(self.path, str(self.sequence).zfill(2))
        # Read times
        times = np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32)
        ids = list(range(len(times)))

        # Read poses
        if self.poses_path:
            tmp_ids, tmp_poses = read_poses(self.poses_path)
            assert max(tmp_ids) <= max(ids)
            poses = [None] * len(ids)
            for id, pose in zip(tmp_ids, tmp_poses):
                poses[id] = pose
        else:
            if pose_provider == 'surf_slam':
                poses = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations)
            elif pose_provider == 'odom':
                poses = self.parse_poses(normpath(join(self.path, '..', 'poses', "%s.txt" % self.sequence)),
                                         self.calibrations)
            assert len(ids) == len(poses)
        return ids, poses, times

    @staticmethod
    def parse_calibration(filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose
        calib_file.close()
        calib['Tr_cam2_to_velo'] = np.array([[2.34773698e-04, -9.99944155e-01, -1.05634778e-02, 5.93721868e-02],
                                             [1.04494074e-02,  1.05653536e-02, -9.99889574e-01, -7.51087914e-02],
                                             [9.99945389e-01,  1.24365378e-04,  1.04513030e-02, -2.72132796e-01],
                                             [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
        return calib

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)
        poses = []
        # Tr = calibration["Tr"]
        # Tr_inv = np.linalg.inv(Tr)
        Tr_cam2_to_velo = calibration['Tr_cam2_to_velo']
        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            # poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
            poses.append(np.matmul(pose, Tr_cam2_to_velo))
        return poses

    def local_cloud(self, i, dtype=np.float32):
        cloud_path = join(self.path, self.sequence, 'velodyne/%06d.bin' % i)
        # Read 32-bit floats from a binary file.
        cloud = np.fromfile(cloud_path, dtype=np.float32).reshape((-1, 4))
        cloud = cloud[:, :3]  # Keep only xyz, drop intensity.
        cloud = unstructured_to_structured(cloud.astype(dtype=dtype), names=['x', 'y', 'z'])
        return cloud

    def cloud_pose(self, i, dtype=np.float64):
        pose = self.poses[i].astype(dtype=dtype)
        return pose

    @staticmethod
    def transform(cloud, pose):
        cloud = np.matmul(cloud, pose[:3, :3].T) + pose[:3, 3:].T
        return cloud

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            cloud = self.local_cloud(id)
            pose = self.cloud_pose(id)
            return cloud, pose
        ds = Sequence(seq=self.sequence)
        ds.sequence = self.sequence
        ds.path = self.path
        ds.poses = self.poses
        ds.calibrations = self.calibrations
        ds.times = self.times
        if isinstance(item, (list, tuple)):
            ds.ids = [self.ids[i] for i in item]
        else:
            assert isinstance(item, slice)
            ds.ids = self.ids[item]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class Dataset(Sequence):
    def __init__(self, name, path=None, poses_csv=None, poses_path=None, pose_provider='surf_slam'):
        """Semantic KITTI dataset or a dataset in that format.

        :param name: Dataset name in format NN_start_SS_end_EE_step_ss
        :param path: Dataset path, takes precedence over name.
        :param poses_csv: Poses CSV file name.
        :param poses_path: Override for poses path.
        :param pose_provider: Poses either from Semantic KITTI (surfel based slam) or KITTI odometry (GPS/IMU).
        """
        super(Dataset, self).__init__(seq=name, poses_path=poses_path, pose_provider=pose_provider)
        assert isinstance(name, str)
        parts = name.split('/')
        assert 1 <= len(parts) <= 2
        if len(parts) == 2:
            assert parts[0] == prefix
            name = parts[1]
        self.name = name

        # Use slice specification from name.
        start = re.search('start_(\d+)', name)
        end = re.search('end_(\d+)', name)
        step = re.search('step_(\d+)', name)
        start = int(start.group(1)) if start else None
        end = int(end.group(1)) if end else None
        step = int(step.group(1)) if step else None
        sub_seq = slice(start, end, step)
        self.ids = self.ids[sub_seq]

        # move poses to origin to 0:
        Tr_inv = np.linalg.inv(self.cloud_pose(self.ids[0]))
        self.poses = [(np.matmul(Tr_inv, pose) if pose is not None else None) for pose in self.poses]


# np.random.seed(135)
# dataset_names = []
# size_seq = 50
# n_subseq = 8
# step = 1
# assert n_subseq % 4 == 0  # for train (1/2), val (1/4), test (1/4) split
# # for seq in np.random.choice(range(0, 22), n_subseq):
# for seq in np.random.choice(range(0, 11), n_subseq, replace=False):
#     s = Sequence(int(seq))
#     start = np.random.choice(range(0, min(len(s), 500) - size_seq))
#     # '11_start_264_end_314_step_1'
#     dataset_names.append("%02d_start_%d_end_%d_step_%d" % (seq, start, start + size_seq, step))
# dataset_names = sorted(dataset_names)

dataset_names = [
    '09_start_311_end_361_step_1',
    '10_start_223_end_273_step_1',
    '00_start_127_end_177_step_1',
    '03_start_366_end_416_step_1',
    '01_start_221_end_271_step_1',
    '04_start_72_end_122_step_1',
    '02_start_264_end_314_step_1',
    '07_start_28_end_78_step_1',
]
# print('Subsequences:')
# for ds_name in dataset_names:
#     print(ds_name)


def poses_demo():
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from pytorch3d.transforms import so3_relative_angle

    for seq in sequence_names[:11]:
        ds_odom = Dataset(name=seq, pose_provider='odom')
        ds_slam = Dataset(name=seq, pose_provider='surf_slam')
        assert len(ds_odom) == len(ds_slam)

        poses_odom = []
        for _, pose in tqdm(ds_odom[::]):
            poses_odom.append(pose)

        poses_slam = []
        for _, pose in tqdm(ds_slam[::]):
            poses_slam.append(pose)

        rot_diffs = []
        trans_diffs = []
        for p1, p2 in zip(poses_slam, poses_odom):
            R1 = torch.as_tensor(p1[:3, :3][None])
            R2 = torch.as_tensor(p2[:3, :3][None])
            rot_diff = so3_relative_angle(R1, R2)
            rot_diffs.append(rot_diff)
            trans_diffs.append(np.linalg.norm(p1[:3, 3] - p2[:3, 3]))

        poses_odom = np.asarray(poses_odom)
        poses_slam = np.asarray(poses_slam)

        plt.figure()
        plt.title('Trajectory for sequence: %s' % seq)
        plt.plot(poses_odom[:, 0, 3], poses_odom[:, 1, 3], '.', label='odom')
        plt.plot(poses_slam[:, 0, 3], poses_slam[:, 1, 3], '.', label='surf_slam')
        plt.grid()
        plt.legend()

        plt.figure()
        # Rotation difference for KITTI semantic and KITTI odometry poses
        plt.subplot(1, 2, 1)
        plt.plot(rot_diffs)
        plt.grid()
        plt.title('delta R (semantic, odom)')
        # Translation difference for KITTI semantic and KITTI odometry poses
        plt.subplot(1, 2, 2)
        plt.plot(trans_diffs)
        plt.grid()
        plt.title('delta t (semantic, odom)')

        plt.show()


def seq_demo():
    from tqdm import tqdm
    import open3d as o3d
    import matplotlib.pyplot as plt

    for seq in sequence_names[:11]:
        seq += '_step_10'
        ds = Dataset(name=seq)

        clouds = []
        poses = []
        for cloud, pose in tqdm(ds):
            cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
            cloud = ds.transform(cloud, pose)
            clouds.append(cloud)
            poses.append(pose)

        cloud = np.concatenate(clouds)
        poses = np.asarray(poses)

        plt.figure()
        plt.axis('equal')
        plt.title('Trajectory for sequence: %s' % seq)
        plt.plot(poses[:, 0, 3], poses[:, 1, 3], '.')
        plt.grid()
        plt.show()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[::10, :])
        o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.2)])


def subseq_demo():
    import open3d as o3d
    import matplotlib.pyplot as plt
    from pytorch3d.structures import Pointclouds
    from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

    subseq_pcds = []
    plt.figure()
    plt.grid()
    plt.axis('equal')
    for i, name in enumerate(dataset_names):
        ds = Dataset(name= prefix + '/' + name, pose_provider='odom')

        clouds = []
        poses = []
        dists = []
        pose_prev = np.eye(4)
        for id in ds.ids:
            cloud = ds.local_cloud(id)
            pose = ds.cloud_pose(id)
            cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
            cloud = ds.transform(cloud, pose)
            clouds.append(cloud)
            poses.append(pose)
            dists.append(np.linalg.norm(pose[:3, 3] - pose_prev[:3, 3]))
            pose_prev = pose

        cloud = np.concatenate(clouds)
        poses = np.asarray(poses)
        dists.pop(0)

        print('Subseq %s, N points: %d, length: %.2f, distances between poses: min=%.2f, mean=%.2f, max=%.2f'
              % (name, len(cloud), np.sum(dists), np.min(dists), np.mean(dists), np.max(dists)))

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(cloud)
    #     o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.5)])
    #
    #     plt.title('Trajectory for sequence: %s' % name)
    #     plt.plot(poses[:, 0, 3], poses[:, 1, 3], '.', label=name)
    #     plt.legend()
    # plt.show()

        pc_dist = 300
        offset = [pc_dist * i, 0, 0] if i <= len(dataset_names) // 2 - 1 else\
            [pc_dist * (i - len(dataset_names) // 2), -pc_dist, 0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[::50, :] + np.asarray(offset).reshape(([-1, 3])))
        subseq_pcds.append(pcd)
        # subseq_pcds.append(torch.as_tensor(cloud[::10, :] + np.asarray(offset).reshape(([-1, 3]))))  # for Pytorch3d
    # visualization with Open3d
    o3d.visualization.draw_geometries(subseq_pcds)

    # # visualization with Pytorch3d
    # point_cloud_batch = Pointclouds(points=subseq_pcds)
    # vis = {'Pointcloud': {}}
    # for i in range(len(dataset_names)):
    #     vis['Pointcloud'][dataset_names[i]] = point_cloud_batch[i]
    # fig = plot_scene(vis)
    # fig.show()


def main():
    # subseq_demo()
    seq_demo()
    # poses_demo()


if __name__ == '__main__':
    main()
