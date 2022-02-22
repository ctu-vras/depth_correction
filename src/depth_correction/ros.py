from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .transform import matrix_to_xyz_axis_angle, xyz_axis_angle_to_path_msg
from .utils import timing
import torch
import rospy

__all__ = [
    'clear_publishers',
    'initialize_ros',
    'publish',
    'publish_data',
]

_ros_initialized = False
_pubs = {}


def initialize_ros():
    global _ros_initialized
    if not _ros_initialized:
        rospy.init_node('depth_correction', log_level=rospy.INFO)
        _ros_initialized = True


def publish(topic, msg, latch=True):
    initialize_ros()
    if topic not in _pubs:
        pub = rospy.Publisher(topic, msg.__class__, latch=latch, queue_size=2)
        print('Publisher at %s [%s] created.' % (topic, msg.__class__.__name__))
        _pubs[topic] = pub

    _pubs[topic].publish(msg)


def clear_publishers():
    for pub in _pubs.values():
        pub.unregister()

    _pubs.clear()


@timing
def publish_data(clouds: list, paths: list, names: list, cfg: Config):
    assert isinstance(clouds[0], DepthCloud)
    assert isinstance(paths[0], torch.Tensor)

    stamp = rospy.Time.now()
    for name, path, cloud in zip(names, paths, clouds):
        pc_opt_msg = cloud.to_msg(frame_id=cfg.world_frame, stamp=stamp)
        path_opt_msg = xyz_axis_angle_to_path_msg(matrix_to_xyz_axis_angle(path),
                                                  frame_id=cfg.world_frame, stamp=stamp)
        publish('%s/path' % name, path_opt_msg)
        publish('%s/global_cloud' % name, pc_opt_msg)
