from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from argparse import ArgumentParser
import os
import roslaunch
from rospkg import RosPack


package_dir = RosPack().get_path('depth_correction')
slam_eval_launch = os.path.join(package_dir, 'launch', 'slam_eval.launch')


def eval_slam(cfg: Config):
    cfg_path = os.path.join(cfg.log_dir, 'eval_slam.yaml')
    if os.path.exists(cfg_path):
        print('Config %s already exists.' % cfg_path)
    else:
        cfg.to_yaml(cfg_path)

    # TODO: Actually use SLAM id if multiple pipelines are to be tested.
    assert cfg.slam == 'ethzasl_icp_mapper'
    # csv = os.path.join(cfg.log_dir, 'slam_eval.csv')
    assert cfg.slam_eval_csv
    assert not cfg.slam_poses_csv or len(cfg.test_names) == 1
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    for i, name in enumerate(cfg.test_names):
        # Allow overriding poses paths, assume valid if non-empty.
        poses_path = cfg.test_poses_path[i] if cfg.test_poses_path else None
        print('SLAM evaluation on %s started.' % name)
        # print(cfg.to_yaml())

        cli_args = [slam_eval_launch]
        cli_args.append('dataset:=%s' % name)
        if poses_path:
            cli_args.append('dataset_poses_path:=%s' % poses_path)
        cli_args.append('odom:=true')
        cli_args.append('rviz:=true' if cfg.rviz else 'rviz:=false')
        cli_args.append('slam_eval_csv:=%s' % cfg.slam_eval_csv)
        if cfg.slam_poses_csv:
            cli_args.append('slam_poses_csv:=%s' % cfg.slam_poses_csv)
        cli_args.append('min_depth:=%.1f' % cfg.min_depth)
        cli_args.append('max_depth:=%.1f' % cfg.max_depth)
        cli_args.append('grid_res:=%.1f' % cfg.grid_res)
        if cfg.pose_correction != PoseCorrection.none and cfg.model_class != 'BaseModel':
            cli_args.append('depth_correction:=true')
        else:
            cli_args.append('depth_correction:=false')
        cli_args.append('nn_r:=%.2f' % cfg.nn_r)
        # TODO: Pass eigenvalue bounds to launch.
        cli_args.append('eigenvalue_bounds:=[[0, -.inf, 0.0004], [1, 0.0025, .inf]]')
        cli_args.append('model_class:=%s' % cfg.model_class)
        cli_args.append('model_state_dict:=%s' % cfg.model_state_dict)
        # print(cli_args)
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, force_log=True)
        # parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, verbose=True, force_log=True)
        parent.start()
        parent.spin()
        print('SLAM evaluation on %s finished.' % name)


def run_from_cmdline():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()
    print('Arguments:')
    print(args)
    cfg = Config()
    cfg.from_yaml(args.config)
    print('Config:')
    print(cfg.to_yaml())
    print('Evaluating SLAM...')
    eval_slam(cfg)
    print('Evaluating SLAM finished.')


def main():
    run_from_cmdline()


if __name__ == '__main__':
    main()
