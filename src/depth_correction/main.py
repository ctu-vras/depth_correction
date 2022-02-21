from __future__ import absolute_import, division, print_function
from .config import Config
from .eval import eval_loss
from .model import *
from .train import train
from .loss import min_eigval_loss
from collections import deque
from data.asl_laser import Dataset, dataset_names
import os
import random
import roslaunch
import roslaunch.parent
import roslaunch.rlutil
from rospkg import RosPack
import rospy


package_dir = RosPack().get_path('depth_correction')
slam_eval_launch = os.path.join(package_dir, 'launch', 'slam_eval.launch')

# TODO: Generate multiple splits.
ds = ['asl_laser/%s' % name for name in dataset_names]
num_splits = 4
shift = len(ds) // num_splits
splits = []

random.seed(135)
random.shuffle(ds)
ds_deque = deque(ds)
for i in range(num_splits):
    # random.shuffle(datasets)
    ds_deque.rotate(shift)
    # copy = list(datasets)
    # random.shuffle(copy)
    # splits.append([copy[:4], copy[4:6], copy[6:]])
    ds_list = list(ds_deque)
    splits.append([ds_list[:4], ds_list[4:6], ds_list[6:]])
for split in splits:
    print(split)

# splits = [[['asl_laser/eth'], ['asl_laser/stairs'], ['asl_laser/gazebo_winter']]]
splits = [[['eth'], ['stairs'], ['gazebo_winter']]]

# models = [Polynomial, ScaledPolynomial]
# models = [ScaledPolynomial]
models = ['ScaledPolynomial']
losses = ['min_eigval_loss']
slams = ['ethzasl_icp_mapper']


# def fit_model(cfg: Config, model, train, val, loss, correct_poses=False):
#     cfg = Config()
#     cfg.train_names = train
#     cfg.model_class = model
#     model = train(cfg)
#     return model


# def eval_loss(cfg: Config=None):
#     pass


# def eval_slam(model: (BaseModel, dict), split: list, slam: str):
def eval_slam(model: (BaseModel, dict)=None, split: list=None, slam: str=None, cfg: Config=None):

    if cfg is not None:
        # model = load_model(cfg=cfg)
        if split is None:
            split = cfg.test_names
        if slam is None:
            slam = cfg.slam

    # # TODO: Store weights into a temp file.
    # if isinstance(model, BaseModel):
    #     pass
    #     # torch.save(model.state_dict(),
    #     #            '%s/config/weights/%s_train_%s_val_%s_r%.2f_eig_%.4f_%.4f_min_eigval_loss_%.9f.pth'
    #     #            % (pkg_dir, MODEL_TYPE, ','.join(train_names), ','.join(val_names),
    #     #               r, max_eig_0, min_eig_1, val_loss.item()))
    #     model_class = 'BaseModel'
    #     model_state_dict = ''
    # elif isinstance(model, dict):
    #     model_class = model['model_state_dict']
    #     model_state_dict = model['model_state_dict']
    # else:
    #     raise ValueError('Unsupported model type.')

    # TODO: Actually use slam id if multiple slam pipelines are to be tested.
    assert slam == 'ethzasl_icp_mapper'

    # TODO: Run slam for each sequence in split.
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    for seq in split:
        print('SLAM evaluation on %s started.' % seq)
        # cli_args = [slam_eval_launch, 'dataset:=%s' % seq, 'odom:=true', 'depth_correction:=true', 'rviz:=true',
        #             'model_class:=%s' % model_class, 'model_state_dict:=%s' % model_state_dict]
        cli_args = [slam_eval_launch, 'dataset:=%s' % seq, 'odom:=true', 'depth_correction:=true', 'rviz:=true',
                    'model_class:=%s' % cfg.model_class, 'model_state_dict:=%s' % cfg.model_state_dict]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
        parent.start()
        parent.spin()
        print('SLAM evaluation on %s finished.' % seq)


def run_model_from_ground_truth():
    base_cfg = Config()
    base_cfg.data_step = 10
    base_cfg.n_opt_iters = 10
    for model in models:
        for train_names, val_names, test_names in splits:  # train, val, test dataset split (cross validation)
            for loss in losses:
                # learn depth correction model using ground truth poses using the loss
                #     get best model from validation set using the loss
                # model = fit_model(model, train, val, loss)
                # cfg = Config()
                # TODO: Set log dir
                cfg = base_cfg.copy()
                cfg.model_class = model
                cfg.train_names = train_names
                cfg.val_names = val_names
                cfg.test_names = test_names
                cfg.loss = loss
                best_cfg = train(cfg)
                # evaluate consistency on test (train, validation) set
                # for test_loss in losses:
                for split in [train_names, val_names, test_names]:
                    eval_cfg = best_cfg.copy()
                    eval_cfg.test_names = split
                    # eval_loss(model, loss, split)
                    eval_loss(eval_cfg)
                    for slam in slams:
                        eval_cfg.slam = slam
                        # evaluate slam localization on test (train, validation) set
                        # eval_slam(model, split, slam)
                        eval_slam(eval_cfg)


def run_model_from_slam():
    for model in models:
        for train_names, val, test in splits:  # train, val, test dataset split (cross validation)
            for loss in losses:
                for slam in slams:
                    # generate slam poses for the splits
                    train_slam, val_slam, test_slam = [eval_slam(None, split, slam) for split in [train, val, test]]
                    # learn depth correction model using poses from slam
                    #     get best model from validation set
                    # train_slam = fit_model_slam(model, train, val, loss, slam)
                    cfg = Config()
                    cfg.train_names = train
                    # model = fit_model(model, train_slam, val_slam, loss, correct_poses=True)
                    # evaluate consistency on test (train, validation) set
                    # for test_loss in losses:
                    for split in [train, val, test]:
                        eval_loss(model, loss, split)
                        for slam in slams:
                            # evaluate slam localization on test (train, validation) set
                            eval_slam(model, split, slam)


def run_calibration():
    pass


def run_experiments():
    # eval_slam(BaseModel(), splits[0][0], slams[0])
    # return
    run_model_from_ground_truth()
    run_model_from_slam()
    run_calibration()


def main():
    run_experiments()


if __name__ == '__main__':
    main()
