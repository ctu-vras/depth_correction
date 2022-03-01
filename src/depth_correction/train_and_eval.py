from __future__ import absolute_import, division, print_function
from .config import Config
from .eval import eval_loss
from .slam_eval import eval_slam
from .train import train
from argparse import ArgumentParser
import os


def train_and_eval(cfg: Config):
    best_cfg = train(cfg)
    # Evaluate consistency loss and SLAM on all subsets.
    # Use ground-truth poses for evaluation.
    for split, suffix in zip([cfg.train_names, cfg.val_names, cfg.test_names],
                             ['train', 'val', 'test']):
        for loss in cfg.eval_losses:
            eval_cfg = best_cfg.copy()
            eval_cfg.test_names = split
            eval_cfg.train_poses_path = []
            eval_cfg.val_poses_path = []
            eval_cfg.test_poses_path = []
            eval_cfg.loss = loss
            eval_cfg.loss_eval_csv = os.path.join(cfg.log_dir,
                                                  'loss_eval_%s_%s.csv' % (loss, suffix))
            eval_loss(cfg=eval_cfg)

        for slam in cfg.eval_slams:
            eval_cfg = best_cfg.copy()
            eval_cfg.test_names = split
            eval_cfg.train_poses_path = []
            eval_cfg.val_poses_path = []
            eval_cfg.test_poses_path = []
            eval_cfg.slam = slam
            eval_cfg.slam_eval_csv = os.path.join(cfg.log_dir,
                                                  'slam_eval_%s_%s.csv' % (slam, suffix))
            eval_slam(cfg=eval_cfg)


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
    print('Training and evaluating...')
    train_and_eval(cfg)
    print('Training and evaluating finished.')


def main():
    run_from_cmdline()


if __name__ == '__main__':
    main()
