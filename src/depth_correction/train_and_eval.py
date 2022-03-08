from __future__ import absolute_import, division, print_function
from .config import Config
from .eval import eval_loss_all, eval_slam_all
from .train import train
from argparse import ArgumentParser


def train_and_eval(cfg: Config):
    best_cfg = train(cfg)
    eval_loss_all(best_cfg)
    eval_slam_all(best_cfg)


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
