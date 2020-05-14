import yaml
import comet_ml
import torch
import random
import argparse
import numpy as np

from src.utils import read_config, set_device, set_debug

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '--c', type=str)
    parser.add_argument('--mode', type=str, default='baseline')
    parser.add_argument('--path', type=str, default='./config/path.yaml')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', default = -1, type=int)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    use_cuda = not args.cpu
    set_device(use_cuda)
    set_debug(args.debug)

    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    config = read_config(args.config, args.path)

    mode = args.mode
    test = args.test

    if mode == 'baseline':
        if not args.test:
            from src.train_baseline import Trainer as Solver
        else:
            from src.test_baseline import Tester as Solver
    elif mode == 'uns':
        if not args.test:
            from src.train_uns import Trainer as Solver
        else:
            pass
    elif mode == 'semi':
        if not args.test:
            from src.train_semi import Trainer as Solver
        else:
            from src.test_semi import Tester as Solver
    elif mode == 'da':
        if not args.test:
            from src.train_da import Trainer as Solver
        else:
            pass
    elif mode == 'dagan':
        if not args.test:
            from src.train_dagan import Trainer as Solver
        else:
            from src.test_dagan import Tester as Solver
    elif mode == 'dagan_pl':
        from src.train_dagan_pl import Trainer as Solver
    elif mode == 'mmd':
        from src.train_mmd import Trainer as Solver
    elif mode == 'pimt':
        from src.train_pimt import Trainer as Solver
    elif mode == 'debug':
        from src.train_debug import Trainer as Solver
    elif mode == 'freeze':
        from src.train_freeze import Trainer as Solver
    elif mode == 'perturb':
        from src.train_perturb import Trainer as Solver
    elif mode == 'augm':
        from src.train_augm import Trainer as Solver
    elif mode == 'phncls':
        from src.train_phncls import Trainer as Solver
    elif mode == 'mixemb':
        from src.train_mixemb import Trainer as Solver
    elif mode == 'mcd':
        from src.train_mcd import Trainer as Solver
    elif mode == 'limit':
        from src.train_limit import Trainer as Solver
    elif mode == 'remix':
        from src.train_remix import Trainer as Solver
    elif mode == 'cluster':
        if not args.test:
            print('Not imp')
            exit()
        else:
            from src.test_cluster import Tester as Solver
    elif mode == 'dacluster':
        if not args.test:
            print('Not imp')
            exit()
        else:
            from src.test_dacluster import Tester as Solver
    elif mode == 'danorm':
        if not args.test:
            print('Not imp')
            exit()
        else:
            from src.test_danorm import Tester as Solver
    elif mode == 'transnorm':
        if not args.test:
            from src.train_transnorm import Trainer as Solver
        else:
            from src.test_transnorm import Tester as Solver
    else:
        print('Not imp')
        exit()

    s = Solver(config)
    s.exec()
