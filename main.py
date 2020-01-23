import yaml
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

    with open(args.config, 'r') as stream:
        data = stream.read().replace('\n', '<br/>').replace(' ', '&nbsp; &nbsp;')
    stream = data

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
            pass
    elif mode == 'mixup':
        if not args.test:
            from src.train_mixup import Trainer as Solver
        else:
            pass
    elif mode == 'vat':
        if not args.test:
            from src.train_vat import Trainer as Solver
        else:
            pass
    elif mode == 'debug':
        from src.train_debug import Trainer as Solver
    elif mode == 'freeze':
        from src.train_freeze import Trainer as Solver
    else:
        print('Not imp')
        exit()

    s = Solver(config, stream)
    s.exec()
