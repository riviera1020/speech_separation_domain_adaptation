
import os
import importlib
from src.utils import read_path_conf

class Solver():
    def __init__(self, config):
        self.config = config
        self.test_after_finished = self.config['solver'].get('test_after_finished', True)

    def construct_test_conf(self, dsets = 'all', sdir = '', choose_best = False, compute_sdr = False):
        exp_name = os.path.basename(self.save_dir)
        if dsets == 'all':
            dsets = [ 'wsj0', 'vctk', 'wham', 'wham-easy' ]
        conf = { 'data': {}, 'solver': {} }

        conf['data']['dsets'] = dsets
        conf['data']['sample_rate'] = 8000

        pconf = read_path_conf()
        conf['data'].update(pconf)

        conf['solver']['train_config'] = os.path.join(self.save_dir, 'config.yaml')
        conf['solver']['compute_sdr'] = compute_sdr

        if sdir == '':
            rdir = os.path.join('./result/', exp_name)
        else:
            rdir = os.path.join('./result/', sdir, exp_name)
        conf['solver']['result_dir'] = rdir

        if choose_best:
            with open(os.path.join(self.save_dir, 'save.log')) as f:
                line = f.readlines()[0].rstrip()
                cpath, score = line.split(':')
        else:
            cpath = os.path.join(self.save_dir, 'latest.pth')
        conf['solver']['checkpoint']  = cpath
        return conf

    @staticmethod
    def safe_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def run_tester(name, config):
        name = name.replace('.py', '')
        name = f'src.{name}'
        mod = importlib.import_module(name)
        tester = mod.Tester(config)
        return tester.exec()
