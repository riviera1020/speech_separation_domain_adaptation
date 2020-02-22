"""
Ref:
    https://medium.com/@sunprince12014/comet-ml-%E4%BD%A0%E5%BF%85%E9%A0%88%E7%9F%A5%E9%81%93%E7%9A%84-ml-%E5%AF%A6%E9%A9%97%E7%AE%A1%E7%90%86%E7%A5%9E%E5%99%A8-a4d3b4b16716?
"""
import os
from pathlib import Path
from comet_ml import Experiment, ExistingExperiment

from src.utils import DEBUG

class Dashboard:
    """Record training/evaluation statistics to comet
    :params config: dict
    :params paras: namespace
    :params log_dir: Path
    """
    def __init__(self, exp_name, config, log_dir, resume=False):
        self.log_dir = log_dir
        self.expkey_f = Path(self.log_dir, 'exp_key')

        self.global_step = 1
        self.global_epoch = 1

        if resume:
            assert self.expkey_f.exists(), f"Cannot find comet exp key in {self.log_dir}"
            with open(Path(self.log_dir,'exp_key'),'r') as f:
                exp_key = f.read().strip()
            self.exp = ExistingExperiment(previous_experiment=exp_key,
                                          auto_output_logging=None,
                                          auto_metric_logging=None,
                                          display_summary=False,
                                          )
        else:
            self.exp = Experiment(auto_output_logging=None,
                                  auto_metric_logging=None,
                                  display_summary=False,
                                  )
            with open(self.expkey_f, 'w') as f:
                print(self.exp.get_key(), file=f)

            self.log_config(config)

            self.exp.set_name(exp_name)
            if DEBUG:
                self.exp.add_tag("debug")

        ##slurm-related, record the jobid
        hostname = os.uname()[1]
        if len(hostname.split('.')) == 2 and hostname.split('.')[1] == 'speech':
            self.exp.log_other('jobid', int(os.getenv('PMIX_NAMESPACE').split('.')[2]))
        else:
            self.exp.log_other('jobid', -1)

    def log_config(self,config):
        #NOTE: depth at most 2
        for block in config:
            for n, p in config[block].items():
                if isinstance(p, dict):
                    self.exp.log_parameters(p, prefix=f'{block}-{n}')
                else:
                    self.exp.log_parameter(f'{block}-{n}', p)

    def set_status(self,status):
        ## training / trained / decode / completed
        self.exp.log_other('status', status)

    def step(self, n=1):
        self.global_step += n

    def set_step(self, global_step=1):
        self.global_step = global_step

    def epoch(self, n=1):
        self.global_epoch += n

    def set_epoch(self, global_epoch=1):
        self.global_epoch = global_epoch

    def log_step_info(self, prefix, info):
        self.exp.log_metrics({k: float(v) for k, v in info.items()}, prefix=prefix, step=self.global_step)

    def log_epoch_info(self, prefix, info):
        self.exp.log_metrics({k: float(v) for k, v in info.items()}, prefix=prefix, step=self.global_epoch)

    def log_step(self):
        self.exp.log_other('step', self.global_step)

    def log_epoch(self):
        self.exp.log_other('epoch', self.global_epoch)

    def add_figure(self, fig_name, data):
        self.exp.log_figure(figure_name=fig_name, figure=data, step=self.global_step)

    def check(self):
        if not self.exp.alive:
            print("Comet logging stopped")
