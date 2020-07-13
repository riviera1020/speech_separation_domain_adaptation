
import os
import time
import yaml
import copy
import math
import datetime
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from src.solver import Solver
from src.saver import Saver
from src.utils import DEV, DEBUG, NCOL, inf_data_gen
from src.pimt_conv_tasnet import PiMtConvTasNet, InputTransform
from src.pit_criterion import cal_loss, SISNR
from src.dataset import wsj0, wsj0_eval
from src.ranger import Ranger
from src.dashboard import Dashboard
from src.pimt_utils import PITMSELoss

"""
from src.scheduler import FlatCosineLR, CosineWarmupLR
"""

class Trainer(Solver):

    def __init__(self, config):
        #def __init__(self, data, model, optimizer, args):
        super(Trainer, self).__init__(config)

        self.exp_name = config['solver']['exp_name']

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

        self.resume_model = False
        resume_exp_name = config['solver'].get('resume_exp_name', '')
        if resume_exp_name:
            self.resume_model = True
            exp_name = resume_exp_name
            self.save_dir = os.path.join(self.config['solver']['save_dir'], exp_name)
            self.log_dir = os.path.join(self.config['solver']['log_dir'], exp_name)

            if not os.path.isdir(self.save_dir) or not os.path.isdir(self.log_dir):
                print('Resume Exp name Error')
                exit()

            self.saver = Saver(
                    self.config['solver']['max_save_num'],
                    self.save_dir,
                    'max',
                    resume = True,
                    resume_score_fn = lambda x: x['valid_score']['valid_sisnri'])

            #self.writer = Dashboard(exp_name, self.config, self.log_dir, resume=True)

        else:
            save_name = self.exp_name + '-' + st
            self.save_dir = os.path.join(config['solver']['save_dir'], save_name)
            self.safe_mkdir(self.save_dir)
            self.saver = Saver(config['solver']['max_save_num'], self.save_dir, 'max')
            yaml.dump(config, open(os.path.join(self.save_dir, 'config.yaml'), 'w'),
                    default_flow_style = False ,indent = 4)

            log_name = self.exp_name + '-' + st
            self.log_dir = os.path.join(config['solver']['log_dir'], log_name)
            self.safe_mkdir(self.log_dir)
            #self.writer = Dashboard(log_name, config, self.log_dir)
        self.writer = None

        self.epochs = config['solver']['epochs']
        self.start_epoch = config['solver']['start_epoch']
        self.batch_size = config['solver']['batch_size']
        self.grad_clip = config['solver']['grad_clip']
        self.num_workers = config['solver']['num_workers']

        self.pi_conf = config['solver'].get('pi', {'use': False})
        self.mt_conf = config['solver'].get('mt', {'use': False})
        if self.pi_conf['use'] == self.mt_conf['use']:
            print('Specify to only use pi or mt algo')
            exit()
        elif self.pi_conf['use']:
            self.algo = 'pi'
            self.loss_type = self.pi_conf['loss_type']
            self.sup_pi_lambda = self.pi_conf['sup_lambda']
            self.uns_pi_lambda = self.pi_conf['uns_lambda']
        elif self.mt_conf['use']:
            self.algo = 'mt'
            self.mt_lambda = self.mt_conf['lambda']

        input_transform = config['solver']['input_transform']
        self.set_transform(input_transform)

        self.step = 0
        self.valid_times = 0

        self.load_data()
        self.set_model()

    def set_transform(self, t_conf):
        self.transform = InputTransform(t_conf)

    def load_data(self):

        self.load_wsj0_data()
        self.load_vctk_data()

        # Set training dataset
        dset = 'wsj0'
        if 'dset' in self.config['data']:
            dset = self.config['data']['dset']

        if dset == 'wsj0':
            self.sup_tr_loader = self.wsj0_tr_loader
            self.sup_cv_loader = self.wsj0_cv_loader
            self.uns_tr_loader = self.vctk_tr_loader
            self.uns_tr_gen = inf_data_gen(self.vctk_tr_loader)
            self.uns_cv_loader = self.vctk_cv_loader
            self.sup_dset = 'wsj0'
            self.uns_dset = 'vctk'
        else:
            self.sup_tr_loader = self.vctk_tr_loader
            self.sup_cv_loader = self.vctk_cv_loader
            self.uns_tr_gen = inf_data_gen(self.wsj0_tr_loader)
            self.uns_cv_loader = self.wsj0_cv_loader
            self.sup_dset = 'vctk'
            self.uns_dset = 'wsj0'

    def load_wsj0_data(self):

        seg_len = self.config['data']['wsj0']['segment']
        audio_root = self.config['data']['wsj_root']

        trainset = wsj0('./data/wsj0/id_list/cv.pkl',
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr')
        self.wsj0_tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                drop_last = True)

        devset = wsj0_eval('./data/wsj0/id_list/cv.pkl',
                audio_root = audio_root,
                pre_load = False)
        self.wsj0_cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

    def load_vctk_data(self):

        seg_len = self.config['data']['vctk']['segment']
        audio_root = self.config['data']['vctk_root']

        trainset = wsj0('./data/vctk/id_list/cv.pkl',
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr')
        self.vctk_tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                drop_last = True)

        devset = wsj0_eval('./data/vctk/id_list/cv.pkl',
                audio_root = audio_root,
                pre_load = False)
        self.vctk_cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

    def set_model(self):
        self.model = PiMtConvTasNet(self.config['model'])
        self.model = self.model.to(DEV)

        #self.mse_loss = nn.MSELoss().to(DEV)
        self.mse_loss = PITMSELoss().to(DEV)

        if self.algo == 'mt':
            # TODO gen teacher model
            pass

        # TODO, get optim_dict from pretrained and resume
        # maybe buggy
        optim_dict = None
        pre_path = self.config['solver'].get('pretrained', '')
        if pre_path != '':
            optim_dict = self.load_pretrain(pre_path)

        optim_dict = None
        if self.resume_model:
            model_path = os.path.join(self.save_dir, 'latest.pth')
            if model_path != '':
                print('Resuming Training')
                print(f'Loading Model: {model_path}')

                info_dict = torch.load(model_path)

                print(f"Previous score: {info_dict['valid_score']}")
                self.start_epoch = info_dict['epoch'] + 1

                if 'step' in info_dict:
                    self.step = info_dict['step']

                self.model.load_state_dict(info_dict['state_dict'])
                print('Loading model complete')

                if self.config['solver']['resume_optim']:
                    optim_dict = info_dict['optim']

                # dashboard is one-base
                self.writer.set_epoch(self.start_epoch + 1)
                self.writer.set_step(self.step + 1)

        lr = self.config['optim']['lr']
        weight_decay = self.config['optim']['weight_decay']

        optim_type = self.config['optim']['type']
        if optim_type == 'SGD':
            momentum = self.config['optim']['momentum']
            self.opt = torch.optim.SGD(
                    self.model.parameters(),
                    lr = lr,
                    momentum = momentum,
                    weight_decay = weight_decay)
        elif optim_type == 'Adam':
            self.opt = torch.optim.Adam(
                    self.model.parameters(),
                    lr = lr,
                    weight_decay = weight_decay)
        elif optim_type == 'ranger':
            self.opt = Ranger(
                    self.model.parameters(),
                    lr = lr,
                    weight_decay = weight_decay)
        else:
            print('Specify optim')
            exit()

        if optim_dict != None:
            print('Resume optim')
            self.opt.load_state_dict(optim_dict)

        self.use_scheduler = False
        if 'scheduler' in self.config['solver']:
            self.use_scheduler = self.config['solver']['scheduler']['use']
            self.scheduler_type = self.config['solver']['scheduler']['type']

            if self.scheduler_type == 'ReduceLROnPlateau':
                patience = self.config['solver']['scheduler']['patience']
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.opt,
                        mode = 'min',
                        factor = 0.5,
                        patience = patience,
                        verbose = True)

    def load_pretrain(self, pre_path):
        pretrained_optim = self.config['solver'].get('pretrained_optim', False)

        info_dict = torch.load(pre_path)
        self.model.load_state_dict(info_dict['state_dict'])

        print('Load pretrained model')
        if 'epoch' in info_dict:
            print(f"Epochs: {info_dict['epoch']}")
        elif 'step' in info_dict:
            print(f"Steps : {info_dict['step']}")
        print(info_dict['valid_score'])

        if pretrained_optim:
            return info_dict['optim']
        else:
            return None

    def update_ema(self, model, ema_model, alpha, global_step):

        # TODO, weird alpha
        alpha = min(1 - 1/(global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)

    def exec(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):
            self.train_pi_model(epoch, self.sup_tr_loader, self.uns_tr_loader)
            exit()

            # Valid training dataset
            self.valid(self.sup_cv_loader, epoch, prefix = self.sup_dset)

            # Valid not training dataset
            self.valid(self.uns_cv_loader, epoch, no_save = True, prefix = self.uns_dset)

            #self.writer.epoch()

    def train_pi_model(self, epoch, sup_loader, uns_loader):
        self.model.train()
        total_loss = 0.
        total_sisnri = 0
        cnt = 0

        with torch.no_grad():
            total_sisnri = 0
            cnt = 0

            for i, sample in enumerate(tqdm(sup_loader, ncols = NCOL)):

                # sup part
                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)
                B = padded_mixture.size(0)

                #estimate_source = self.model.noise_forward(padded_mixture, self.transform)

                with torch.no_grad():
                    #estimate_source = self.model(padded_mixture)
                    estimate_source = self.model.K_forward(padded_mixture, K = 2)

                sup_loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)
                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                total_sisnri += (max_snr - mix_sisnr).sum()
                cnt += B

            sup_sisnri = total_sisnri / cnt

            total_sisnri = 0
            cnt = 0
            for i, sample in enumerate(tqdm(uns_loader, ncols = NCOL)):
                # pi training
                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)

                #estimate_source = self.model.noise_forward(padded_mixture, self.transform)
                with torch.no_grad():
                    #estimate_source = self.model(padded_mixture)
                    estimate_source = self.model.K_forward(padded_mixture, K = 2)

                sup_loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)
                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                total_sisnri += (max_snr - mix_sisnr).sum()
                cnt += B

            uns_sisnri = total_sisnri / cnt

        print(f'sup sisnri: {sup_sisnri}')
        print(f'uns sisnri: {uns_sisnri}')

    def valid(self, loader, epoch, no_save = False, prefix = ""):
        self.model.eval()
        total_loss = 0.
        total_sisnri = 0.
        cnt = 0

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)
                B = padded_mixture.size(0)

                ml = mixture_lengths.max().item()
                padded_mixture = padded_mixture[:, :ml]
                padded_source = padded_source[:, :, :ml]

                estimate_source = self.model(padded_mixture)

                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                max_sisnri = (max_snr - mix_sisnr)

                total_loss += loss.item() * B
                total_sisnri += max_sisnri.sum().item()
                cnt += B

        total_loss = total_loss / cnt
        total_sisnri = total_sisnri / cnt

        meta = { f'{prefix}_epoch_loss': total_loss,
                 f'{prefix}_epoch_sisnri': total_sisnri }
        self.writer.log_epoch_info('valid', meta)

        valid_score = {}
        valid_score['valid_loss'] = total_loss
        valid_score['valid_sisnri'] = total_sisnri

        if no_save:
            return

        model_name = f'{epoch}.pth'
        info_dict = { 'epoch': epoch, 'step': self.step, 'valid_score': valid_score, 'config': self.config }
        info_dict['optim'] = self.opt.state_dict()

        self.saver.update(self.model, total_sisnri, model_name, info_dict)

        model_name = 'latest.pth'
        self.saver.force_save(self.model, model_name, info_dict)

        if self.use_scheduler:
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.lr_scheduler.step(total_loss)
            #elif self.scheduler_type in [ 'FlatCosine', 'CosineWarmup' ]:
            #    self.lr_scheduler.step(epoch)
