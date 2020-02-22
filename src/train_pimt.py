
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
from src.pimt_conv_tasnet import PiMtConvTasNet, SpecTransform
from src.pit_criterion import cal_loss
from src.dataset import wsj0
from src.vctk import VCTK
from src.ranger import Ranger
from src.dashboard import Dashboard

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

        save_name = self.exp_name + '-' + st
        self.save_dir = os.path.join(config['solver']['save_dir'], save_name)
        self.safe_mkdir(self.save_dir)
        self.saver = Saver(config['solver']['max_save_num'], self.save_dir, 'min')
        yaml.dump(config, open(os.path.join(self.save_dir, 'config.yaml'), 'w'),
                default_flow_style = False ,indent = 4)

        log_name = self.exp_name + '-' + st
        self.log_dir = os.path.join(config['solver']['log_dir'], log_name)
        self.safe_mkdir(self.log_dir)
        self.writer = Dashboard(log_name, config, self.log_dir)

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
            self.pi_lambda = self.pi_conf['lambda']
        elif self.mt_conf['use']:
            self.algo = 'mt'
            self.mt_lambda = self.mt_conf['lambda']

        spec_transform = config['solver']['spec_transform']
        self.set_transform(spec_transform)

        self.step = 0
        self.valid_times = 0

        self.load_data()
        self.set_model()

    def set_transform(self, spec_transform):
        self.transform = SpecTransform(spec_transform)

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

        trainset = wsj0('./data/wsj0/id_list/tr.pkl',
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

        devset = wsj0('./data/wsj0/id_list/cv.pkl',
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = False,
                mode = 'cv')
        self.wsj0_cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

    def load_vctk_data(self):

        seg_len = self.config['data']['vctk']['segment']
        audio_root = self.config['data']['vctk_root']

        trainset = VCTK('./data/vctk/id_list/tr.pkl',
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

        devset = VCTK('./data/vctk/id_list/cv.pkl',
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = False,
                mode = 'cv')
        self.vctk_cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

    def set_model(self):
        self.model = PiMtConvTasNet(self.config['model'])
        self.model = self.model.to(DEV)

        self.mse_loss = nn.MSELoss().to(DEV)

        if self.algo == 'mt':
            # TODO gen teacher model
            pass

        optim_dict = None
        if 'resume' in self.config['solver']:
            model_path = self.config['solver']['resume']
            if model_path != '':
                print('Resuming Training')
                print(f'Loading Model: {model_path}')

                info_dict = torch.load(model_path)

                print(f"Previous score: {info_dict['valid_score']}")
                self.start_step = info_dict['epoch'] + 1

                self.model.load_state_dict(info_dict['state_dict'])
                print('Loading model complete')

                self.ema_model.load_state_dict(info_dict['ema_state_dict'])
                print('Loading ema_model complete')

                if self.config['solver']['resume_optim']:
                    optim_dict = info_dict['optim']

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

    def update_ema(self, model, ema_model, alpha, global_step):

        # TODO, weird alpha
        alpha = min(1 - 1/(global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)

    def exec(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):
            if self.algo == 'pi':
                self.train_pi_model(epoch, self.sup_tr_loader, self.uns_tr_gen)
            elif self.algo == 'mt':
                self.train_mt(epoch, self.sup_tr_loader, self.uns_tr_gne)

            # Valid training dataset
            self.valid(self.sup_cv_loader, epoch, prefix = self.sup_dset)

            # Valid not training dataset
            self.valid(self.uns_cv_loader, epoch, no_save = True, prefix = self.uns_dset)

            self.writer.epoch()

    def cal_consistency_weight(self, epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
        """Sets the weights for the consistency loss"""
        if epoch > end_ep:
            weight_cl = end_w
        elif epoch < init_ep:
            weight_cl = init_w
        else:
            T = float(epoch - init_ep)/float(end_ep - init_ep)
            weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w #exp
        return weight_cl

    def train_pi_model(self, epoch, sup_loader, uns_gen):
        self.model.train()
        total_loss = 0.
        total_pi = 0.
        cnt = 0

        for i, sample in enumerate(tqdm(sup_loader, ncols = NCOL)):

            # sup part
            padded_mixture = sample['mix'].to(DEV)
            padded_source = sample['ref'].to(DEV)
            mixture_lengths = sample['ilens'].to(DEV)
            B = padded_mixture.size(0)

            estimate_source = self.model(padded_mixture)

            sup_loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)

            # pi training
            uns_sample = uns_gen.__next__()
            padded_mixture = uns_sample['mix'].to(DEV)
            mixture_lengths = uns_sample['ilens'].to(DEV)

            _, _, score_clean, score_noise = self.model.consistency_forward(padded_mixture, self.transform)
            loss_pi = self.mse_loss(score_clean, score_noise)

            r = self.cal_consistency_weight(epoch, end_ep = self.epochs, end_w = self.pi_lambda)
            loss = sup_loss + r * loss_pi

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            meta = { 'iter_loss': sup_loss.item(),
                     'iter_pi_loss': loss_pi.item() }
            self.writer.log_step_info('train', meta)

            total_loss += sup_loss.item() * B
            total_pi += loss_pi.item() * B
            cnt += B

            self.step += 1
            self.writer.step()

        total_loss = total_loss / cnt
        total_pi = total_pi / cnt

        meta = { 'epoch_loss': total_loss,
                 'epoch_pi_loss': total_pi }
        self.writer.log_epoch_info('train', meta)

    def train_mt(self, epoch, sup_loader, uns_gen):
        self.model.train()
        total_loss = 0.
        total_mixup = 0.

        for i, sample in enumerate(tqdm(sup_loader, ncols = NCOL)):

            # sup part
            padded_mixture = sample['mix'].to(DEV)
            padded_source = sample['ref'].to(DEV)
            mixture_lengths = sample['ilens'].to(DEV)

            estimate_source = self.model(padded_mixture)

            sup_loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)

            uns_sample = uns_gen.__next__()
            padded_mixture = uns_sample['mix'].to(DEV)
            mixture_lengths = sample['ilens'].to(DEV)

            with torch.no_grad():
                teacher_out = self.ema_model(padded_mixture)
                s1 = teacher_out[:, 0, :]
                s2 = teacher_out[:, 1, :]

                mlambda = self.beta_sampler.sample((s1.size(0),1)).to(DEV)
                teacher_mix = mlambda * s1 + (1-mlambda) * s2

            student_out = self.model(teacher_mix)
            mixup_loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(teacher_out, student_out, mixture_lengths)

            r = np.exp(float(epoch+1)/self.epochs - 1)
            loss = sup_loss + r * mixup_loss

            # SGD update
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.writer.add_scalar('train/iter_loss', sup_loss.item(), self.step)

            # EMA update
            self.update_ema(self.model, self.ema_model, self.ema_alpha, self.step)

            total_loss += sup_loss.item()
            total_mixup += mixup_loss.item()

            self.step += 1

        total_loss = total_loss / len(sup_loader)
        total_mixup = total_mixup / len(sup_loader)
        self.writer.add_scalar('train/epoch_loss', total_loss, epoch)
        self.writer.add_scalar('train/epoch_mixup_loss', total_mixup, epoch)

    def valid(self, loader, epoch, no_save = False, prefix = ""):
        self.model.eval()
        total_loss = 0.
        total_snr = 0.
        cnt = 0

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)
                B = padded_mixture.size(0)

                estimate_source = self.model(padded_mixture)

                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

                total_loss += loss.item() * B
                total_snr += max_snr.sum().item() * B
                cnt += B

        total_loss = total_loss / cnt
        total_snr = total_snr / cnt

        meta = { f'{prefix}_epoch_loss': total_loss,
                 f'{prefix}_epoch_snr': total_snr }
        self.writer.log_epoch_info('valid', meta)

        valid_score = {}
        valid_score['valid_loss'] = total_loss

        if no_save:
            return

        model_name = f'{epoch}.pth'
        info_dict = { 'epoch': epoch, 'valid_score': valid_score, 'config': self.config }
        info_dict['optim'] = self.opt.state_dict()

        self.saver.update(self.model, total_loss, model_name, info_dict)

        model_name = 'latest.pth'
        self.saver.force_save(self.model, model_name, info_dict)

        if self.use_scheduler:
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.lr_scheduler.step(total_loss)
            #elif self.scheduler_type in [ 'FlatCosine', 'CosineWarmup' ]:
            #    self.lr_scheduler.step(epoch)
