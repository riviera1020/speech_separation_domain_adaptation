
import os
import time
import yaml
import copy
import math
import random
import datetime
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.solver import Solver
from src.saver import Saver
from src.utils import DEV, DEBUG, NCOL, inf_data_gen, read_scale
from src.mcd_conv_tasnet import MCDConvTasNet, DiscrepancyLoss
from src.pit_criterion import cal_loss, SISNR
from src.dataset import wsj0, wsj0_eval
from src.wham import wham, wham_eval
from src.ranger import Ranger
from src.dashboard import Dashboard
from src.scheduler import RampScheduler, ConstantScheduler, DANNScheduler

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

            self.writer = Dashboard(exp_name, self.config, self.log_dir, resume=True)

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
            self.writer = Dashboard(log_name, config, self.log_dir)

        self.epochs = config['solver']['epochs']
        self.start_epoch = config['solver']['start_epoch']
        self.batch_size = config['solver']['batch_size']
        self.grad_clip = config['solver']['grad_clip']
        self.num_workers = config['solver']['num_workers']
        self.num_k = config['solver']['num_k']

        self.step = 0

        self.load_data()
        self.set_model()

        steps_per_epoch = len(self.sup_tr_loader)
        self.b_scheduler = self.set_scheduler(self.config['solver']['b_scheduler'], steps_per_epoch)
        self.c_scheduler = self.set_scheduler(self.config['solver']['c_scheduler'], steps_per_epoch)

    def set_scheduler(self, sch_config):
        if sch_config['function'] == 'ramp':
            return RampScheduler(sch_config['start_step'],
                                 sch_config['end_step'],
                                 sch_config['start_value'],
                                 sch_config['end_value'])
        elif sch_config['function'] == 'constant':
            return ConstantScheduler(sch_config['value'])

    def load_data(self):

        # Set sup&uns dataset
        dset = self.config['data'].get('dset', 'wsj0')
        seg_len = self.config['data']['segment']

        uns_dset = self.config['data'].get('uns_dset', 'vctk')
        uns_seg_len = self.config['data'].get('uns_segment', 2.0)

        print(f'Supvised Dataset   : {dset}')
        print(f'Unsupvised Dataset : {uns_dset}')

        self.sup_dset = dset
        self.uns_dset = uns_dset
        self.sup_tr_loader, self.sup_cv_loader = self.load_dset(self.sup_dset, seg_len)
        self.uns_tr_loader, self.uns_cv_loader = self.load_dset(self.uns_dset, uns_seg_len)
        self.uns_tr_gen = inf_data_gen(self.uns_tr_loader)

    def load_dset(self, dset, seg_len):
        # root: wsj0_root, vctk_root, libri_root
        d = 'wsj' if dset == 'wsj0' else dset # stupid error
        if 'wham' in dset:
            return self.load_wham(dset, seg_len)

        audio_root = self.config['data'][f'{d}_root']
        tr_list = f'./data/{dset}/id_list/tr.pkl'
        cv_list = f'./data/{dset}/id_list/cv.pkl'
        sp_factors = None

        trainset = wsj0(tr_list,
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr',
                sp_factors = sp_factors)
        tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers)

        devset = wsj0_eval(cv_list,
                audio_root = audio_root,
                pre_load = False)
        cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)
        return tr_loader, cv_loader

    def load_wham(self, dset, seg_len):
        audio_root = self.config['data'][f'wsj_root']
        tr_list = f'./data/wsj0/id_list/tr.pkl'
        cv_list = f'./data/wsj0/id_list/cv.pkl'

        scale = read_scale(f'./data/{dset}')
        print(f'Load wham data with scale {scale}')

        trainset = wham(tr_list,
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr',
                scale = scale)
        tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers)

        devset = wham_eval(cv_list,
                audio_root = audio_root,
                pre_load = False,
                mode = 'cv',
                scale = scale)
        cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)
        return tr_loader, cv_loader

    def set_optim(self, config, parameters, optim_dict = None):
        lr = config['lr']
        weight_decay = config['weight_decay']
        optim_type = config['type']

        if optim_type == 'SGD':
            momentum = config['momentum']
            opt = torch.optim.SGD(
                    parameters,
                    lr = lr,
                    momentum = momentum,
                    weight_decay = weight_decay)
        elif optim_type == 'Adam':
            opt = torch.optim.Adam(
                    parameters,
                    lr = lr,
                    weight_decay = weight_decay)
        elif optim_type == 'ranger':
            opt = Ranger(
                    parameters(),
                    lr = lr,
                    weight_decay = weight_decay)
        else:
            print('Specify optim')
            exit()

        if optim_dict != None:
            print('Resume optim')
            opt.load_state_dict(optim_dict)
        return opt

    def set_scheduler(self, sch_config, steps_per_epoch = -1):
        if sch_config['function'] == 'ramp':
            return RampScheduler(sch_config['start_step'],
                                 sch_config['end_step'],
                                 sch_config['start_value'],
                                 sch_config['end_value'],
                                 steps_per_epoch = steps_per_epoch)
        elif sch_config['function'] == 'constant':
            return ConstantScheduler(sch_config['value'])

    def set_model(self):
        self.model = MCDConvTasNet(self.config['model'])
        self.model = self.model.to(DEV)

        dtype = self.config['solver']['discrepancy_type']
        use_pit = self.config['solver']['discrepancy_pit']
        self.dis_loss = DiscrepancyLoss(self.config['model']['C'], dtype, use_pit).to(DEV)

        pre_path = self.config['solver'].get('pretrained', '')
        if pre_path != '':
            self.load_pretrain(pre_path)

        optim_dict = None
        f_optim_dict = None
        if self.resume_model:
            model_path = os.path.join(self.save_dir, 'latest.pth')
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
                f_optim_dict = info_dict['f_optim']

            # dashboard is one-base
            self.writer.set_epoch(self.start_epoch + 1)
            self.writer.set_step(self.step + 1)

        self.opt = self.set_optim(self.config['optim'],
                self.model.get_parameters('G'), optim_dict)
        self.f_opt = self.set_optim(self.config['f_optim'],
                self.model.get_parameters('F'), f_optim_dict)

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

        info_dict = torch.load(pre_path)

        # miss: F1.weight, F2.weight
        # unexpect: network.3.weight(mask1x1)
        miss, unexpect = self.model.load_state_dict(info_dict['state_dict'], strict = False)

        print('Load pretrained model')
        if 'epoch' in info_dict:
            print(f"Epochs: {info_dict['epoch']}")
        elif 'step' in info_dict:
            print(f"Steps : {info_dict['step']}")
        print(info_dict['valid_score'])

    def exec(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):
            self.train_one_epoch(epoch, self.sup_tr_loader, self.uns_tr_gen)

            ## Valid training dataset
            self.valid(self.sup_cv_loader, epoch, prefix = self.sup_dset)
            # Valid not training dataset
            self.valid(self.uns_cv_loader, epoch, no_save = True, prefix = self.uns_dset)

            self.writer.epoch()

    def train_one_epoch(self, epoch, sup_loader, uns_gen):
        self.model.train()
        total_loss = 0.
        total_stepb_loss = 0.
        total_stepc_loss = 0.
        total_maximize_discrepancy = 0.
        cnt = 0

        for i, sample in enumerate(tqdm(sup_loader, ncols = NCOL)):

            sup_mixture = sample['mix'].to(DEV)
            sup_source = sample['ref'].to(DEV)
            sup_lengths = sample['ilens'].to(DEV)
            B = sup_mixture.size(0)

            uns_sample = uns_gen.__next__()
            uns_mixture = uns_sample['mix'].to(DEV)
            uns_lengths = uns_sample['ilens'].to(DEV)

            # sup part
            est_s1, est_s2, m1, m2 = self.model(sup_mixture)

            sup_loss1, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(sup_source, est_s1, sup_lengths)
            sup_loss2, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(sup_source, est_s2, sup_lengths)
            sup_loss = sup_loss1 + sup_loss2

            self.model.zero_grad()
            sup_loss.backward()
            stepa_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.f_opt.step()

            # step b, only update F
            est_s1, est_s2, m1, m2 = self.model(sup_mixture)
            _, _, uns_m1, uns_m2 = self.model(uns_mixture)

            sup_loss1, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(sup_source, est_s1, sup_lengths)
            sup_loss2, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(sup_source, est_s2, sup_lengths)
            anchor_loss = sup_loss1 + sup_loss2
            discrepancy_loss, idx = self.dis_loss(uns_m1, uns_m2)

            max_d = discrepancy_loss.item()
            if self.dis_loss.pit:
                stepb_perms = idx.float().mean().item()

            w = self.b_scheduler.value(self.step)
            maximize_dloss = anchor_loss - w * discrepancy_loss
            self.model.zero_grad()
            maximize_dloss.backward()
            stepb_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.f_opt.step()

            # step c, only update G (not decoder part)
            min_d = 0
            stepc = 0
            stepc_perms = 0
            for i in range(self.num_k):
                _, _, uns_m1, uns_m2 = self.model(uns_mixture)
                discrepancy_loss, idx = self.dis_loss(uns_m1, uns_m2)
                w = self.c_scheduler.value(self.step)
                minimize_dloss = w * discrepancy_loss

                self.model.zero_grad()
                minimize_dloss.backward()
                stepc_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

                min_d += minimize_dloss.item()
                stepc += stepc_grad_norm
                if self.dis_loss.pit:
                    stepc_perms += idx.float().mean().item()

            min_d /= self.num_k
            stepc_grad_norm = stepc / self.num_k
            stepc_perms /= self.num_k

            meta = { 'iter_loss': sup_loss.item(),
                     'iter_stepb_loss': maximize_dloss.item(),
                     'maximize_discrepancy': max_d,
                     'iter_stepc_loss': min_d,
                     'iter_stepa_grad_norm': stepa_grad_norm,
                     'iter_stepb_grad_norm': stepb_grad_norm,
                     'iter_stepc_gard_norm': stepc_grad_norm }
            if self.dis_loss.pit:
                meta['iter_stepb_perms'] = stepb_perms
                meta['iter_stepc_perms'] = stepc_perms
            self.writer.log_step_info('train', meta)

            total_loss += sup_loss.item() * B
            total_stepb_loss += maximize_dloss.item() * B
            total_stepc_loss += min_d * B
            total_maximize_discrepancy += max_d * B
            cnt += B

            self.step += 1
            self.writer.step()

        total_loss /= cnt
        total_stepb_loss /= cnt
        total_stepc_loss /= cnt
        total_maximize_discrepancy /= cnt

        meta = { 'epoch_loss': total_loss,
                 'epoch_stepb_loss': total_stepb_loss,
                 'epoch_stepc_loss': total_stepc_loss,
                 'epoch_maximize_discrepancy': total_maximize_discrepancy }
        self.writer.log_epoch_info('train', meta)

    def valid(self, loader, epoch, no_save = False, prefix = ""):
        self.model.eval()
        total_loss = 0.
        total_sisnri1 = 0.
        total_sisnri2 = 0.
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

                est_s1, est_s2, m1, m2 = self.model(padded_mixture)

                loss1, max_snr1, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, est_s1, mixture_lengths)
                loss2, max_snr2, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, est_s2, mixture_lengths)

                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                max_sisnri1 = (max_snr1 - mix_sisnr)
                max_sisnri2 = (max_snr2 - mix_sisnr)

                total_loss += (loss1 + loss2).item() / 2 * B
                total_sisnri1 += max_sisnri1.sum().item()
                total_sisnri2 += max_sisnri2.sum().item()
                cnt += B

        total_loss = total_loss / cnt
        total_sisnri1 = total_sisnri1 / cnt
        total_sisnri2 = total_sisnri2 / cnt

        meta = { f'{prefix}_epoch_loss': total_loss,
                 f'{prefix}_epoch_sisnri': total_sisnri1,
                 f'{prefix}_epoch_sisnri_2': total_sisnri2 }
        self.writer.log_epoch_info('valid', meta)

        crit = max(total_sisnri1, total_sisnri2)
        valid_score = {}
        valid_score['valid_loss'] = total_loss
        valid_score['valid_sisnri'] = crit
        valid_score['valid_sisnri_1'] = total_sisnri1
        valid_score['valid_sisnri_2'] = total_sisnri2

        if no_save:
            return

        model_name = f'{epoch}.pth'
        info_dict = { 'epoch': epoch, 'step': self.step, 'valid_score': valid_score, 'config': self.config }
        info_dict['optim'] = self.opt.state_dict()
        info_dict['f_optim'] = self.f_opt.state_dict()

        self.saver.update(self.model, crit, model_name, info_dict)

        model_name = 'latest.pth'
        self.saver.force_save(self.model, model_name, info_dict)

        if self.use_scheduler:
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.lr_scheduler.step(total_loss)
