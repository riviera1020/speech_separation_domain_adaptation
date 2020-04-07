
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
from src.mcd_conv_tasnet import MCDConvTasNet
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

    def set_model(self):
        self.model = MCDConvTasNet(self.config['model'])
        self.model = self.model.to(DEV)

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

                if self.fp16 and 'amp' in info_dict:
                    amp.load_state_dict(info_dict['amp'])

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
        # TODO, need to handle mask1x1 not in separator.network
        pass
        #pretrained_optim = self.config['solver'].get('pretrained_optim', False)

        #info_dict = torch.load(pre_path)
        #self.model.load_state_dict(info_dict['state_dict'])

        #print('Load pretrained model')
        #if 'epoch' in info_dict:
        #    print(f"Epochs: {info_dict['epoch']}")
        #elif 'step' in info_dict:
        #    print(f"Steps : {info_dict['step']}")
        #print(info_dict['valid_score'])

        #if pretrained_optim:
        #    return info_dict['optim']
        #else:
        #    return None

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
        total_maximize_discrepancy = 0.
        total_minimize_discrepancy = 0.
        cnt = 0

        for i, sample in enumerate(tqdm(sup_loader, ncols = NCOL)):
            self.opt.zero_grad()

            # sup part
            padded_mixture = sample['mix'].to(DEV)
            padded_source = sample['ref'].to(DEV)
            mixture_lengths = sample['ilens'].to(DEV)
            B = padded_mixture.size(0)

            estimate_source = self.model(padded_mixture)

            sup_loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            loss = sup_loss

            if self.fp16:
                with amp.scale_loss(loss, self.opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # pi on sup
            with torch.no_grad():
                estimate_clean_sup, feat_clean_sup = self.model.fetch_forward(padded_mixture, self.locs)
            estimate_noise_sup, feat_noise_sup = self.model.fetch_forward(padded_mixture, self.locs, self.transform)
            loss_pi_sup = self.con_loss(estimate_clean_sup, estimate_noise_sup, mixture_lengths, feat_clean_sup, feat_noise_sup)

            # pi on uns
            uns_sample = uns_gen.__next__()
            padded_mixture = uns_sample['mix'].to(DEV)
            mixture_lengths = uns_sample['ilens'].to(DEV)

            with torch.no_grad():
                estimate_clean_uns, feat_clean_uns = self.model.fetch_forward(padded_mixture, self.locs)
            estimate_noise_uns, feat_noise_uns = self.model.fetch_forward(padded_mixture, self.locs, self.transform)
            loss_pi_uns = self.con_loss(estimate_clean_uns, estimate_noise_uns, mixture_lengths, feat_clean_uns, feat_noise_uns)

            w_sup = self.cal_consistency_weight(self.step, end_ep = self.warmup_step, init_w = self.sup_init_w, end_w = self.sup_pi_lambda)
            w_uns = self.cal_consistency_weight(self.step, end_ep = self.warmup_step, init_w = self.uns_init_w, end_w = self.uns_pi_lambda)
            loss = w_sup * loss_pi_sup + w_uns * loss_pi_uns

            if self.fp16:
                with amp.scale_loss(loss, self.opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            # forward_hook cause memory leak, need to release(del) them
            for feat in [ feat_clean_sup, feat_noise_sup, feat_clean_uns, feat_noise_uns ]:
                self.model.clean_hook_tensor(feat)

            meta = { 'iter_loss': sup_loss.item(),
                     'iter_pi_sup_loss': loss_pi_sup.item(),
                     'iter_pi_uns_loss': loss_pi_uns.item(),
                     'w_sup': w_sup,
                     'w_uns': w_uns }
            self.writer.log_step_info('train', meta)

            total_loss += sup_loss.item() * B
            total_pi_sup += loss_pi_sup.item() * B
            total_pi_uns += loss_pi_uns.item() * B
            cnt += B

            self.step += 1
            self.writer.step()

        total_loss = total_loss / cnt
        total_pi_sup = total_pi_sup / cnt
        total_pi_uns = total_pi_uns / cnt

        meta = { 'epoch_loss': total_loss,
                 'epoch_pi_sup_loss': total_pi_sup,
                 'epoch_pi_uns_loss': total_pi_uns }
        self.writer.log_epoch_info('train', meta)

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
        if self.fp16:
            info_dict['amp'] = amp.state_dict()

        self.saver.update(self.model, total_sisnri, model_name, info_dict)

        model_name = 'latest.pth'
        self.saver.force_save(self.model, model_name, info_dict)

        if self.use_scheduler:
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.lr_scheduler.step(total_loss)
            #elif self.scheduler_type in [ 'FlatCosine', 'CosineWarmup' ]:
            #    self.lr_scheduler.step(epoch)
