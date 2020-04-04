import os
import time
import yaml
import math
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.solver import Solver
from src.saver import Saver
from src.utils import DEV, DEBUG, NCOL, inf_data_gen, read_scale
from src.conv_tasnet import ConvTasNet
from src.da_conv_tasnet import DAConvTasNet
from src.domain_cls import DomainClassifier
from src.pit_criterion import cal_loss, SISNR
from src.dataset import wsj0, wsj0_eval
from src.wham import wham, wham_eval
from src.scheduler import RampScheduler, ConstantScheduler, DANNScheduler
from src.gradient_penalty import calc_gradient_penalty
from src.dashboard import Dashboard
from src.ranger import Ranger
from src.mmd import MMDLoss

class Trainer(Solver):
    def __init__(self, config):
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
        self.step = 0

        self.load_data()
        self.set_model()

    def load_data(self):
        # Set training dataset
        dset = 'wsj0'
        if 'dset' in self.config['data']:
            dset = self.config['data']['dset']
        self.dset = dset
        self.uns_dset = self.config['solver'].get('uns_dset', 'vctk')

        # Load loader for sup training
        seg_len = self.config['data']['segment']
        self.sup_loader = self.load_tr_dset(self.dset, seg_len)

        # Load data gen for gan training
        uns_len = self.config['data'].get('uns_segment', 2.0)
        self.sup_gen = inf_data_gen(self.load_tr_dset(self.dset, uns_len))
        self.uns_gen = inf_data_gen(self.load_tr_dset(self.uns_dset, uns_len))

        # Load cv loader
        self.dsets = {}
        for d in [ self.dset, self.uns_dset ]:
            self.dsets[d] = { 'cv': self.load_cv_dset(d) }

    def load_tr_dset(self, dset, seg_len):
        # root: wsj0_root, vctk_root, libri_root
        d = 'wsj' if dset == 'wsj0' else dset # stupid error
        if 'wham' in dset:
            return self.load_wham(dset, seg_len, 'tr')

        audio_root = self.config['data'][f'{d}_root']
        tr_list = f'./data/{dset}/id_list/tr.pkl'
        trainset = wsj0(tr_list,
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr',
                sp_factors = None)
        tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers)
        return tr_loader

    def load_cv_dset(self, dset):
        # root: wsj0_root, vctk_root, libri_root
        d = 'wsj' if dset == 'wsj0' else dset # stupid error
        if 'wham' in dset:
            return self.load_wham(dset, 'cv')

        audio_root = self.config['data'][f'{d}_root']
        cv_list = f'./data/{dset}/id_list/cv.pkl'
        devset = wsj0_eval(cv_list,
                audio_root = audio_root,
                pre_load = False)
        cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)
        return cv_loader

    def load_wham(self, dset, seg_len, mode):
        audio_root = self.config['data'][f'wsj_root']
        tr_list = f'./data/wsj0/id_list/tr.pkl'
        cv_list = f'./data/wsj0/id_list/cv.pkl'

        scale = read_scale(f'./data/{dset}')
        print(f'Load wham data with scale {scale}')

        if mode == 'tr':
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
            return tr_loader
        else:
            devset = wham_eval(cv_list,
                    audio_root = audio_root,
                    pre_load = False,
                    mode = 'cv',
                    scale = scale)
            cv_loader = DataLoader(devset,
                    batch_size = self.batch_size,
                    shuffle = False,
                    num_workers = self.num_workers)
            return cv_loader

    def set_optim(self, models, opt_config):

        params = []
        for m in models:
            params += list(m.parameters())

        lr = opt_config['lr']
        weight_decay = opt_config['weight_decay']

        optim_type = opt_config['type']
        if optim_type == 'SGD':
            momentum = opt_config['momentum']
            opt = torch.optim.SGD(
                    params,
                    lr = lr,
                    momentum = momentum,
                    weight_decay = weight_decay)
        elif optim_type == 'Adam':
            opt = torch.optim.Adam(
                    params,
                    lr = lr,
                    weight_decay = weight_decay)
        elif optim_type == 'ranger':
            opt = Ranger(
                    params,
                    lr = lr,
                    weight_decay = weight_decay)
        else:
            print('Specify optim')
            exit()
        return opt

    def set_scheduler(self, sch_config):
        if sch_config['function'] == 'ramp':
            return RampScheduler(sch_config['start_step'],
                                 sch_config['end_step'],
                                 sch_config['start_value'],
                                 sch_config['end_value'])
        elif sch_config['function'] == 'constant':
            return ConstantScheduler(sch_config['value'])

    def set_model(self):

        self.model = DAConvTasNet(self.config['model']).to(DEV)
        self.opt = self.set_optim([self.model], self.config['optim'])

        sigmas = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
                1e3, 1e4, 1e5, 1e6 ]
        self.mmd_loss = MMDLoss(sigmas).to(DEV)

        pretrained = self.config['solver']['pretrained']
        if pretrained != '':
            info_dict = torch.load(pretrained)
            self.model.load_state_dict(info_dict['state_dict'])

            print('Load pretrained model')
            if 'epoch' in info_dict:
                print(f"Epochs: {info_dict['epoch']}")
            elif 'step' in info_dict:
                print(f"Steps : {info_dict['step']}")
            print(info_dict['valid_score'])

        if self.resume_model:
            model_path = os.path.join(self.save_dir, 'latest.pth')
            print('Resuming Training')
            print(f'Loading Model: {model_path}')

            info_dict = torch.load(model_path)

            print(f"Previous score: {info_dict['valid_score']}")

            self.model.load_state_dict(info_dict['state_dict'])

            print('Loading complete')

            if self.config['solver']['resume_optim']:
                print('Loading optim')

                optim_dict = info_dict['optim']
                self.opt.load_state_dict(optim_dict)

            # dashboard is one-base
            self.writer.set_epoch(self.start_epoch + 1)
            self.writer.set_step(self.step + 1)

        self.mmd_scheduler = self.set_scheduler(self.config['solver']['mmd_scheduler'])

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

    def exec(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):

            self.train_one_epoch(epoch, self.sup_loader, self.sup_gen, self.uns_gen)

            self.valid(self.dsets[self.dset]['cv'], epoch, prefix = self.dset)

            # Valid not training dataset
            for dset in self.dsets:
                if dset != self.dset:
                    self.valid(self.dsets[dset]['cv'], epoch, no_save = True, prefix = dset)

            self.writer.epoch()

    def train_one_epoch(self, epoch, tr_loader, sup_gen, uns_gen):
        self.model.train()
        total_loss = 0.
        total_sisnri = 0.
        total_mmd = 0.
        cnt = 0

        for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):

            padded_mixture = sample['mix'].to(DEV)
            padded_source = sample['ref'].to(DEV)
            mixture_lengths = sample['ilens'].to(DEV)

            estimate_source, _ = self.model(padded_mixture)

            pit_loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)

            src_sample = sup_gen.__next__()
            src_mixture = src_sample['mix'].to(DEV)
            _, src_feature = self.model(src_mixture)

            tgt_sample = uns_gen.__next__()
            tgt_mixture = tgt_sample['mix'].to(DEV)
            _, tgt_feature = self.model(tgt_mixture)

            mmd_loss = self.mmd_loss(src_feature, tgt_feature)

            l = self.mmd_scheduler.value(epoch)
            loss = pit_loss + l * mmd_loss

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            B = padded_source.size(0)
            total_loss += pit_loss.item() * B
            total_mmd += mmd_loss.item() * B
            cnt += B
            with torch.no_grad():
                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                total_sisnri += (max_snr - mix_sisnr).sum()

            meta = { 'iter_loss': loss.item() }
            self.writer.log_step_info('train', meta)

            self.step += 1
            self.writer.step()

        total_loss = total_loss / cnt
        total_sisnri = total_sisnri / cnt
        total_mmd = total_mmd / cnt

        meta = { 'epoch_loss': total_loss,
                 'epoch_sisnri': total_sisnri,
                 'epoch_mmd': total_mmd }
        self.writer.log_epoch_info('train', meta)

    def valid(self, loader, epoch, no_save = False, prefix = "", force_save = False):
        self.model.eval()
        total_loss = 0.
        total_sisnri = 0.
        cnt = 0

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)

                ml = mixture_lengths.max().item()
                padded_mixture = padded_mixture[:, :ml]
                padded_source = padded_source[:, :, :ml]
                B = padded_source.size(0)

                estimate_source, _ = self.model(padded_mixture)

                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                max_sisnri = (max_snr - mix_sisnr)

                total_loss += loss.item() * B
                total_sisnri += max_sisnri.sum().item()
                cnt += B

        total_sisnri = total_sisnri / cnt
        total_loss = total_loss / cnt

        meta = { f'{prefix}_epoch_loss': total_loss,
                 f'{prefix}_epoch_sisnri': total_sisnri }
        self.writer.log_epoch_info('valid', meta)

        valid_score = {}
        valid_score['valid_loss'] = total_loss
        valid_score['valid_sisnri'] = total_sisnri

        if no_save:
            return

        model_name = f'{epoch}.pth'
        info_dict = { 'epoch': epoch, 'valid_score': valid_score, 'config': self.config }
        info_dict['optim'] = self.opt.state_dict()

        self.saver.update(self.model, total_sisnri, model_name, info_dict)

        if force_save:
            self.saver.force_save(self.model, model_name, info_dict)

        model_name = 'latest.pth'
        self.saver.force_save(self.model, model_name, info_dict)

        if self.use_scheduler:
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.lr_scheduler.step(total_loss)
            #elif self.scheduler_type in [ 'FlatCosine', 'CosineWarmup' ]:
            #    self.lr_scheduler.step(epoch)
