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
from src.conv_tasnet import ConvTasNet
from src.pit_criterion import cal_loss, SISNR
from src.dataset import wsj0, wsj0_eval
from src.wham import wham, wham_eval
from src.limited_dataset import LimitDataset, LimitWham
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

        self.jointly = config['solver']['jointly']
        self.jointly_w = config['solver']['jointly_w']

        self.load_data()
        self.set_model()

        config['limit_info'] = self.limit_info
        if not resume_exp_name:
            yaml.dump(config, open(os.path.join(self.save_dir, 'config.yaml'), 'w'),
                    default_flow_style = False ,indent = 4)

        self.script_name = os.path.basename(__file__).split('.')[0].split('_')[-1]
        self.writer.add_tags(self.script_name)

    def set_scheduler(self, sch_config):
        if sch_config['function'] == 'ramp':
            return RampScheduler(sch_config['start_step'],
                                 sch_config['end_step'],
                                 sch_config['start_value'],
                                 sch_config['end_value'])
        elif sch_config['function'] == 'constant':
            return ConstantScheduler(sch_config['value'])

    def load_data(self):
        #def load_limit(self, dset, seg_len, spk_num, utts_per_spk):

        # Set sup&uns dataset
        dset = self.config['data'].get('dset', 'wsj0')
        seg_len = self.config['data']['segment']

        limit_dset = self.config['data']['limit_dset']
        limit_seg_len = self.config['data']['limit_segment']
        limit_spk_num = self.config['data']['limit_spk_num']
        limit_utts_per_spk = self.config['data']['limit_utts_per_spk']

        print(f'Pretrained Dataset   : {dset}')
        print(f'Limited Dataset : {limit_dset}')

        self.sup_dset = dset
        self.limit_dset = limit_dset
        self.sup_tr_loader, self.sup_cv_loader = self.load_dset(self.sup_dset, seg_len)
        self.pretrained_tr_gen = inf_data_gen(self.sup_tr_loader)

        _, self.limit_cv_loader = self.load_dset(self.limit_dset, 4.0)
        self.limit_tr_loader = self.load_limit(limit_dset, limit_seg_len, limit_spk_num, limit_utts_per_spk)

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

    def load_limit(self, dset, seg_len, spk_num, utts_per_spk):
        if 'wham' not in dset:
            d = 'wsj' if dset == 'wsj0' else dset
            audio_root = self.config['data'][f'{d}_root']
            tr_list = f'./data/{dset}/single_list/tr.pkl'
            spk_info = f'./data/{dset}/spk_gender.pkl'

            trainset = LimitDataset(tr_list,
                    audio_root = audio_root,
                    seg_len = seg_len,
                    spk_info = spk_info,
                    spk_num = spk_num,
                    utts_per_spk = utts_per_spk,
                    mode = 'tr')
        else:
            audio_root = self.config['data'][f'wsj_root']
            tr_list = f'./data/wsj0/single_list/tr.pkl'
            spk_info = f'./data/wsj0/spk_gender.pkl'
            scale = read_scale(f'./data/{dset}')
            print(f'Load wham data with scale {scale}')

            trainset = LimitWham(tr_list,
                    audio_root = audio_root,
                    seg_len = seg_len,
                    spk_info = spk_info,
                    spk_num = spk_num,
                    utts_per_spk = utts_per_spk,
                    mode = 'tr',
                    scale = scale)

        self.limit_info = trainset.get_info()
        tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers)
        return tr_loader

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
                    parameters,
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
        self.model = ConvTasNet(self.config['model'])
        self.model = self.model.to(DEV)

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

            # dashboard is one-base
            self.writer.set_epoch(self.start_epoch + 1)
            self.writer.set_step(self.step + 1)

        self.opt = self.set_optim(self.config['optim'], self.model.parameters(), optim_dict)

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
        self.model.load_state_dict(info_dict['state_dict'])

        print('Load pretrained model')
        if 'epoch' in info_dict:
            print(f"Epochs: {info_dict['epoch']}")
        elif 'step' in info_dict:
            print(f"Steps : {info_dict['step']}")
        print(info_dict['valid_score'])

    def exec(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):
            self.train_one_epoch(epoch, self.limit_tr_loader, self.pretrained_tr_gen)

            # Valid limit dataset
            self.valid(self.limit_cv_loader, epoch, prefix = self.limit_dset)
            # Valid on pretraining dataset
            self.valid(self.sup_cv_loader, epoch, no_save = True, prefix = self.sup_dset)

            self.writer.epoch()

    def train_one_epoch(self, epoch, sup_loader, pretrained_gen):
        self.model.train()
        total_loss = 0.
        total_pretrained_loss = 0.
        cnt = 0

        for i, sample in enumerate(tqdm(sup_loader, ncols = NCOL)):

            sup_mixture = sample['mix'].to(DEV)
            sup_source = sample['ref'].to(DEV)
            sup_lengths = sample['ilens'].to(DEV)
            B = sup_mixture.size(0)

            # sup part
            est_source = self.model(sup_mixture)
            limit_loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(sup_source, est_source, sup_lengths)
            loss = limit_loss

            if self.jointly:
                pre_sample = pretrained_gen.__next__()
                pre_mixture = pre_sample['mix'].to(DEV)
                pre_source = pre_sample['ref'].to(DEV)
                pre_lengths = pre_sample['ilens'].to(DEV)

                est_source = self.model(pre_mixture)
                pre_loss, max_snr, estimate_source, reorder_estimate_source = \
                         cal_loss(pre_source, est_source, pre_lengths)

                loss += self.jointly_w * pre_loss

            self.model.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            meta = { 'iter_loss': limit_loss.item() }
            total_loss += limit_loss.item() * B
            cnt += B

            if self.jointly:
                meta['iter_pretrained_loss'] = pre_loss.item()
                total_pretrained_loss += pre_loss.item() * B

            self.writer.log_step_info('train', meta)
            self.step += 1
            self.writer.step()

        total_loss /= cnt
        total_pretrained_loss /= cnt

        meta = { 'epoch_loss': total_loss,
                 'epoch_pretrained_loss': total_pretrained_loss }
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

                estimate_source = self.model(padded_mixture)

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
            model_name = f'{epoch}_force.pth'
            self.saver.force_save(self.model, model_name, info_dict)

        model_name = 'latest.pth'
        self.saver.force_save(self.model, model_name, info_dict)

        if self.use_scheduler:
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.lr_scheduler.step(total_loss)
