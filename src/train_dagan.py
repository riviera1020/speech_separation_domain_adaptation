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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from src.solver import Solver
from src.saver import Saver
from src.utils import DEV, DEBUG, NCOL, inf_data_gen
from src.conv_tasnet import ConvTasNet
from src.da_conv_tasnet import DAConvTasNet
from src.domain_cls import DomainClassifier
from src.pit_criterion import cal_loss, cal_norm
from src.dataset import wsj0
from src.vctk import VCTK
from src.scheduler import RampScheduler, ConstantScheduler, DANNScheduler
from src.gradient_penalty import calc_gradient_penalty

class Trainer(Solver):

    def __init__(self, config, stream = None):
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
        self.writer = SummaryWriter(self.log_dir)

        if stream != None:
            self.writer.add_text('Config', stream)

        self.total_steps = config['solver']['total_steps']
        self.start_step = config['solver']['start_step']
        self.batch_size = config['solver']['batch_size']
        self.D_grad_clip = config['solver']['D_grad_clip']
        self.G_grad_clip = config['solver']['G_grad_clip']
        self.num_workers = config['solver']['num_workers']
        self.valid_step = config['solver']['valid_step']
        self.valid_time = 0

        self.g_iters = config['solver']['g_iters']
        self.d_iters = config['solver']['d_iters']

        self.gp_lambda = config['solver']['gp_lambda']

        self.load_data()
        self.set_model()

    def load_data(self):
        self.load_wsj0_data()
        self.load_vctk_data()

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
        self.wsj0_gen = inf_data_gen(self.wsj0_tr_loader)

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
        self.vctk_gen = inf_data_gen(self.vctk_tr_loader)

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
                    model.params,
                    lr = lr,
                    weight_decay = weight_decay)
        else:
            print('Specify optim')
            exit()
        return opt

    def set_model(self):

        self.G = DAConvTasNet(self.config['model']['gen']).to(DEV)
        self.D = DomainClassifier(self.G.B, self.config['model']['domain_cls']).to(DEV)

        self.g_optim = self.set_optim([self.G], self.config['g_optim'])
        self.d_optim = self.set_optim([self.D], self.config['d_optim'])

        #self.src_label = torch.Tensor([0]).long().to(DEV)
        #self.tgt_label = torch.Tensor([1]).long().to(DEV)

        model_path = self.config['solver']['resume']
        if model_path != '':
            print('Resuming Training')
            print(f'Loading Model: {model_path}')

            info_dict = torch.load(model_path)

            print(f"Previous score: {info_dict['valid_score']}")

            self.G.load_state_dict(info_dict['state_dict'])
            self.D.load_state_dict(info_dict['D_state_dict'])

            print('Loading complete')

            if self.config['solver']['resume_optim']:
                print('Loading optim')

                optim_dict = info_dict['g_optim']
                self.g_optim.load_state_dict(optim_dict)

                optim_dict = info_dict['d_optim']
                self.d_optim.load_state_dict(optim_dict)

        Lg_config = self.config['solver']['Lg_scheduler']
        if Lg_config['function'] == 'ramp':
            self.Lg_scheduler = RampScheduler(Lg_config['start_step'],
                                              Lg_config['end_step'],
                                              Lg_config['start_value'],
                                              Lg_config['end_value'])
        elif Lg_config['function'] == 'constant':
            self.Lg_scheduler = ConstantScheduler(Lg_config['value'])

        self.use_scheduler = False
        if 'scheduler' in self.config['solver']:
            self.use_scheduler = self.config['solver']['scheduler']['use']
            self.scheduler_type = self.config['solver']['scheduler']['type']

            if self.scheduler_type == 'ReduceLROnPlateau':
                patience = self.config['solver']['scheduler']['patience']
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.g_optim,
                        mode = 'min',
                        factor = 0.5,
                        patience = patience,
                        verbose = True)

    def log_meta(self, meta, dset):
        for key in meta:
            value = meta[key]
            name = f'{dset}_{key}'
            self.writer.add_scalar(f'valid/{name}', value, self.valid_time)

    def exec(self):

        self.G.train()
        for step in tqdm(range(self.start_step, self.total_steps), ncols = NCOL):

            # supervised
            self.train_sup_once(step, self.wsj0_gen)

            # semi part
            self.train_dis_once(step, self.wsj0_gen, self.vctk_gen)
            self.train_gen_once(step, self.wsj0_gen, self.vctk_gen)

            if step % self.valid_step == 0 and step != 0:
                self.G.eval()
                wsj0_meta = self.valid(self.wsj0_cv_loader)
                vctk_meta = self.valid(self.vctk_cv_loader)
                self.G.train()

                if self.use_scheduler:
                    if self.scheduler_type == 'ReduceLROnPlateau':
                        self.lr_scheduler.step(wsj0_meta['valid_loss'])

                # Do saving
                self.log_meta(wsj0_meta, 'wsj0')
                self.log_meta(vctk_meta, 'vctk')

                model_name = f'{step}.pth'
                valid_score = { 'wsj0': wsj0_meta, 'vctk': vctk_meta }
                info_dict = { 'step': step, 'valid_score': valid_score }
                info_dict['g_optim'] = self.g_optim.state_dict()
                info_dict['d_optim'] = self.d_optim.state_dict()
                info_dict['D_state_dict'] = self.D.state_dict()

                # TODO, use vctk_loss as save crit
                save_crit = vctk_meta['valid_loss']
                self.saver.update(self.G, save_crit, model_name, info_dict)

                model_name = 'latest.pth'
                self.saver.force_save(self.G, model_name, info_dict)

                self.valid_time += 1


    def train_sup_once(self, step, data_gen):

        sample = data_gen.__next__()

        padded_mixture = sample['mix'].to(DEV)
        padded_source = sample['ref'].to(DEV)
        mixture_lengths = sample['ilens'].to(DEV)

        estimate_source, _ = self.G(padded_mixture)

        loss, max_snr, estimate_source, reorder_estimate_source = \
            cal_loss(padded_source, estimate_source, mixture_lengths)

        self.g_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.G_grad_clip)
        self.g_optim.step()

    def train_dis_once(self, step, src_gen, tgt_gen):
        # assert batch_size is even

        total_d_loss = 0.
        total_gp = 0.
        for _ in range(self.d_iters):

            # fake(src) sample
            sample = src_gen.__next__()
            src_mixture = sample['mix'].to(DEV)

            with torch.no_grad():
                _, src_feat = self.G(src_mixture)

            d_fake_loss = self.D(src_feat).mean()

            # true(tgt) sample
            sample = tgt_gen.__next__()
            tgt_mixture = sample['mix'].to(DEV)

            with torch.no_grad():
                _, tgt_feat = self.G(tgt_mixture)

            d_real_loss = - self.D(tgt_feat).mean()

            d_loss = d_real_loss + d_fake_loss

            gp = calc_gradient_penalty(self.D, tgt_feat, src_feat)
            d_loss = d_loss + self.gp_lambda * gp

            self.d_optim.zero_grad()
            d_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.D_grad_clip)
            if math.isnan(grad_norm):
                print('Error : grad norm is NaN @ step '+str(step))
            else:
                self.d_optim.step()

            total_d_loss += d_loss.item()

        total_d_loss /= self.d_iters
        total_gp /= self.d_iters

        self.writer.add_scalar('train/d_loss', total_d_loss, step)
        self.writer.add_scalar('train/gradient_penalty', total_gp, step)

    def train_gen_once(self, step, src_gen, tgt_gen):
        # Only remain gan now

        total_g_loss = 0.
        weighted_g_loss = 0.
        for _ in range(self.g_iters):

            # fake(src) sample
            sample = src_gen.__next__()
            src_mixture = sample['mix'].to(DEV)

            _, src_feat = self.G(src_mixture)

            g_fake_loss = self.D(src_feat).mean()

            # true(tgt) sample
            sample = tgt_gen.__next__()
            tgt_mixture = sample['mix'].to(DEV)

            _, tgt_feat = self.G(tgt_mixture)

            g_real_loss = - self.D(tgt_feat).mean()

            g_loss = -(g_real_loss + g_fake_loss)
            g_lambda = self.Lg_scheduler.value(step)
            _g_loss = g_loss * g_lambda

            self.g_optim.zero_grad()
            _g_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.G_grad_clip)
            if math.isnan(grad_norm):
                print('Error : grad norm is NaN @ step '+str(step))
            else:
                self.g_optim.step()

            total_g_loss += g_loss.item()
            weighted_g_loss += _g_loss.item()

        total_g_loss /= self.g_iters
        weighted_g_loss /= self.g_iters
        self.writer.add_scalar('train/g_loss', total_g_loss, step)
        self.writer.add_scalar('train/weighted_g_loss', weighted_g_loss, step)


    def valid(self, loader):
        total_loss = 0.
        total_snr = 0.

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)

                estimate_source, feature = self.G(padded_mixture)

                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

                total_loss += loss.item()
                total_snr += max_snr.mean().item()

        total_loss = total_loss / len(loader)
        total_snr = total_snr / len(loader)

        meta = {}
        meta['valid_loss'] = total_loss
        meta['valid_snr'] = total_snr

        return meta
