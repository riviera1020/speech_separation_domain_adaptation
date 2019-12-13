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
from src.da_conv_tasnet import DAConvTasNet, DomainClassifier
from src.pit_criterion import cal_loss, cal_norm
from src.dataset import wsj0
from src.vctk import VCTK
from src.discriminator import RWD
from src.MSD import MultiScaleDiscriminator
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

        self.epochs = config['solver']['epochs']
        self.start_epoch = config['solver']['start_epoch']
        self.batch_size = config['solver']['batch_size']
        self.grad_clip = config['solver']['grad_clip']
        self.num_workers = config['solver']['num_workers']
        self.step = 0
        self.valid_time = 0

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

        self.optim = self.set_optim([self.G, self.D], self.config['optim'])
        self.domain_loss = nn.CrossEntropyLoss().to(DEV)

        self.src_label = torch.Tensor([0]).long().to(DEV)
        self.tgt_label = torch.Tensor([1]).long().to(DEV)

        model_path = self.config['solver']['resume']
        if model_path != '':
            print('Resuming Training')
            print(f'Loading Model: {model_path}')

            info_dict = torch.load(model_path)

            print(f"Previous score: {info_dict['valid_score']}")
            self.start_epoch = info_dict['epoch'] + 1

            self.G.load_state_dict(info_dict['state_dict'])
            print('Loading complete')

            if self.config['solver']['resume_optim']:
                print('Loading optim')

                optim_dict = info_dict['g_optim']
                self.optim.load_state_dict(optim_dict)

        self.use_scheduler = False
        if 'scheduler' in self.config['solver']:
            self.use_scheduler = self.config['solver']['scheduler']['use']
            self.scheduler_type = self.config['solver']['scheduler']['type']

            if self.scheduler_type == 'ReduceLROnPlateau':
                patience = self.config['solver']['scheduler']['patience']
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optim,
                        mode = 'min',
                        factor = 0.5,
                        patience = patience,
                        verbose = True)

        alpha_config = self.config['solver']['alpha_scheduler']
        if alpha_config['function'] == 'ramp':
            self.alpha_scheduler = RampScheduler(alpha_config['start_step'],
                                                 alpha_config['end_step'],
                                                 alpha_config['start_value'],
                                                 alpha_config['end_value'])
        elif alpha_config['function'] == 'constant':
            self.alpha_scheduler = ConstantScheduler(alpha_config['value'])
        elif alpha_config['function'] == 'DANN':
            total_step = self.epochs * len(self.wsj0_tr_loader)
            self.alpha_scheduler = DANNScheduler(alpha_config['gamma'],
                                                 alpha_config['scale'],
                                                 total_step)

    def log_meta(self, meta, dset):
        for key in meta:
            value = meta[key]
            name = f'{dset}_{key}'
            self.writer.add_scalar(f'valid/{name}', value, self.valid_time)

    def exec(self):

        self.G.train()
        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):

            # supervised
            self.train_sup_one_epoch(epoch)

            self.G.eval()
            wsj0_meta = self.valid(self.wsj0_cv_loader, self.src_label)
            vctk_meta = self.valid(self.vctk_cv_loader, self.tgt_label)
            self.G.train()

            # Do saving
            self.log_meta(wsj0_meta, 'wsj0')
            self.log_meta(vctk_meta, 'vctk')

            model_name = f'{epoch}.pth'
            valid_score = { 'wsj0': wsj0_meta, 'vctk': vctk_meta }
            info_dict = { 'epoch': epoch, 'valid_score': valid_score }
            info_dict['optim'] = self.optim.state_dict()
            info_dict['D_state_dict'] = self.D.state_dict()

            # TODO, use vctk_loss as save crit
            save_crit = vctk_meta['valid_loss']
            self.saver.update(self.G, save_crit, model_name, info_dict)

            model_name = 'latest.pth'
            self.saver.force_save(self.G, model_name, info_dict)

            self.valid_time += 1

            loss = wsj0_meta['valid_loss']
            if self.use_scheduler:
                if self.scheduler_type == 'ReduceLROnPlateau':
                    self.lr_scheduler.step(loss)

    def train_sup_one_epoch(self, epoch):

        total_snr_loss = 0.
        total_domain_loss = 0.
        total_domain_acc = 0.
        cnt = 0
        for i, sup_sample in enumerate(tqdm(self.wsj0_tr_loader, ncols = NCOL)):

            step = float(i + epoch * len(self.wsj0_tr_loader))
            alpha = self.alpha_scheduler.value(step)

            padded_mixture = sup_sample['mix'].to(DEV)
            padded_source = sup_sample['ref'].to(DEV)
            mixture_lengths = sup_sample['ilens'].to(DEV)

            estimate_source, feature = self.G(padded_mixture)
            domain_out = self.D(feature, alpha)

            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)

            if len(domain_out.size()) == 3:
                domain_out = domain_out.permute(0, 2, 1)
                B, T, C = domain_out.size()
                domain_out = domain_out.contiguous().view(B * T, -1)
            dloss = self.domain_loss(domain_out, self.src_label.expand(domain_out.size(0)))

            sup_loss = loss + dloss

            params = list(self.G.parameters()) + list(self.D.parameters())

            self.optim.zero_grad()
            sup_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

            uns_sample = self.vctk_gen.__next__()
            padded_mixture = uns_sample['mix'].to(DEV)
            _, feature = self.G(padded_mixture)
            uns_domain_out = self.D(feature, alpha)
            if len(uns_domain_out.size()) == 3:
                uns_domain_out = uns_domain_out.permute(0, 2, 1)
                B, T, C = uns_domain_out.size()
                uns_domain_out = uns_domain_out.contiguous().view(B * T, -1)
            uns_loss = self.domain_loss(uns_domain_out, self.tgt_label.expand(uns_domain_out.size(0)))

            self.optim.zero_grad()
            uns_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

            self.optim.step()

            src_acc = (domain_out.argmax(dim = -1) == self.src_label).float().sum()
            tgt_acc = (uns_domain_out.argmax(dim = -1) == self.tgt_label).float().sum()
            cnt += domain_out.size(0) + uns_domain_out.size(0)

            self.writer.add_scalar('train/iter_snr_loss', loss.item(), self.step)
            self.writer.add_scalar('train/iter_domain_loss', dloss.item()+uns_loss.item(), self.step)
            self.writer.add_scalar('train/iter_domain_acc', (src_acc + tgt_acc) / (domain_out.size(0) + uns_domain_out.size(0)), self.step)
            self.step += 1

            total_snr_loss += loss.item()
            total_domain_loss += (dloss.item() + uns_loss.item())
            total_domain_acc += (src_acc + tgt_acc)
            cnt += domain_out.size(0) + uns_domain_out.size(0)

        total_snr_loss = total_snr_loss / len(self.wsj0_tr_loader)
        total_domain_loss = total_domain_loss / len(self.wsj0_tr_loader)
        total_domain_acc = total_domain_acc / cnt

        self.writer.add_scalar('train/epoch_snr_loss', total_snr_loss, epoch)
        self.writer.add_scalar('train/epoch_domain_loss', total_domain_loss, epoch)
        self.writer.add_scalar('train/epoch_domain_acc', total_domain_acc, epoch)

    def valid(self, loader, label):
        total_loss = 0.
        total_snr = 0.
        total_domain_acc = 0.
        cnt = 0

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)

                estimate_source, feature = self.G(padded_mixture)
                domain_out = self.D(feature, 0.)

                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

                total_loss += loss.item()
                total_snr += max_snr.mean().item()

                if len(domain_out.size()) == 3:
                    domain_out = domain_out.permute(0, 2, 1)
                    B, T, C = domain_out.size()
                    domain_out = domain_out.contiguous().view(B * T, -1)

                pred = domain_out.argmax(dim = -1)
                acc = (pred == label).float().sum()
                cnt += pred.size(0)
                total_domain_acc += acc

        total_loss = total_loss / len(loader)
        total_snr = total_snr / len(loader)
        total_domain_acc = total_domain_acc / cnt

        meta = {}
        meta['valid_loss'] = total_loss
        meta['valid_snr'] = total_snr
        meta['total_domain_acc'] = total_domain_acc

        return meta
