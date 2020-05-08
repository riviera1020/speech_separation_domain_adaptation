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
from src.gender_dset import wsj0_gender
from src.scheduler import RampScheduler, ConstantScheduler, DANNScheduler
from src.gradient_penalty import calc_gradient_penalty
from src.dashboard import Dashboard
from src.ranger import Ranger
from src.gender_mapper import GenderMapper

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
        self.D_grad_clip = config['solver']['D_grad_clip']
        self.G_grad_clip = config['solver']['G_grad_clip']
        self.num_workers = config['solver']['num_workers']
        self.step = 0
        self.pretrain_d_step = config['solver'].get('pretrain_d_step', 0)

        self.g_iters = config['solver']['g_iters']
        self.d_iters = config['solver']['d_iters']

        self.adv_loss = config['solver']['adv_loss']
        self.gp_lambda = config['solver']['gp_lambda']

        self.load_data()
        self.set_model()
        self.gender_mapper = GenderMapper()

        self.script_name = os.path.basename(__file__).split('.')[0].split('_')[-1]
        self.writer.add_tag(self.script_name)

    def is_gender_dset(self, dset):
        if '-' in dset:
            g = dset.split('-')[1]
            if g in [ 'MF', 'MM', 'FF' ]:
                return True
        return False

    def load_data(self):
        # Set training dataset
        dset = 'wsj0'
        if 'dset' in self.config['data']:
            dset = self.config['data']['dset']
        self.dset = dset
        self.uns_dset = self.config['data'].get('uns_dset', 'vctk')

        print(f'Sup: {self.dset}')
        print(f'Uns: {self.uns_dset}')

        self.gender_exp = False
        if self.is_gender_dset(self.dset):
            print('Run WSJ0 Gender Exp')
            d, sup_g = self.dset.split('-')
            _, uns_g = self.uns_dset.split('-')
            self.gender_exp = True
            self.main_dset = d
            self.sup_gender = sup_g
            self.uns_gender = uns_g

        # Load loader for sup training
        seg_len = self.config['data']['segment']
        self.sup_loader = self.load_tr_dset(self.dset, seg_len)

        # Load data gen for gan training
        uns_len = self.config['data'].get('uns_segment', 2.0)
        self.sup_gen = inf_data_gen(self.load_tr_dset(self.dset, uns_len))
        self.uns_gen = inf_data_gen(self.load_tr_dset(self.uns_dset, uns_len))

        # Load cv loader
        self.dsets = {}
        if not self.gender_exp:
            cv_dsets = [ self.dset, self.uns_dset ]
        else:
            d, _ = self.dset.split('-')
            cv_dsets = [ d ]
        for d in cv_dsets:
            self.dsets[d] = { 'cv': self.load_cv_dset(d) }

    def load_tr_dset(self, dset, seg_len):
        # root: wsj0_root, vctk_root, libri_root
        d = 'wsj' if dset == 'wsj0' else dset # stupid error
        if 'wham' in dset:
            return self.load_wham(dset, seg_len, 'tr')
        if self.is_gender_dset(dset):
            dset, g = dset.split('-')
            return self.load_tr_gender_dset(dset, seg_len, g)

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
                num_workers = self.num_workers,
                drop_last = True)
        return tr_loader

    def load_tr_gender_dset(self, dset, seg_len, gender):
        """
        dset: only wsj0 now
        """
        assert dset == 'wsj0'
        d = 'wsj' if dset == 'wsj0' else dset # stupid error
        audio_root = self.config['data'][f'{d}_root']
        tr_list = f'./data/{dset}/id_list/tr.pkl'
        trainset = wsj0_gender(tr_list,
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr',
                gender = gender)
        tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                drop_last = True)
        return tr_loader

    def load_cv_dset(self, dset):
        # root: wsj0_root, vctk_root, libri_root
        d = 'wsj' if dset == 'wsj0' else dset # stupid error
        if 'wham' in dset:
            return self.load_wham(dset, seg_len = -1, mode = 'cv')

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
                    num_workers = self.num_workers,
                    drop_last = True)
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

        self.G = DAConvTasNet(self.config['model']['gen']).to(DEV)
        self.D = DomainClassifier(self.G.feature_dim, self.config['model']['domain_cls']).to(DEV)

        self.g_optim = self.set_optim([self.G], self.config['g_optim'])
        self.d_optim = self.set_optim([self.D], self.config['d_optim'])

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.src_label = torch.FloatTensor([0]).to(DEV)
        self.tgt_label = torch.FloatTensor([1]).to(DEV)

        pretrained = self.config['solver']['pretrained']
        if pretrained != '':
            info_dict = torch.load(pretrained)
            self.G.load_state_dict(info_dict['state_dict'])

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

            self.G.load_state_dict(info_dict['state_dict'])
            self.D.load_state_dict(info_dict['D_state_dict'])

            print('Loading complete')

            if self.config['solver']['resume_optim']:
                print('Loading optim')

                optim_dict = info_dict['g_optim']
                self.g_optim.load_state_dict(optim_dict)

                optim_dict = info_dict['d_optim']
                self.d_optim.load_state_dict(optim_dict)

            # dashboard is one-base
            self.writer.set_epoch(self.start_epoch + 1)
            self.writer.set_step(self.step + 1)

        self.Lg_scheduler = self.set_scheduler(self.config['solver']['Lg_scheduler'])
        self.Ld_scheduler = self.set_scheduler(self.config['solver']['Ld_scheduler'])

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

    def exec(self):
        self.G.train()
        for step in tqdm(range(0, self.pretrain_d_step), ncols = NCOL):
            self.train_dis_once(step, self.sup_gen, self.uns_gen, pretrain = True)
            self.writer.step()
        self.writer.set_step(1)

        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):

            # supervised
            self.train_one_epoch(epoch, self.sup_loader)

            valid_score = {}
            if not self.gender_exp:
                # Valid sup training dataset
                sup_score = self.valid(self.dsets[self.dset]['cv'], epoch, prefix = self.dset, label = self.src_label)
                valid_score[self.dset] = sup_score

                # Valid uns training dataset
                uns_score = self.valid(self.dsets[self.uns_dset]['cv'], epoch, no_save = True, prefix = self.uns_dset, label = self.tgt_label)
                valid_score[self.uns_dset] = uns_score

                # Use Uns score for model selection ( Cheat )
                save_crit = uns_score['valid_sisnri']
            else:
                sup_score = self.gender_valid(self.dsets[self.main_dset]['cv'], epoch, prefix = self.main_dset)
                valid_score[self.main_dset] = sup_score
                save_crit = sup_score['valid_gender_sisnri'][self.uns_gender]

            # Valid not training dataset
            #for dset in self.dsets:
            #    if dset != self.dset and dset != self.uns_dset:
            #        s = self.valid(self.dsets[dset]['cv'], epoch, no_save = True, prefix = dset)
            #        valid_score[dset] = s

            if self.use_scheduler:
                if self.scheduler_type == 'ReduceLROnPlateau':
                    self.lr_scheduler.step(sup_score['valid_loss'])

            model_name = f'{epoch}.pth'
            info_dict = { 'epoch': epoch, 'step': self.step, 'valid_score': valid_score, 'config': self.config }
            info_dict['g_optim'] = self.g_optim.state_dict()
            info_dict['d_optim'] = self.d_optim.state_dict()
            info_dict['D_state_dict'] = self.D.state_dict()

            self.saver.update(self.G, save_crit, model_name, info_dict)

            model_name = 'latest.pth'
            self.saver.force_save(self.G, model_name, info_dict)
            self.writer.epoch()

        if self.test_after_finished:
            conf = self.construct_test_conf(dsets = 'all', sdir = 'chapter4', choose_best = True, compute_sdr = False)
            result = self.run_tester('test_dagan.py', conf)
            result['tt_config'] = conf
            self.writer.log_result(result, 'best.json')

            conf = self.construct_test_conf(dsets = 'all', sdir = 'chapter4', choose_best = False, compute_sdr = False)
            result = self.run_tester('test_dagan.py', conf)
            result['tt_config'] = conf
            self.writer.log_result(result, 'result.json')

    def train_one_epoch(self, epoch, tr_loader):
        self.G.train()
        total_loss = 0.
        total_sisnri = 0.
        total_uns_sisnri = 0.
        cnt = 0

        for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):

            padded_mixture = sample['mix'].to(DEV)
            padded_source = sample['ref'].to(DEV)
            mixture_lengths = sample['ilens'].to(DEV)

            estimate_source, _ = self.G(padded_mixture)

            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)

            self.D.zero_grad()
            self.G.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.G_grad_clip)
            self.g_optim.step()

            B = padded_source.size(0)
            total_loss += loss.item() * B
            cnt += B
            with torch.no_grad():
                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                total_sisnri += (max_snr - mix_sisnr).sum()

            meta = { 'iter_loss': loss.item() }
            self.writer.log_step_info('train', meta)

            # semi part
            self.train_dis_once(self.step, self.sup_gen, self.uns_gen)
            self.train_gen_once(self.step, self.sup_gen, self.uns_gen)

            with torch.no_grad():
                uns_sample = self.uns_gen.__next__()
                padded_mixture = uns_sample['mix'].to(DEV)
                padded_source = uns_sample['ref'].to(DEV)
                mixture_lengths = uns_sample['ilens'].to(DEV)

                estimate_source, _ = self.G(padded_mixture)
                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)
                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                total_uns_sisnri += (max_snr - mix_sisnr).sum()

            self.step += 1
            self.writer.step()

        total_loss = total_loss / cnt
        total_sisnri = total_sisnri / cnt
        total_uns_sisnri = total_uns_sisnri / cnt

        meta = { 'epoch_loss': total_loss,
                 'epoch_sisnri': total_sisnri,
                 'epoch_uns_sisnri': total_uns_sisnri }
        self.writer.log_epoch_info('train', meta)

    def train_dis_once(self, step, src_gen, tgt_gen, pretrain = False):
        # assert batch_size is even

        if pretrain:
            prefix = 'pretrain_'
        else:
            prefix = ''

        total_d_loss = 0.
        weighted_d_loss = 0.
        total_gp = 0.
        src_domain_acc = 0.
        tgt_domain_acc = 0.
        total_grad_norm = 0.
        src_cnt = 0
        tgt_cnt = 0
        for _ in range(self.d_iters):

            # fake(src) sample
            sample = src_gen.__next__()
            src_mixture = sample['mix'].to(DEV)

            with torch.no_grad():
                _, src_feat = self.G(src_mixture)

            if self.adv_loss == 'wgan-gp':
                d_fake_loss = self.D(src_feat).mean()
            elif self.adv_loss == 'gan':
                d_fake_out = self.D(src_feat)
                d_fake_loss = self.bce_loss(d_fake_out,
                                            self.src_label.expand_as(d_fake_out))
                with torch.no_grad():
                    src_dp = ((F.sigmoid(d_fake_out) >= 0.5).float() == self.src_label).float()
                    src_domain_acc += src_dp.sum().item()
                    src_cnt += src_dp.numel()
            elif self.adv_loss == 'hinge':
                d_fake_out = self.D(src_feat)
                d_fake_loss = F.relu(d_fake_out).mean()
                with torch.no_grad():
                    src_dp = ((F.sigmoid(d_fake_out) >= 0.5).float() == self.src_label).float()
                    src_domain_acc += src_dp.sum().item()
                    src_cnt += src_dp.numel()

            # true(tgt) sample
            sample = tgt_gen.__next__()
            tgt_mixture = sample['mix'].to(DEV)

            with torch.no_grad():
                _, tgt_feat = self.G(tgt_mixture)

            if self.adv_loss == 'wgan-gp':
                d_real_loss = - self.D(tgt_feat).mean()
            elif self.adv_loss == 'gan':
                d_real_out = self.D(tgt_feat)
                d_real_loss = self.bce_loss(d_real_out,
                                            self.tgt_label.expand_as(d_real_out))
                with torch.no_grad():
                    tgt_dp = ((F.sigmoid(d_real_out) >= 0.5).float() == self.tgt_label).float()
                    tgt_domain_acc += tgt_dp.sum().item()
                    tgt_cnt += tgt_dp.numel()
            elif self.adv_loss == 'hinge':
                d_real_out = self.D(tgt_feat)
                d_real_loss = F.relu(1.0 - d_real_out).mean()
                with torch.no_grad():
                    tgt_dp = ((F.sigmoid(d_real_out) >= 0.5).float() == self.tgt_label).float()
                    tgt_domain_acc += tgt_dp.sum().item()
                    tgt_cnt += tgt_dp.numel()

            d_loss = d_real_loss + d_fake_loss

            if self.adv_loss == 'wgan-gp':
                gp = calc_gradient_penalty(self.D, tgt_feat, src_feat)
                d_lambda = self.Ld_scheduler.value(step)
                d_loss = d_loss + self.gp_lambda * gp
                total_gp += gp.item()

            if pretrain:
                d_lambda = 1
            else:
                d_lambda = self.Ld_scheduler.value(step)
            _d_loss = d_lambda * d_loss

            total_d_loss += d_loss.item()
            weighted_d_loss += _d_loss.item()

            self.D.zero_grad()
            self.G.zero_grad()
            _d_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.D_grad_clip)
            total_grad_norm += grad_norm
            if math.isnan(grad_norm):
                print('Error : grad norm is NaN @ step '+str(step))
            else:
                self.d_optim.step()

        total_d_loss /= self.d_iters
        weighted_d_loss /= self.d_iters
        total_gp /= self.d_iters
        total_grad_norm /= self.d_iters
        meta = { f'{prefix}d_loss': total_d_loss,
                 f'{prefix}gradient_penalty': total_gp,
                 f'{prefix}grad_norm': total_grad_norm }
        if self.adv_loss == 'gan' or self.adv_loss == 'hinge':
            domain_acc = (src_domain_acc + tgt_domain_acc) / (src_cnt + tgt_cnt)
            src_domain_acc /= src_cnt
            tgt_domain_acc /= tgt_cnt
            meta[f'{prefix}dis_src_domain_acc'] = src_domain_acc
            meta[f'{prefix}dis_tgt_domain_acc'] = tgt_domain_acc
            meta[f'{prefix}dis_domain_acc'] = domain_acc

        self.writer.log_step_info('train', meta)

    def train_gen_once(self, step, src_gen, tgt_gen):
        # Only remain gan now

        total_g_loss = 0.
        weighted_g_loss = 0.
        domain_acc = 0.
        src_domain_acc = 0.
        tgt_domain_acc = 0.
        total_grad_norm = 0.
        cnt = 0
        for _ in range(self.g_iters):

            # fake(src) sample
            sample = src_gen.__next__()
            src_mixture = sample['mix'].to(DEV)

            _, src_feat = self.G(src_mixture)

            if self.adv_loss == 'wgan-gp':
                g_fake_loss = - self.D(src_feat).mean()
            elif self.adv_loss == 'gan':
                g_fake_out = self.D(src_feat)
                g_fake_loss = self.bce_loss(g_fake_out,
                                            self.tgt_label.expand_as(g_fake_out))
                with torch.no_grad():
                    src_dp = ((F.sigmoid(g_fake_out) >= 0.5).float() == self.src_label).float()
                    domain_acc += src_dp.sum().item()
                    cnt += src_dp.numel()
            elif self.adv_loss == 'hinge':
                g_fake_out = self.D(src_feat)
                g_fake_loss = - g_fake_out.mean()

            # true(tgt) sample
            sample = tgt_gen.__next__()
            tgt_mixture = sample['mix'].to(DEV)

            _, tgt_feat = self.G(tgt_mixture)

            if self.adv_loss == 'wgan-gp':
                g_real_loss = self.D(tgt_feat).mean()
            elif self.adv_loss == 'gan':
                g_real_out = self.D(tgt_feat)
                g_real_loss = self.bce_loss(g_real_out,
                                            self.src_label.expand_as(g_real_out))
                with torch.no_grad():
                    tgt_dp = ((F.sigmoid(g_real_out) >= 0.5).float() == self.tgt_label).float()
                    domain_acc += tgt_dp.sum().item()
                    cnt += tgt_dp.numel()
            elif self.adv_loss == 'hinge':
                g_real_out = self.D(tgt_feat)
                g_real_loss = g_real_out.mean()

            g_loss = g_real_loss + g_fake_loss
            g_lambda = self.Lg_scheduler.value(step)
            _g_loss = g_loss * g_lambda

            self.D.zero_grad()
            self.G.zero_grad()
            _g_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.G_grad_clip)
            total_grad_norm += grad_norm
            if math.isnan(grad_norm):
                print('Error : grad norm is NaN @ step '+str(step))
            else:
                self.g_optim.step()

            total_g_loss += g_loss.item()
            weighted_g_loss += _g_loss.item()

        total_g_loss /= self.g_iters
        weighted_g_loss /= self.g_iters
        total_grad_norm /= self.g_iters
        meta = { 'g_loss': total_g_loss,
                 'weighted_g_loss': weighted_g_loss }
        if self.adv_loss == 'gan':
            domain_acc = domain_acc / cnt
            meta['gen_domain_acc'] = domain_acc
        self.writer.log_step_info('train', meta)

    def valid(self, loader, epoch, no_save = False, prefix = "", label = None):
        self.G.eval()
        total_loss = 0.
        total_sisnri = 0.
        domain_acc = 0.
        cnt = 0
        dcnt = 0

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)

                ml = mixture_lengths.max().item()
                padded_mixture = padded_mixture[:, :ml]
                padded_source = padded_source[:, :, :ml]
                B = padded_source.size(0)

                estimate_source, feature = self.G(padded_mixture)

                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                max_sisnri = (max_snr - mix_sisnr)

                if self.adv_loss != 'wgan-gp' and label != None:
                    dp = (F.sigmoid(self.D(feature)) >= 0.5).float()
                    dcnt += dp.numel()
                    acc_num = (dp == label).sum().item()
                    domain_acc += float(acc_num)

                total_loss += loss.item() * B
                total_sisnri += max_sisnri.sum().item()
                cnt += B

        total_sisnri = total_sisnri / cnt
        total_loss = total_loss / cnt
        if self.adv_loss != 'wgan-gp' and label != None:
            domain_acc = domain_acc / dcnt

        meta = { f'{prefix}_epoch_loss': total_loss,
                 f'{prefix}_epoch_sisnri': total_sisnri,
                 f'{prefix}_epoch_domain_acc': domain_acc }
        self.writer.log_epoch_info('valid', meta)

        valid_score = {}
        valid_score['valid_loss'] = total_loss
        valid_score['valid_sisnri'] = total_sisnri
        valid_score['valid_domain_acc'] = domain_acc
        return valid_score

    def gender_valid(self, loader, epoch, no_save = False, prefix = ""):
        def get_label(g):
            if g == self.sup_gender:
                return self.src_label
            elif g == self.uns_gender:
                return self.tgt_label
            return -1

        dset = prefix

        self.G.eval()
        total_loss = 0.
        total_sisnri = 0.
        domain_acc = 0.
        cnt = 0
        dcnt = 0

        genders = [ 'MF', 'MM', 'FF' ]
        gender_sisnri = { 'MF': 0., 'FF': 0., 'MM': 0, }
        gender_cnt = { 'MF': 0., 'FF': 0., 'MM': 0, }

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)
                uids = sample['uid']

                ml = mixture_lengths.max().item()
                padded_mixture = padded_mixture[:, :ml]
                padded_source = padded_source[:, :, :ml]
                B = padded_source.size(0)

                estimate_source, feature = self.G(padded_mixture)

                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

                mix_sisnr = SISNR(padded_source, padded_mixture, mixture_lengths)
                max_sisnri = (max_snr - mix_sisnr)

                if self.adv_loss != 'wgan-gp':
                    dp = (F.sigmoid(self.D(feature)) >= 0.5).float()

                total_loss += loss.item() * B
                total_sisnri += max_sisnri.sum().item()
                cnt += B

                for b in range(B):
                    g = self.gender_mapper(uids[b], dset)
                    gender_sisnri[g] += max_sisnri[b].item()
                    gender_cnt[g] += 1

                    if self.adv_loss != 'wgan-gp':
                        dp_b = dp[b]
                        label = get_label(g)

                        if label != -1:
                            dcnt = dp_b.numel()
                            acc_num = (dp_b == label).sum().item()
                            domain_acc += float(acc_num)

        total_sisnri = total_sisnri / cnt
        total_loss = total_loss / cnt
        if self.adv_loss != 'wgan-gp' and label != None:
            domain_acc = domain_acc / dcnt

        meta = { f'{prefix}_epoch_loss': total_loss,
                 f'{prefix}_epoch_sisnri': total_sisnri,
                 f'{prefix}_epoch_domain_acc': domain_acc }

        for g in genders:
            gender_sisnri[g] /= gender_cnt[g]
            meta[f'{prefix}_epoch_{g}_sisnri'] = gender_sisnri[g]

        self.writer.log_epoch_info('valid', meta)

        valid_score = {}
        valid_score['valid_loss'] = total_loss
        valid_score['valid_sisnri'] = total_sisnri
        valid_score['valid_domain_acc'] = domain_acc
        valid_score['valid_gender_sisnri'] = gender_sisnri
        return valid_score
