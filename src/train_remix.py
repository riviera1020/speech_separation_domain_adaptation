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
from src.pit_criterion import cal_loss, cal_norm, SISNR
from src.dataset import wsj0, wsj0_eval
from src.discriminator import RWD
from src.wham import wham, wham_eval
from src.gender_dset import wsj0_gender
from src.MSD import MultiScaleDiscriminator
from src.scheduler import RampScheduler, ConstantScheduler
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
        self.Lc_lambda = config['solver']['Lc_lambda']
        #self.Le_lambda = config['solver']['Le_lambda']

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
        self.steps_per_epoch = len(self.sup_loader)

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
                                 sch_config['end_value'],
                                 now_step = self.step,
                                 steps_per_epoch = self.steps_per_epoch)
        elif sch_config['function'] == 'constant':
            return ConstantScheduler(sch_config['value'])

    def set_model(self):
        self.G = ConvTasNet(self.config['model']['gen']).to(DEV)

        dis_type = self.config['model']['dis']['type']
        if dis_type == 'RWD':
            self.D = RWD(self.config['model']['dis']).to(DEV)
        elif dis_type == 'MSD':
            self.D = MultiScaleDiscriminator(self.config['model']['dis']).to(DEV)

        self.g_optim = self.set_optim([self.G], self.config['g_optim'])
        self.d_optim = self.set_optim([self.D], self.config['d_optim'])

        if self.adv_loss == 'gan':
            self.bce_loss = nn.BCEWithLogitsLoss().to(DEV)

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
            self.start_epoch = info_dict['epoch'] + 1
            self.step = self.steps_per_epoch * self.start_epoch

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
        #TODO
        """
        1. train jointly from start
        2. pretrained on wsj0 first, then jointly on vctk
        3. EMA (how? no teacher
        4. separatly trained on vctk(?
        """
        self.train_jointly()

    def train_jointly(self):
        self.G.train()
        for step in tqdm(range(0, self.pretrain_d_step), ncols = NCOL):
            self.train_dis_once(step, self.uns_gen, pretrain = True)
            self.writer.step()
        self.writer.set_step(1)

        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):

            # supervised
            self.train_sup_one_epoch(epoch, self.sup_loader)

            valid_score = {}
            if not self.gender_exp:
                # Valid sup training dataset
                sup_score = self.valid(self.dsets[self.dset]['cv'], epoch, prefix = self.dset)
                valid_score[self.dset] = sup_score

                # Valid uns training dataset
                uns_score = self.valid(self.dsets[self.uns_dset]['cv'], epoch, no_save = True, prefix = self.uns_dset)
                valid_score[self.uns_dset] = uns_score

                # Use Uns score for model selection ( Cheat )
                save_crit = uns_score['valid_sisnri']
            else:
                sup_score = self.gender_valid(self.dsets[self.main_dset]['cv'], epoch, prefix = self.main_dset)
                valid_score[self.main_dset] = sup_score
                save_crit = sup_score['valid_gender_sisnri'][self.uns_gender]

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
            result = self.run_tester('test_baseline.py', conf)
            result['tt_config'] = conf
            self.writer.log_result(result, 'best.json')

            conf = self.construct_test_conf(dsets = 'all', sdir = 'chapter4', choose_best = False, compute_sdr = False)
            result = self.run_tester('test_baseline.py', conf)
            result['tt_config'] = conf
            self.writer.log_result(result, 'result.json')

    def train_sup_one_epoch(self, step, tr_loader):
        self.G.train()
        total_loss = 0.
        total_sisnri = 0.
        total_uns_sisnri = 0.
        cnt = 0

        for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):
            padded_mixture = sample['mix'].to(DEV)
            padded_source = sample['ref'].to(DEV)
            mixture_lengths = sample['ilens'].to(DEV)

            estimate_source = self.G(padded_mixture)

            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)

            self.g_optim.zero_grad()
            self.d_optim.zero_grad()
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
            self.train_dis_once(self.step, self.uns_gen)
            self.train_gen_once(self.step, self.uns_gen)

            with torch.no_grad():
                uns_sample = self.uns_gen.__next__()
                padded_mixture = uns_sample['mix'].to(DEV)
                padded_source = uns_sample['ref'].to(DEV)
                mixture_lengths = uns_sample['ilens'].to(DEV)

                estimate_source = self.G(padded_mixture)
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

    def train_dis_once(self, step, data_gen, pretrain = False):
        # assert batch_size is even

        if pretrain:
            prefix = 'pretrain_'
        else:
            prefix = ''

        total_d_loss = 0.
        weighted_d_loss = 0.
        total_gp = 0.
        total_grad_norm = 0.

        fake_acc = 0.
        real_acc = 0.
        fake_cnt = 0.
        real_cnt = 0.
        for _ in range(self.d_iters):

            # fake sample
            sample = data_gen.__next__()
            padded_mixture = sample['mix'].to(DEV)
            estimate_source = self.G(padded_mixture)

            with torch.no_grad():
                y1, y2 = torch.chunk(estimate_source, 2, dim = 0)

                # random mix
                if random.random() < 0.5:
                    y2 = y2.flip(dims = [1])

                remix = y1 + y2
                remix = remix.view(-1, remix.size(-1))

            d_fake_loss = 0.
            d_fakes = self.D(remix)
            if self.adv_loss == 'wgan-gp':
                for d_fake in d_fakes:
                    d_fake_loss += d_fake.mean()
            elif self.adv_loss == 'hinge':
                for d_fake in d_fakes:
                    d_fake_loss += F.relu(1.0 + d_fake).mean()
            elif self.adv_loss == 'gan':
                for d_fake in d_fakes:
                    d_fake_loss += self.bce_loss(d_fake,
                                                 torch.zeros_like(d_fake))
            elif self.adv_loss == 'lsgan':
                for d_fake in d_fakes:
                    d_fake_loss += (d_fake ** 2).mean()

            if self.adv_loss == 'hinge' or self.adv_loss == 'gan':
                with torch.no_grad():
                    for d_fake in d_fakes:
                        if self.adv_loss == 'gan':
                            fp = ((F.sigmoid(d_fake) >= 0.5).float() == 0).float()
                        if self.adv_loss == 'hinge':
                            fp = ((d_fake >= 0).float() == 0).float()
                        fake_acc += fp.sum().item()
                        fake_cnt += fp.numel()

            # true sample
            sample = data_gen.__next__()
            padded_mixture = sample['mix'].to(DEV)

            d_real_loss = 0.
            d_reals = self.D(padded_mixture)
            if self.adv_loss == 'wgan-gp':
                for d_real in d_reals:
                    d_real_loss += (- d_real.mean())
            elif self.adv_loss == 'hinge':
                for d_real in d_reals:
                    d_real_loss += F.relu(1.0 - d_real).mean()
            elif self.adv_loss == 'gan':
                for d_real in d_reals:
                    d_real_loss += self.bce_loss(d_real,
                                                 torch.ones_like(d_real))
            elif self.adv_loss == 'lsgan':
                for d_real in d_reals:
                    d_real_loss += ((d_real - 1) ** 2).mean()

            if self.adv_loss == 'hinge' or self.adv_loss == 'gan':
                with torch.no_grad():
                    for d_real in d_reals:
                        if self.adv_loss == 'gan':
                            rp = ((F.sigmoid(d_real) >= 0.5).float() == 1).float()
                        if self.adv_loss == 'hinge':
                            rp = ((d_real >= 0).float() == 1).float()
                        real_acc += rp.sum().item()
                        real_cnt += rp.numel()

            d_loss = d_real_loss + d_fake_loss

            if self.adv_loss == 'wgan-gp':
                gp = calc_gradient_penalty(self.D, remix, padded_mixture)
                _d_loss = d_loss + self.gp_lambda * gp
                total_gp += gp.item()
            else:
                _d_loss = d_loss

            d_lambda = self.Ld_scheduler.value(step)
            _d_loss = d_lambda * d_loss

            self.g_optim.zero_grad()
            self.d_optim.zero_grad()
            _d_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.D_grad_clip)
            total_grad_norm += grad_norm
            if math.isnan(grad_norm):
                print('Error : grad norm is NaN @ step '+str(step))
            else:
                self.d_optim.step()

            total_d_loss += d_loss.item()
            weighted_d_loss += _d_loss.item()

        total_d_loss /= self.d_iters
        weighted_d_loss /= self.d_iters
        total_gp /= self.d_iters
        total_grad_norm /= self.d_iters
        meta = { f'{prefix}d_loss': total_d_loss,
                 f'{prefix}dis_gradient_penalty': total_gp,
                 f'{prefix}dis_grad_norm': total_grad_norm }

        if self.adv_loss == 'gan' or self.adv_loss == 'hinge':
            rf_acc = (real_acc + fake_acc) / (real_cnt + fake_cnt)
            real_acc /= real_cnt
            fake_acc /= fake_cnt
            meta[f'{prefix}dis_acc'] = rf_acc
            meta[f'{prefix}dis_fake_acc'] = fake_acc
            meta[f'{prefix}dis_real_acc'] = real_acc

        self.writer.log_step_info('train', meta)

    def train_gen_once(self, step, data_gen):
        # Only remain gan now

        total_g_loss = 0.
        total_gan_loss = 0.
        total_Le = 0.
        total_Lc = 0.
        total_grad_norm = 0.
        real_acc = 0.
        real_cnt = 0.
        for _ in range(self.g_iters):

            sample = data_gen.__next__()
            padded_mixture = sample['mix'].to(DEV)
            T = padded_mixture.size(-1)

            estimate_source = self.G(padded_mixture)
            y1, y2 = torch.chunk(estimate_source, 2, dim = 0)

            if random.random() < 0.5:
                y2 = y2.flip(dims = [1])
            remix = (y1 + y2).view(-1, T)
            g_fakes = self.D(remix)

            g_loss = 0.

            if self.adv_loss == 'wgan-gp' or 'hinge':
                for g_fake in g_fakes:
                    g_loss += (- g_fake.mean())
            elif self.adv_loss == 'gan':
                for g_fake in g_fakes:
                    g_loss += self.bce_loss(g_fake,
                                            torch.ones_like(g_fake))
            elif self.adv_loss == 'lsgan':
                for g_fake in g_fakes:
                    g_loss += ((g_fake - 1) ** 2).mean()

            if self.adv_loss == 'hinge' or self.adv_loss == 'gan':
                with torch.no_grad():
                    for g_fake in g_fakes:
                        if self.adv_loss == 'gan':
                            rp = ((F.sigmoid(g_fake) >= 0.5).float() == 1).float()
                        if self.adv_loss == 'hinge':
                            rp = ((g_fake >= 0).float() == 1).float()
                        real_acc += rp.sum().item()
                        real_cnt += rp.numel()

            # TODO, better Le?
            #Le = (padded_mixture.unsqueeze(1) * estimate_source).sum(dim = -1)
            #Le = (Le ** 2).sum(dim = -1).mean()

            Lc = 0.
            if self.Lc_lambda > 0:
                estimate_source = self.G(remix)
                y1, y2 = torch.chunk(estimate_source, 2, dim = 0)

                reremix1 = (y1 + y2).view(-1, T)
                reremix2 = (y1 + y2.flip(dims = [1])).view(-1, T)

                Lc = cal_norm(padded_mixture, reremix1, reremix2)

            g_lambda = self.Lg_scheduler.value(step)
            _g_loss = g_lambda * (g_loss + self.Lc_lambda * Lc)

            self.g_optim.zero_grad()
            self.d_optim.zero_grad()
            _g_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.G_grad_clip)
            total_grad_norm += grad_norm
            if math.isnan(grad_norm):
                print('Error : grad norm is NaN @ step '+str(step))
            else:
                self.g_optim.step()

            total_g_loss += _g_loss.item()
            total_gan_loss += g_loss.item()
            #total_Le += Le.item()
            if self.Lc_lambda > 0:
                total_Lc += Lc.item()

        total_g_loss /= self.g_iters
        total_gan_loss /= self.g_iters
        total_Lc /= self.g_iters
        total_Le /= self.g_iters
        total_grad_norm /= self.g_iters
        meta = { 'total_gen_loss': total_gan_loss,
                 'total_g_loss': total_g_loss,
                 'total_Lc': total_Lc,
                 'total_gen_grad_norm': total_grad_norm }

        if self.adv_loss == 'gan' or self.adv_loss == 'hinge':
            real_acc /= real_cnt
            meta['gen_real_acc'] = real_acc
        self.writer.log_step_info('train', meta)

    def valid(self, loader, epoch, no_save = False, prefix = ""):
        self.G.eval()
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

                estimate_source = self.G(padded_mixture)

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
        return valid_score

    def gender_valid(self, loader, epoch, no_save = False, prefix = ""):
        dset = prefix
        self.G.eval()
        total_loss = 0.
        total_sisnri = 0.
        cnt = 0

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

                estimate_source = self.G(padded_mixture)

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
        for g in genders:
            gender_sisnri[g] /= gender_cnt[g]
            meta[f'{prefix}_epoch_{g}_sisnri'] = gender_sisnri[g]

        self.writer.log_epoch_info('valid', meta)

        valid_score = {}
        valid_score['valid_loss'] = total_loss
        valid_score['valid_sisnri'] = total_sisnri
        valid_score['valid_gender_sisnri'] = gender_sisnri
        return valid_score
