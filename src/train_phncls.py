
import os
import time
import yaml
import math
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchaudio.transforms import MFCC, Spectrogram

from src.phn_cls import get_model, FeatExt
from src.solver import Solver
from src.saver import Saver
from src.utils import DEV, DEBUG, NCOL, inf_data_gen
from src.conv_tasnet import ConvTasNet
from src.pit_criterion import cal_loss
from src.phn_dataset import PhnSepDataset
from src.phone_mapper import PhoneMapper
from src.dashboard import Dashboard
from src.vat import VAT, EntMin
from src.ranger import Ranger
from src.pimt_conv_tasnet import AddNoise
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

        # crit: phn acc
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

        self.sample_rate = config['data']['sample_rate']
        self.n_mfcc = config['solver']['n_mfcc']
        self.mel_args = config['solver']['mel_args']
        self.win_length = self.mel_args['win_length']
        self.hop_length = self.mel_args['hop_length']

        self.semi_conf = config['solver'].get('semi', {'use': False})
        self.semi = self.semi_conf['use']
        if self.semi:
            self.methods = {}
            self.semi_dset = self.semi_conf['dset']

        self.step = 0
        self.valid_times = 0

        self.load_data()
        self.set_model()

    def load_data(self):

        # Set training dataset
        dset = 'wsj0'
        if 'dset' in self.config['data']:
            dset = self.config['data']['dset']
        self.dset = dset

        self.vocab_size = 40
        self.load_wsj0_data()
        self.load_vctk_data()

        self.dsets = {
                'wsj0': {
                    'tr': self.wsj0_tr_loader,
                    'cv': self.wsj0_cv_loader,
                    },
                'vctk': {
                    'tr': self.vctk_tr_loader,
                    'cv': self.vctk_cv_loader,
                    },
                }

    def load_wsj0_data(self):

        seg_len = self.config['data']['segment']
        audio_root = self.config['data']['wsj_root']

        phone_mapper = PhoneMapper(f'./data/wsj0/phn.pkl',
                self.sample_rate,
                'wsj0',
                self.win_length,
                self.hop_length)

        trainset = PhnSepDataset('./data/wsj0/fa_id_list/tr.pkl',
                audio_root = audio_root,
                phone_mapper = phone_mapper,
                sample_rate = self.sample_rate,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr')
        self.wsj0_tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers)

        devset = PhnSepDataset('./data/wsj0/fa_id_list/cv.pkl',
                audio_root = audio_root,
                phone_mapper = phone_mapper,
                sample_rate = self.sample_rate,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = False,
                mode = 'cv')
        self.wsj0_cv_loader = DataLoader(devset,
                batch_size = 1,
                shuffle = False,
                num_workers = self.num_workers)

    def load_vctk_data(self):

        seg_len = self.config['data']['segment']
        audio_root = self.config['data']['vctk_root']
        phone_mapper = PhoneMapper(f'./data/vctk/phn.pkl',
                self.sample_rate,
                'vctk',
                self.win_length,
                self.hop_length)

        trainset = PhnSepDataset('./data/vctk/fa_id_list/tr.pkl',
                audio_root = audio_root,
                phone_mapper = phone_mapper,
                sample_rate = self.sample_rate,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr')
        self.vctk_tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers)

        devset = PhnSepDataset('./data/vctk/fa_id_list/cv.pkl',
                audio_root = audio_root,
                phone_mapper = phone_mapper,
                sample_rate = self.sample_rate,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = False,
                mode = 'cv')
        self.vctk_cv_loader = DataLoader(devset,
                batch_size = 1,
                shuffle = False,
                num_workers = self.num_workers)

    def set_model(self):

        self.feat_ext = FeatExt(
                sample_rate = self.sample_rate,
                n_mfcc = self.n_mfcc,
                mel_args = self.mel_args,
                w = 9).to(DEV)

        self.model = get_model(self.config['model']['phn_cls'], self.vocab_size).to(DEV)

        self.cross_entropy = nn.CrossEntropyLoss().to(DEV)

        if self.semi:
            methods = self.semi_conf['methods']

            # Vat config
            conf = methods.get('vat', {'use': False})
            if conf['use']:
                self.methods['vat'] = conf
                self.VATLoss = VAT(
                        xi = conf['xi'],
                        eps = conf['eps'],
                        num_iters = 1)

                self.vat_lambda = conf['lambda']

            # Ent config
            conf = methods.get('ent', {'use': False})
            if conf['use']:
                self.methods['ent'] = conf
                self.EntLoss = EntMin()
                self.ent_lambda = conf['lambda']

            # Pi config
            conf = methods.get('pi', {'use': False})
            if conf['use']:
                self.methods['pi'] = conf
                self.pi_lambda = conf['lambda']
                self.warmup = conf['warmup']
                self.add_noise = AddNoise({'scale': conf['scale']})

            if 'pi' in self.methods and ('vat' in self.methods or 'ent' in self.methods):
                print('No pi + vat/ent Now')
                exit()

        '''
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
        '''

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
                print('Loading complete')

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
        elif optim_type == 'adadelta':
            rho = self.config['optim']['rho']
            eps = self.config['optim']['eps']
            self.opt = torch.optim.Adadelta(
                    self.model.parameters(),
                    lr = lr,
                    rho = rho,
                    eps = eps,
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

    def exec(self):

        if self.semi:
            self.semi_supervise()
        else:
            self.supervise()

    def semi_supervise(self):

        semi_gen = inf_data_gen(self.dsets[self.semi_dset]['tr'])
        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):

            if 'pi' in self.methods:
                self.train_pi(epoch, self.dsets[self.dset]['tr'], semi_gen)
            else:
                self.train_semi(epoch, self.dsets[self.dset]['tr'], semi_gen)

            # Valid training dataset
            self.valid(self.dsets[self.dset]['cv'], epoch, prefix = self.dset)

            # Valid not training dataset
            for dset in self.dsets:
                if dset != self.dset:
                    self.valid(self.dsets[dset]['cv'], epoch, no_save = True, prefix = dset)

            self.writer.epoch()

    def supervise(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs), ncols = NCOL):

            self.train_one_epoch(epoch, self.dsets[self.dset]['tr'])

            # Valid training dataset
            self.valid(self.dsets[self.dset]['cv'], epoch, prefix = self.dset)

            # Valid not training dataset
            for dset in self.dsets:
                if dset != self.dset:
                    self.valid(self.dsets[dset]['cv'], epoch, no_save = True, prefix = dset)

            self.writer.epoch()

    def train_one_epoch(self, epoch, tr_loader):

        self.model.train()
        total_loss = 0.
        cnt = 0

        total_acc = 0.
        num = 0

        total_sil = 0.
        dset_sil = 0.

        for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):

            padded_source = sample['ref'].to(DEV)
            phn_label = sample['ref_phns'].to(DEV)

            B, C, T = padded_source.size()
            batch_size = B
            padded_source = padded_source.view(B * C, T)

            phn_label = phn_label.view(B * C, -1)

            feat = self.feat_ext(padded_source)
            logits = self.model(feat)

            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            phn_label = phn_label.view(B * T)
            loss = self.cross_entropy(logits, phn_label)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            total_loss += (loss.item() * batch_size)
            cnt += batch_size

            p = F.softmax(logits).argmax(dim = -1)
            acc = (p == phn_label).int()
            total_acc += acc.sum()
            num += acc.numel()

            acc = (p == 39).int()
            total_sil += acc.sum()

            acc = (phn_label == 39).int()
            dset_sil += acc.sum()

            self.step += 1

        total_loss = total_loss / cnt
        total_acc = total_acc / num
        total_sil = total_sil / num
        dset_sil = dset_sil / num
        print(f'Epoch {epoch} Loss: {total_loss}')
        print(f'Epoch {epoch} Acc : {total_acc}')
        print(f'Epoch {epoch} Sil : {total_sil}')
        print(f'Epoch {epoch} Dset Sil : {dset_sil}')

        meta = {
                'total_loss': total_loss,
                'total_acc': total_acc,
                'total_sil': total_sil,
                'dset_sil': dset_sil }
        self.writer.log_epoch_info('train', meta)

    def train_semi(self, epoch, tr_loader, semi_gen):

        self.model.train()
        total_loss = 0.
        total_vat = 0.
        total_ent = 0.
        cnt = 0

        total_acc = 0.
        num = 0

        total_sil = 0.
        dset_sil = 0.

        for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):

            loss = 0.

            semi_sample = semi_gen.__next__()
            semi_wav = semi_sample['ref'].to(DEV)

            B, C, T = semi_wav.size()
            semi_wav = semi_wav.view(B * C, T)
            semi_feat = self.feat_ext(semi_wav)

            if 'vat' in self.methods:
                vat_loss = self.VATLoss(self.model, semi_feat)

                total_vat += vat_loss.item() * B
                loss += self.vat_lambda * vat_loss

            if 'ent' in self.methods:
                ent_loss = self.EntLoss(semi_feat)

                total_ent += ent_loss.item() * B
                loss += self.ent_lambda * ent_loss

            padded_source = sample['ref'].to(DEV)
            phn_label = sample['ref_phns'].to(DEV)

            B, C, T = padded_source.size()
            batch_size = B
            padded_source = padded_source.view(B * C, T)

            phn_label = phn_label.view(B * C, -1)

            feat = self.feat_ext(padded_source)
            logits = self.model(feat)

            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            phn_label = phn_label.view(B * T)
            loss += self.cross_entropy(logits, phn_label)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            total_loss += (loss.item() * batch_size)
            cnt += batch_size

            p = F.softmax(logits).argmax(dim = -1)
            acc = (p == phn_label).int()
            total_acc += acc.sum()
            num += acc.numel()

            acc = (p == 39).int()
            total_sil += acc.sum()

            acc = (phn_label == 39).int()
            dset_sil += acc.sum()

            self.step += 1

        total_loss = total_loss / cnt
        total_vat = total_vat / cnt
        total_ent = total_ent / cnt
        total_acc = total_acc / num
        total_sil = total_sil / num
        dset_sil = dset_sil / num
        print(f'Epoch {epoch} Loss: {total_loss}')
        print(f'Epoch {epoch} Vat : {total_vat}')
        print(f'Epoch {epoch} Ent : {total_ent}')
        print(f'Epoch {epoch} Acc : {total_acc}')
        print(f'Epoch {epoch} Sil : {total_sil}')
        print(f'Epoch {epoch} Dset Sil : {dset_sil}')

        meta = {
                'total_loss': total_loss,
                'total_vat': total_vat,
                'total_ent': total_ent,
                'total_acc': total_acc,
                'total_sil': total_sil,
                'dset_sil': dset_sil }
        self.writer.log_epoch_info('train', meta)

    def train_pi(self, epoch, tr_loader, semi_gen):

        self.model.train()
        total_loss = 0.
        total_pi = 0.
        cnt = 0

        total_acc = 0.
        num = 0

        total_sil = 0.
        dset_sil = 0.

        for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):

            loss = 0.

            semi_sample = semi_gen.__next__()
            semi_wav = semi_sample['ref'].to(DEV)

            B, C, T = semi_wav.size()

            with torch.no_grad():
                semi_wav = semi_wav.view(B * C, T)
                semi_feat = self.feat_ext(semi_wav)
                semi_logits = self.model(semi_feat)

            noise_wav = self.add_noise(semi_wav)
            noise_feat = self.feat_ext(noise_wav)

            noise_logits = self.model(noise_feat)

            semi_dis = F.softmax(semi_logits, dim = -1)
            noise_dis = F.softmax(noise_logits, dim = -1)
            loss_pi = ((semi_dis - noise_dis) ** 2).sum(-1).mean()
            #loss_pi = ((semi_logits - noise_logits) ** 2).sum(-1).mean()
            total_pi = loss_pi.item() * B

            coef = self.pi_lambda * math.exp(-5 * (1 - min(float(self.step)/self.warmup, 1)) ** 2)
            loss += loss_pi * coef

            padded_source = sample['ref'].to(DEV)
            phn_label = sample['ref_phns'].to(DEV)

            B, C, T = padded_source.size()
            batch_size = B
            padded_source = padded_source.view(B * C, T)

            phn_label = phn_label.view(B * C, -1)

            feat = self.feat_ext(padded_source)
            logits = self.model(feat)

            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            phn_label = phn_label.view(B * T)
            loss += self.cross_entropy(logits, phn_label)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            total_loss += (loss.item() * batch_size)
            cnt += batch_size

            p = F.softmax(logits).argmax(dim = -1)
            acc = (p == phn_label).int()
            total_acc += acc.sum()
            num += acc.numel()

            acc = (p == 39).int()
            total_sil += acc.sum()

            acc = (phn_label == 39).int()
            dset_sil += acc.sum()

            self.step += 1

        total_loss = total_loss / cnt
        total_pi = total_pi / cnt
        total_acc = total_acc / num
        total_sil = total_sil / num
        dset_sil = dset_sil / num
        print(f'Epoch {epoch} Loss: {total_loss}')
        print(f'Epoch {epoch} Pi : {total_pi}')
        print(f'Epoch {epoch} Acc : {total_acc}')
        print(f'Epoch {epoch} Sil : {total_sil}')
        print(f'Epoch {epoch} Dset Sil : {dset_sil}')

        meta = {
                'total_loss': total_loss,
                'total_pi': total_pi,
                'total_acc': total_acc,
                'total_sil': total_sil,
                'dset_sil': dset_sil }
        self.writer.log_epoch_info('train', meta)

    def valid(self, loader, epoch, no_save = False, prefix = ""):
        self.model.eval()
        total_loss = 0.
        cnt = 0

        total_acc = 0.
        num = 0

        total_sil = 0.
        dset_sil = 0.

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):
                padded_source = sample['ref'].to(DEV)
                phn_label = sample['ref_phns'].to(DEV)

                B, C, T = padded_source.size()
                batch_size = B
                padded_source = padded_source.view(B * C, T)

                phn_label = phn_label.view(B * C, -1)

                feat = self.feat_ext(padded_source)
                logits = self.model(feat)

                B, T, C = logits.size()
                logits = logits.view(B * T, C)
                phn_label = phn_label.view(B * T)
                loss = self.cross_entropy(logits, phn_label)

                total_loss += (loss.item() * batch_size)
                cnt += batch_size

                p = F.softmax(logits).argmax(dim = -1)
                acc = (p == phn_label).int()
                total_acc += acc.sum()
                num += acc.numel()

                acc = (p == 39).int()
                total_sil += acc.sum()

                acc = (phn_label == 39).int()
                dset_sil += acc.sum()

        total_loss = total_loss / cnt
        total_acc = total_acc / num
        total_sil = total_sil / num
        dset_sil = dset_sil / num
        print('Validation: ')
        print(f'\t{prefix} Loss: {total_loss}')
        print(f'\t{prefix} Acc : {total_acc}')
        print(f'\t{prefix} Sil : {total_sil}')

        valid_score = {}
        valid_score['valid_loss'] = total_loss
        valid_score['valid_acc'] = total_acc

        meta = {
                'total_loss': total_loss,
                'total_acc': total_acc,
                'total_sil': total_sil,
                'dset_sil': dset_sil }
        self.writer.log_epoch_info(f'valid_{prefix}', meta)

        if no_save:
            return

        model_name = f'{epoch}.pth'
        info_dict = { 'epoch': epoch, 'valid_score': valid_score, 'config': self.config }
        info_dict['optim'] = self.opt.state_dict()

        self.saver.update(self.model, total_acc, model_name, info_dict)

        model_name = 'latest.pth'
        self.saver.force_save(self.model, model_name, info_dict)

        if self.use_scheduler:
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.lr_scheduler.step(total_loss)
            #elif self.scheduler_type in [ 'FlatCosine', 'CosineWarmup' ]:
            #    self.lr_scheduler.step(epoch)
