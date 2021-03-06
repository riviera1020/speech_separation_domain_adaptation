
import os
import time
import yaml
import datetime

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.solver import Solver
from src.saver import Saver
from src.utils import DEV, DEBUG, NCOL
from src.conv_tasnet import ConvTasNet
from src.pimt_conv_tasnet import PiMtConvTasNet, InputTransform
from src.pit_criterion import cal_loss, SISNR
from src.dataset import wsj0, wsj0_eval
from src.ranger import Ranger
from src.dprnn import DualRNN
from src.evaluation import cal_SDR, cal_SISNRi, cal_SISNR
from src.sep_utils import remove_pad, load_mix_sdr
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

        input_transform = config['solver']['input_transform']
        self.set_transform(input_transform)

        self.step = 0
        self.valid_times = 0

        self.load_data()
        self.set_model()

    def set_transform(self, t_conf):
        self.transform = InputTransform(t_conf)

    def load_data(self):
        # Set training dataset
        dset = 'wsj0'
        if 'dset' in self.config['data']:
            dset = self.config['data']['dset']
        self.dset = dset

        self.load_wsj0_data()
        self.load_vctk_data()
        self.load_libri_data()

        self.dsets = {
                'wsj0': {
                    'tr': self.wsj0_tr_loader,
                    'cv': self.wsj0_cv_loader,
                    },
                'vctk': {
                    'tr': self.vctk_tr_loader,
                    'cv': self.vctk_cv_loader,
                    },
                'libri': {
                    'tr': self.libri_tr_loader,
                    'cv': self.libri_cv_loader,
                    },
                }

    def load_wsj0_data(self):

        seg_len = self.config['data']['segment']
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
                num_workers = self.num_workers)

        devset = wsj0_eval('./data/wsj0/id_list/cv.pkl',
                audio_root = audio_root,
                pre_load = False)
        self.wsj0_cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

    def load_vctk_data(self):

        seg_len = self.config['data']['segment']
        audio_root = self.config['data']['vctk_root']

        trainset = wsj0('./data/vctk/id_list/tr.pkl',
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr')
        self.vctk_tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers)

        devset = wsj0_eval('./data/vctk/id_list/cv.pkl',
                audio_root = audio_root,
                pre_load = False)
        self.vctk_cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

    def load_libri_data(self):

        seg_len = self.config['data']['segment']
        audio_root = self.config['data']['libri_root']

        trainset = wsj0('./data/libri/id_list/tr.pkl',
                audio_root = audio_root,
                seg_len = seg_len,
                pre_load = False,
                one_chunk_in_utt = True,
                mode = 'tr')
        self.libri_tr_loader = DataLoader(trainset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers)

        devset = wsj0_eval('./data/libri/id_list/cv.pkl',
                audio_root = audio_root,
                pre_load = False)
        self.libri_cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

    def set_model(self):
        #self.model = ConvTasNet(self.config['model']).to(DEV)
        self.model = PiMtConvTasNet(self.config['model']).to(DEV)

        # pretrained conf is only for debugging
        pretrained = self.config['solver'].get('pretrained', '')
        if pretrained != '':
            info_dict = torch.load(pretrained)
            self.model.load_state_dict(info_dict['state_dict'])

            print('Load pretrained model')
            if 'epoch' in info_dict:
                print(f"Epochs: {info_dict['epoch']}")
            elif 'step' in info_dict:
                print(f"Steps : {info_dict['step']}")
            print(info_dict['valid_score'])

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
        total_sisnri = 0.
        cnt = 0

        for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):

            padded_mixture = sample['mix'].to(DEV)
            padded_source = sample['ref'].to(DEV)
            mixture_lengths = sample['ilens'].to(DEV)

            estimate_source = self.model.noise_forward(padded_mixture, self.transform)

            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            B = padded_source.size(0)
            total_loss += loss.item() * B
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

        meta = { 'epoch_loss': total_loss,
                 'epoch_sisnri': total_sisnri }
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

        model_name = 'latest.pth'
        self.saver.force_save(self.model, model_name, info_dict)

        if self.use_scheduler:
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.lr_scheduler.step(total_loss)
            #elif self.scheduler_type in [ 'FlatCosine', 'CosineWarmup' ]:
            #    self.lr_scheduler.step(epoch)
