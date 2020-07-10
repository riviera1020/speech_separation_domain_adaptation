import os
import time
import yaml
import json
import random
import datetime
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
from tsnecuda import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

import src.cka as cka
from src.solver import Solver
from src.utils import DEV, DEBUG, NCOL, read_scale
from src.conv_tasnet import ConvTasNet
from src.da_conv_tasnet import DAConvTasNet
from src.slim_conv_tasnet import MyConvTasNet
from src.adanet import ADANet
from src.pit_criterion import cal_loss
from src.dataset import wsj0_eval
from src.wham import wham_eval, wham_parallel_eval
from src.evaluation import cal_SDR, cal_SISNRi
from src.sep_utils import remove_pad, load_mix_sdr
from src.gender_mapper import GenderMapper

import matplotlib.pyplot as plt

class Tester(Solver):
    def __init__(self, config):
        super(Tester, self).__init__(config)

        self.tr_config = config['solver']['train_config']
        self.tr_config = yaml.load(open(self.tr_config), Loader=yaml.FullLoader)

        self.result_dir = config['solver']['result_dir']
        self.safe_mkdir(self.result_dir)
        self.result_name = config['solver'].get('result_name', 'result.json')

        self.checkpoint = config['solver']['checkpoint']

        self.batch_size = 1
        self.num_workers = 4

        save_dict = torch.load(self.checkpoint, map_location=torch.device('cpu'))
        self.epoch = save_dict['epoch']
        self.valid_score = save_dict['valid_score']
        if 'optim' in save_dict:
            self.optim_dict = save_dict['optim']
        else:
            self.optim_dict = save_dict['g_optim']

        state_dict = save_dict['state_dict']
        self.set_model(state_dict)

        self.compute_sdr = config['solver'].get('compute_sdr', True)
        self.g_mapper = GenderMapper()

        self.comp_sim = config['solver'].get('comp_sim', True)
        self.comp_sim_iter = config['solver'].get('comp_sim_iter', 10)

    def load_dset(self, dset):
        # root: wsj0_root, vctk_root, libri_root
        d = 'wsj' if dset == 'wsj0' else dset # stupid error
        if 'wham' in dset:
            # load wham, wham-easy
            return self.load_wham(dset)

        audio_root = self.config['data'][f'{d}_root']
        cv_list = f'./data/{dset}/id_list/cv.pkl'
        tt_list = f'./data/{dset}/id_list/tt.pkl'

        print(f'Load cv from {cv_list}')
        print(f'Load tt from {tt_list}')

        devset = wsj0_eval(cv_list,
                audio_root = audio_root,
                pre_load = False)
        cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

        testset = wsj0_eval(tt_list,
                audio_root = audio_root,
                pre_load = False)
        tt_loader = DataLoader(testset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)
        return cv_loader, tt_loader

    def load_wham(self, dset):
        audio_root = self.config['data'][f'wsj_root']
        cv_list = f'./data/wsj0/id_list/cv.pkl'
        tt_list = f'./data/wsj0/id_list/tt.pkl'

        scale = read_scale(f'./data/{dset}')
        print(f'Load wham data with scale {scale}')

        devset = wham_parallel_eval(cv_list,
                audio_root = audio_root,
                pre_load = False,
                mode = 'cv',
                scale = scale)
        cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

        testset = wham_parallel_eval(tt_list,
                audio_root = audio_root,
                pre_load = False,
                mode = 'tt',
                scale = scale)
        tt_loader = DataLoader(testset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)
        return cv_loader, tt_loader

    def set_model(self, state_dict):
        if 'gen' in self.tr_config['model']:
            mconf = self.tr_config['model']['gen']
        else:
            mconf = self.tr_config['model']

        self.model_type = self.tr_config['model'].get('type', None)
        if self.model_type == 'slim':
            self.model = MyConvTasNet(mconf).to(DEV)
        else:
            self.model = DAConvTasNet(mconf).to(DEV)
        self.model.load_state_dict(state_dict)

    def print_info(self):
        print(f'Epoch: {self.epoch}')

    def exec(self):
        self.print_info()

        self.model.eval()
        dsets = self.config['data']['dsets']

        ds = ', '.join(dsets)
        print(f"Evaluate following datasets: {ds}")

        result_dict = {}

        splts = [ 'cv', 'tt' ]
        gs = [ 'MM', 'FF', 'MF' ]
        sdr_keys = []
        for splt in splts:
            for g in gs:
                sdr_keys.append(f'{splt}_{g}')
        sdr_keys = splts + sdr_keys

        for dset in dsets:
            cv_loader, tt_loader = self.load_dset(dset)
            sdr0 = load_mix_sdr(f'./data/{dset}/mix_sdr/', sdr_keys)

            result_dict[dset] = {}

            if 'wham' not in dset:
                r_cv = self.evaluate(cv_loader, 'cv', dset, sdr0)
                r_tt = self.evaluate(tt_loader, 'tt', dset, sdr0)
            else:
                r_cv = self.evaluate_wham_every_layer(cv_loader, 'cv', dset, sdr0)
                r_tt = self.evaluate_wham_every_layer(tt_loader, 'tt', dset, sdr0)

            result_dict[dset]['cv'] = r_cv
            result_dict[dset]['tt'] = r_tt

        result_dict['tr_config'] = self.tr_config
        rname = os.path.join(self.result_dir, self.result_name)
        json.dump(result_dict, open(rname, 'w'), indent = 1)
        return result_dict

    def evaluate(self, loader, dset, dataset, sdr0):
        total_loss = 0.
        total_SISNRi = 0
        total_SDR = 0
        total_cnt = 0

        gs = [ 'MM', 'FF', 'MF' ]
        gender_SISNRi = { g: 0. for g in gs }
        gender_SDR = { g: 0. for g in gs }
        gender_cnt = { g: 0. for g in gs }

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)
                uids = sample['uid']

                ml = mixture_lengths.max().item()
                padded_mixture = padded_mixture[:, :ml]
                padded_source = padded_source[:, :, :ml]

                estimate_source, feature = self.model.dict_forward(padded_mixture, consider_mask = True)
                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)
                if max_snr.item() < 20:
                    continue

                est_emb = feature['emb']
                est_mask = feature['mask']

                B, emb_size, F, T = est_emb.size()

                one_hot = (est_mask >= 0.99)
                s1_label = one_hot[0, 0, :, :]
                s2_label = one_hot[0, 1, :, :]
                remain = ~(s1_label | s2_label)

                label = torch.zeros(F, T).cuda()
                label.masked_fill_(s1_label, 0)
                label.masked_fill_(s2_label, 1)
                label.masked_fill_(remain, 2)

                emb = est_emb[0].view(emb_size, -1)
                emb = emb.permute(1, 0)
                label = label.view(-1)

                consider_noremain = (label != 2)
                noremain_idx = (consider_noremain == True).nonzero()
                print(noremain_idx.size())

                #emb = torch.gather(emb, dim = 0, index = noremain_idx.expand(-1, 30))
                #label = torch.gather(label, dim = 0, index = noremain_idx.squeeze(-1))

                consider_s1 = (label == 0)
                consider_s1_idx = (consider_s1 == True).nonzero()
                s1_emb = torch.gather(emb, dim = 0, index = consider_s1_idx.expand(-1, 30))
                s1_label = torch.gather(label, dim = 0, index = consider_s1_idx.squeeze())

                consider_s2 = (label == 1)
                consider_s2_idx = (consider_s2 == True).nonzero()
                s2_emb = torch.gather(emb, dim = 0, index = consider_s2_idx.expand(-1, 30))
                s2_label = torch.gather(label, dim = 0, index = consider_s2_idx.squeeze())

                emb = torch.cat([s1_emb, s2_emb], dim = 0)
                label = torch.cat([s1_label, s2_label], dim = 0)

                l = s1_emb.size(0)

                emb = emb.cpu().numpy()
                label = label.cpu().numpy()
                s1_label = s1_label.cpu().numpy()
                s2_label = s2_label.cpu().numpy()

                use_pca = True
                pca_prefix = ''
                if use_pca:
                    pca_prefix = 'pca_'
                    pca = PCA(n_components=10)
                    pca.fit(emb)
                    emb = pca.transform(emb)
                    print(emb.shape)

                #perplexiies = [ 10, 20, 30, 40, 50 ]
                #lrs = [ 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

                perplexiies = [ 40 ]
                lrs = [ 700 ]

                for pers in perplexiies:
                    for lr in lrs:
                        print(pers, lr)
                        tsne_emb = TSNE(n_components = 2, perplexity = pers, learning_rate = lr).fit_transform(emb)
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        s1_tsne = tsne_emb[:l, :]
                        s2_tsne = tsne_emb[l:, :]
                        ax1.scatter(s1_tsne[:,0], s1_tsne[:,1], s = 0.05, c = s1_label, label = 'Source')
                        ax2.scatter(s2_tsne[:,0], s2_tsne[:,1], s = 0.05, c = s2_label, label = 'Target')
                        plt.savefig(f'./plot/tsne/{pca_prefix}sep_test_tsne_per{pers}_lr{lr}.png')
                        plt.close()
                        x = tsne_emb[:, 0]
                        y = tsne_emb[:, 1]
                        plt.figure()
                        scatter = plt.scatter(x, y, s = 0.05, c = label, alpha=0.5)
                        legned = plt.legend(*scatter.legend_elements(), loc="lower left")
                        plt.savefig(f'./plot/tsne/new_{pca_prefix}test_tsne_per{pers}_lr{lr}.png')
                        plt.close()
                exit()


                #one_hot = (idm >= 0.95)
                one_hot = (est_mask >= 0.99)
                # b,XC
                print(est_mask.sum(1))
                print(one_hot.size())
                consider_s1 = torch.masked_select(feat, one_hot[:, 0, :, :])
                print(consider_s1)
                exit()
                consider_s2 = torch.masked_select(feat, one_hot[:, 1, :, :])

                consider_s1 = consider_s1.cpu().numpy()
                consider_s2 = consider_s2.cpu().numpy()

                plt.hist(consider_s1, bins = 100, color = 'b', alpha = 0.5)
                plt.hist(consider_s2, bins = 100, color = 'r', alpha = 0.5)
                #plt.tight_layout()
                plt.show()

                estimate_source, feature = self.model.dict_forward(padded_mixture, consider_mask = True)
                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

                print(max_snr)
                exit()

                total_loss += loss.item()

                B = reorder_estimate_source.size(0)
                total_cnt += B

                padded_mixture = remove_pad(padded_mixture, mixture_lengths)
                padded_source = remove_pad(padded_source, mixture_lengths)
                reorder_estimate_source = remove_pad(reorder_estimate_source, mixture_lengths)

                for b in range(B):
                    mix = padded_mixture[b]
                    src_ref = padded_source[b]
                    src_est = reorder_estimate_source[b]
                    uid = uids[b]

                    g = self.g_mapper(uid, dataset)
                    gender_cnt[g] += 1

                    sisnri = cal_SISNRi(src_ref, src_est, mix)
                    total_SISNRi += sisnri
                    gender_SISNRi[g] += sisnri

                    if self.compute_sdr:
                        sdr = cal_SDR(src_ref, src_est)
                        total_SDR += sdr
                        gender_SDR[g] += sdr

        total_loss /= total_cnt
        total_SISNRi /= total_cnt

        if self.compute_sdr:
            total_SDR /= total_cnt
            total_SDRi = total_SDR - sdr0[dset]
        else:
            total_SDRi = 0

        gender_SDRi = {}
        for g in gender_SISNRi:
            gender_SISNRi[g] /= gender_cnt[g]
            if self.compute_sdr:
                sdr = gender_SDR[g] / gender_cnt[g]
                gender_SDRi[g] = sdr - sdr0[f'{dset}_{g}']
            else:
                gender_SDRi[g] = 0.

        result = { 'total_loss': total_loss, 'total_SDRi': total_SDRi, 'total_SISNRi': total_SISNRi,
                   'gender_SDRi': gender_SDRi, 'gender_SISNRi': gender_SISNRi }
        return result
