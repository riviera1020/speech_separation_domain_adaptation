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
from torch.utils.data import DataLoader

import src.cka as cka
from src.solver import Solver
from src.utils import DEV, DEBUG, NCOL, read_scale
from src.conv_tasnet import ConvTasNet
from src.da_conv_tasnet import DAConvTasNet
from src.adanet import ADANet
from src.pit_criterion import cal_loss
from src.dataset import wsj0_eval
from src.wham import wham_eval, wham_parallel_eval
from src.evaluation import cal_SDR, cal_SISNRi
from src.sep_utils import remove_pad, load_mix_sdr
from src.gender_mapper import GenderMapper

import sys
sys.path.append('../svcca')
import cca_core
import pwcca

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

    def compute_L2(self, cf, nf):
        distance = ((cf - nf) ** 2)
        distance = distance.sum().sqrt() / distance.numel()
        return distance.item()

    def compute_L1(self, cf, nf):
        distance = (cf - nf).abs().mean()
        return distance.item()

    def compute_svcca(self, cf, nf):
        '''
        cf: [F, data_points]
        nf: [F, data_points]
        '''
        sim = cca_core.svcca(cf, nf, keep_dims = 20)
        return sim

    def compute_pwcca(self, cf, nf):
        '''
        cf: [F, data_points]
        nf: [F, data_points]
        '''
        pw, w, c = pwcca.compute_pwcca(cf, nf, epsilon = 1e-10)
        return pw

    def compute_cka(self, cf, nf):
        '''
        cf: [F, data_points]
        nf: [F, data_points]
        '''
        sim = cka.cka(cka.gram_linear(cf.T), cka.gram_linear(nf.T))
        return sim

    def compute_cca(self, cf, nf):
        sim = cka.cca(cka.cca(cf.T), cka.cca(nf.T))
        return sim

    def compute_cos_sim(self, cf, nf):
        '''
        cf: [F, T]
        nf: [F, T]
        '''
        sim = F.cosine_similarity(cf, nf, dim = 0)
        sim = sim.mean().item()
        return sim

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

                estimate_source, _ = self.model(padded_mixture)

                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

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

    def evaluate_wham_every_layer(self, loader, dset, dataset, sdr0):
        total_loss = 0.
        total_SISNRi = 0.
        total_SDR = 0.
        total_cnt = 0.
        total_L2_dis = 0.

        gs = [ 'MM', 'FF', 'MF' ]
        gender_SISNRi = { g: 0. for g in gs }
        gender_SDR = { g: 0. for g in gs }
        gender_L2_dis = { g: 0. for g in gs }
        gender_cnt = { g: 0. for g in gs }

        lkeys = list(range(32)) + [ 'enc' ]
        total_layer_L2_dis = { k: 0. for k in lkeys }
        total_layer_L1_dis = { k: 0. for k in lkeys }
        total_layer_cos_sim = { k: 0. for k in lkeys }

        total_layer_ckas = { k:[] for k in lkeys }
        total_layer_pwccas = { k:[] for k in lkeys }
        total_layer_clean_act = { k:[] for k in lkeys }
        total_layer_noisy_act = { k:[] for k in lkeys }
        total_layer_cka_mean = { k:0 for k in lkeys }
        total_layer_cka_std = { k:0 for k in lkeys }
        total_layer_pwcca_mean = { k:0 for k in lkeys }
        total_layer_pwcca_std = { k:0 for k in lkeys }

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                noisy_mix = sample['noisy_mix'].to(DEV)
                clean_mix = sample['clean_mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)
                uids = sample['uid']

                ml = mixture_lengths.max().item()
                clean_mix = clean_mix[:, :ml]
                noisy_mix = noisy_mix[:, :ml]
                padded_source = padded_source[:, :, :ml]

                est_clean_source, clean_feat = self.model.dict_forward(clean_mix)
                est_noisy_source, noisy_feat = self.model.dict_forward(noisy_mix)

                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, est_noisy_source, mixture_lengths)
                total_loss += loss.item()

                B = reorder_estimate_source.size(0)
                total_cnt += B

                noisy_mix = remove_pad(noisy_mix, mixture_lengths)
                padded_source = remove_pad(padded_source, mixture_lengths)
                reorder_estimate_source = remove_pad(reorder_estimate_source, mixture_lengths)

                for b in range(B):
                    mix = noisy_mix[b]
                    src_ref = padded_source[b]
                    src_est = reorder_estimate_source[b]
                    uid = uids[b]

                    g = self.g_mapper(uid, dataset)
                    gender_cnt[g] += 1

                    for k in lkeys:
                        cf = clean_feat[k][b]
                        nf = noisy_feat[k][b]
                        l2_d = self.compute_L2(cf, nf)
                        total_layer_L2_dis[k] += l2_d

                        l1_d = self.compute_L1(cf, nf)
                        total_layer_L1_dis[k] += l1_d

                        F, T = cf.size()

                        ridx = torch.randint(0, T, size = (1, self.comp_sim_iter)).to(DEV).expand(F, -1)
                        cf_select = torch.gather(cf, dim=1, index=ridx)
                        nf_select = torch.gather(nf, dim=1, index=ridx)

                        total_layer_clean_act[k].append(cf_select.cpu())
                        total_layer_noisy_act[k].append(nf_select.cpu())

                        total_layer_cos_sim[k] += self.compute_cos_sim(cf, nf)

                    sisnri = cal_SISNRi(src_ref, src_est, mix)
                    total_SISNRi += sisnri
                    gender_SISNRi[g] += sisnri

                    if self.compute_sdr:
                        sdr = cal_SDR(src_ref, src_est)
                        total_SDR += sdr
                        gender_SDR[g] += sdr

        total_loss /= total_cnt
        total_SISNRi /= total_cnt
        total_L2_dis /= total_cnt

        if self.compute_sdr:
            total_SDR /= total_cnt
            total_SDRi = total_SDR - sdr0[dset]
        else:
            total_SDRi = 0

        for k in tqdm(lkeys, ncols = NCOL):
            # mean L1/L2 distance, cos sim
            total_layer_L2_dis[k] /= total_cnt
            total_layer_L1_dis[k] /= total_cnt
            total_layer_cos_sim[k] /= total_cnt

            if not self.comp_sim:
                break

            # compute cka/cca sim
            cf_iters = torch.stack(total_layer_clean_act[k], dim = 2).numpy()
            nf_iters = torch.stack(total_layer_noisy_act[k], dim = 2).numpy()

            for it in range(self.comp_sim_iter):
                cf = cf_iters[:, it, :]
                nf = nf_iters[:, it, :]

                cka_sim = self.compute_cka(cf, nf)
                total_layer_ckas[k].append(cka_sim)

                pwcca_sim = self.compute_pwcca(cf, nf)
                total_layer_pwccas[k].append(pwcca_sim)

            total_layer_cka_mean[k] = float(np.mean(total_layer_ckas[k]))
            total_layer_pwcca_mean[k] = float(np.mean(total_layer_pwccas[k]))
            total_layer_cka_std[k] = float(np.std(total_layer_ckas[k]))
            total_layer_pwcca_std[k] = float(np.std(total_layer_pwccas[k]))

        gender_SDRi = {}
        for g in gender_SISNRi:
            gender_SISNRi[g] /= gender_cnt[g]
            gender_L2_dis[g] /= gender_cnt[g]

            if self.compute_sdr:
                sdr = gender_SDR[g] / gender_cnt[g]
                gender_SDRi[g] = sdr - sdr0[f'{dset}_{g}']
            else:
                gender_SDRi[g] = 0.

        result = { 'total_loss': total_loss, 'total_SDRi': total_SDRi, 'total_SISNRi': total_SISNRi, 'total_L2_dis': total_L2_dis,
                'gender_SDRi': gender_SDRi, 'gender_SISNRi': gender_SISNRi, 'gender_L2_dis': gender_L2_dis,
                'total_layer_L2_dis': total_layer_L2_dis, 'total_layer_L1_dis': total_layer_L1_dis, 'total_layer_cos_sim': total_layer_cos_sim,
                'total_layer_pwcca': total_layer_pwcca_mean, 'total_layer_pwcca_std': total_layer_pwcca_std,
                'total_layer_cka': total_layer_cka_mean, 'total_layer_cka_std': total_layer_cka_std }

        return result
