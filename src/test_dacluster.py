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
import matplotlib.patches as mpatches
plt.style.use('seaborn-colorblind')

class Tester(Solver):
    def __init__(self, config):
        super(Tester, self).__init__(config)

        self.batch_size = 1
        self.num_workers = 4

        self.result_dir = config['solver']['result_dir']
        self.safe_mkdir(self.result_dir)
        self.result_name = config['solver'].get('result_name', 'result.json')
        self.result_name = os.path.join(self.result_dir, self.result_name)

        self.baseline = self.set_model(config['solver']['baseline'])
        self.comp = self.set_model(config['solver']['compare'])
        self.source = self.config['data']['source']
        self.target = self.config['data']['target']
        self.load_dset()

        self.compute_sdr = config['solver'].get('compute_sdr', True)
        self.g_mapper = GenderMapper()

        self.pca_components = config['solver'].get('pca_components', 0)

        self.low = config['solver'].get('low', -10)
        self.high = config['solver'].get('high', 100)
        self.frame_num = config['solver'].get('frame_num', 100000)
        self.recompute = config['solver'].get('recompute', False)

        self.layers = config['solver'].get('layers', 'all')
        if self.layers == 'all':
            self.layers = [ 'enc' ] + list(range(32))

        self.splts = config['solver'].get('splts', [ 'cv', 'tt' ])
        self.gender = config['solver'].get('gender', [ 'MF', 'MM', 'FF', 'all' ])

        self.st_parallel = config['solver'].get('st_parallel', False)
        self.bc_parallel = config['solver'].get('st_parallel', False)
        if self.st_parallel and self.target == 'vctk':
            print('VCTK is not parallel to any dset')
            exit()

        self.perplexiies = [ 40 ]
        self.lrs = [ 700 ]

    def load_dset(self):

        self.dsets = [ self.source, self.target ]

        self.datasets = {}
        for dset in self.dsets:
            if 'wham' in dset:
                devset, cv_loader, testset, tt_loader = self.load_wham(dset)
            else:
                devset, cv_loader, testset, tt_loader = self.load_data(dset)
            self.datasets[dset] = {
                    'cv': devset,
                    'cv_loader': cv_loader,
                    'tt': testset,
                    'tt_loader': tt_loader }

    def load_data(self, dset):
        d = 'wsj' if dset == 'wsj0' else dset # stupid error
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
        return devset, cv_loader, testset, tt_loader

    def load_wham(self, dset):
        audio_root = self.config['data'][f'wsj_root']
        cv_list = f'./data/wsj0/id_list/cv.pkl'
        tt_list = f'./data/wsj0/id_list/tt.pkl'

        scale = read_scale(f'./data/{dset}')
        print(f'Load wham data with scale {scale}')

        devset = wham_eval(cv_list,
                audio_root = audio_root,
                pre_load = False,
                mode = 'cv',
                scale = scale)
        cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

        testset = wham_eval(tt_list,
                audio_root = audio_root,
                pre_load = False,
                mode = 'tt',
                scale = scale)
        tt_loader = DataLoader(testset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)
        return devset, cv_loader, testset, tt_loader

    def set_model(self, conf):
        tr_config = conf['train_config']
        checkpoint = conf['checkpoint']
        if tr_config == '' or checkpoint == '':
            return None
        print(f'Load: {checkpoint}')

        tr_config = yaml.load(open(tr_config), Loader=yaml.FullLoader)
        save_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        epoch = save_dict['epoch']
        valid_score = save_dict['valid_score']
        state_dict = save_dict['state_dict']

        if 'gen' in tr_config['model']:
            mconf = tr_config['model']['gen']
        else:
            mconf = tr_config['model']

        model_type = tr_config['model'].get('type', None)
        if model_type == 'slim':
            model = MyConvTasNet(mconf).to(DEV)
        else:
            model = DAConvTasNet(mconf).to(DEV)
        model.load_state_dict(state_dict)
        return model

    def exec(self):
        rname = self.result_name
        if (not os.path.isfile(rname)) or self.recompute:
            result = self.compute_result()
        else:
            result = json.load(open(rname))

        source_baseline = self.filter_data(self.source, result['baseline'][self.source], low = self.low, high = self.high)
        target_baseline = self.filter_data(self.target, result['baseline'][self.target], low = self.low, high = self.high)

        pbar = tqdm(total = len(self.splts) * len(self.gender) )
        for splt in self.splts:
            for gender in self.gender:
                self.plot_st(source_baseline, target_baseline, splt = splt, gender = gender, frame_num = self.frame_num,
                        st_parallel = self.st_parallel, bc_parallel = self.bc_parallel)
                pbar.update(1)
        pbar.close()

    def filter_data(self, dataset, result, low = -10, high = 1000):
        ret = { 'cv': [], 'tt': [] }
        for splt in result:
            uids = {'MF': [], 'MM': [], 'FF': [], 'all': [] }
            for uid in result[splt]:
                sisnri, sdr = result[splt][uid]
                g = self.g_mapper(uid, dataset)
                if low <= sisnri <= high:
                    uids[g].append(uid)
                    uids['all'].append(uid)
            ret[splt] = uids
        return ret

    def compute_result(self):
        dsets = self.dsets
        ds = ', '.join(dsets)
        print(f"Evaluate following datasets: {ds}")

        models = { 'baseline': self.baseline }
        if self.comp is not None:
            models['comp'] = self.comp
        # Get utt performance, first
        result_dict = {}
        for mkey, model in models.items():
            result_dict[mkey] = {}
            for dset in dsets:
                cv_loader = self.datasets[dset]['cv_loader']
                tt_loader = self.datasets[dset]['tt_loader']

                r_cv = self.evaluate(model, cv_loader, 'cv', dset)
                r_tt = self.evaluate(model, tt_loader, 'tt', dset)

                result_dict[mkey][dset] = {}
                result_dict[mkey][dset]['cv'] = r_cv
                result_dict[mkey][dset]['tt'] = r_tt
        rname = os.path.join(self.result_dir, 'result.json')
        json.dump(result_dict, open(rname, 'w'), indent = 1)
        return result_dict

    def evaluate(self, model, loader, dset, dataset):
        model.eval()
        ret = {}
        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)
                uids = sample['uid']

                ml = mixture_lengths.max().item()
                padded_mixture = padded_mixture[:, :ml]
                padded_source = padded_source[:, :, :ml]
                B = padded_mixture.size(0)

                estimate_source, feature = model(padded_mixture)
                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)

                padded_mixture = remove_pad(padded_mixture, mixture_lengths)
                padded_source = remove_pad(padded_source, mixture_lengths)
                reorder_estimate_source = remove_pad(reorder_estimate_source, mixture_lengths)

                for b in range(B):
                    mix = padded_mixture[b]
                    src_ref = padded_source[b]
                    src_est = reorder_estimate_source[b]
                    uid = uids[b]
                    sisnri = cal_SISNRi(src_ref, src_est, mix)
                    sdr = cal_SDR(src_ref, src_est) if self.compute_sdr else 0
                    ret[uid] = [ sisnri, sdr ]
        return ret

    def gather_feature(self, model, uids, dataset, splt, gender, layers, frame_num, spk_num):
        """
        Args:
            uids: dict or list of tuple
                if dict         : gather frame_num based on spk_num, if spk number in uids < spk_num, use all
                if list of tuple: [ (uid, frame_ids), ... ]
        """
        ret = { l:[] for l in layers }
        cnt = 0

        frame_num_utt = frame_num // spk_num

        ret_uids = []
        if isinstance(uids, dict):
            uids = uids[splt][gender]
            random.shuffle(uids)
            uids = uids[:spk_num]
            uids = [ (u, None) for u in uids ]

        with torch.no_grad():
            for uid, frame_idx in tqdm(uids):
                sample = self.datasets[dataset][splt].get_sample_by_uid(uid)

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)
                uids = sample['uid']

                ml = mixture_lengths.max().item()
                padded_mixture = padded_mixture[:, :ml]
                padded_source = padded_source[:, :, :ml]
                B = padded_mixture.size(0)

                estimate_source, feature = model.dict_forward(padded_mixture)

                if frame_idx == None:
                    F, T = feature[0][0].size()
                    if T <= frame_num_utt:
                        frame_idx = list(range(T))
                    else:
                        frame_idx = random.sample(list(range(T)), frame_num_utt)
                        frame_idx.sort()

                for l in layers:
                    fidx = torch.LongTensor(frame_idx).to(DEV)
                    f = torch.index_select(feature[l][0], 1, fidx)
                    ret[l].append(f.cpu().numpy())

                ret_uids.append((uid, frame_idx))

        for l in ret:
            cat_tensor = np.concatenate(ret[l], axis = 1)
            cat_tensor = cat_tensor.T
            ret[l] = cat_tensor
        return ret, ret_uids

    def plot_st(self, source_uids, target_uids, splt, gender, frame_num = 100000, st_parallel = False, bc_parallel = True):
        """
        st_parallel: use same utt for source, target
        bc_parallel: use same utt for baseline, comp
        """
        layers = self.layers
        spk_num = 20

        # Gather draw feature
        baseline_source_features, bs_uids = self.gather_feature(self.baseline, source_uids, self.source, splt, gender, layers, frame_num, spk_num)
        if st_parallel:
            bt_uids = bs_uids
        else:
            bt_uids = target_uids
        baseline_target_features, bt_uids = self.gather_feature(self.baseline, bt_uids, self.target, splt, gender, layers, frame_num, spk_num)
        if bc_parallel:
            cs_uids = bs_uids
            ct_uids = bt_uids
        else:
            cs_uids = source_uids
            ct_uids = target_uids
        comp_source_features, cs_uids = self.gather_feature(self.comp, source_uids, self.source, splt, gender, layers, frame_num, spk_num)
        if st_parallel:
            ct_uids = cs_uids
        comp_target_features, ct_uids = self.gather_feature(self.comp, target_uids, self.target, splt, gender, layers, frame_num, spk_num)

        # Plot
        for l in layers:
            print(f'Layer: {l}')
            b_sfs = baseline_source_features[l]
            b_tfs = baseline_target_features[l]
            c_sfs = comp_source_features[l]
            c_tfs = comp_target_features[l]

            emb = np.concatenate([ b_sfs, b_tfs, c_sfs, c_tfs ], axis = 0)
            b_s_label = np.ones(b_sfs.shape[0]) * 0
            b_t_label = np.ones(b_tfs.shape[0]) * 1
            c_s_label = np.ones(c_sfs.shape[0]) * 0
            c_t_label = np.ones(c_tfs.shape[0]) * 1

            b_idx = b_sfs.shape[0] + b_tfs.shape[0]

            b_st_idx = b_sfs.shape[0]
            c_st_idx = c_sfs.shape[0]

            b_label = np.concatenate([b_s_label, b_t_label], axis = 0)
            c_label = np.concatenate([c_s_label, c_t_label], axis = 0)

            pca_prefix = ''
            if self.pca_components > 0:
                print('Perform pca')
                pca_prefix = 'pca_'
                pca = PCA(n_components=self.pca_components)
                pca.fit(emb)
                emb = pca.transform(emb)
                #print(pca.explained_variance_ratio_)
                print(sum(pca.explained_variance_ratio_))

            def plot_scatter(tsne_emb, label, fig_path, idx):
                scale = 0.5
                source = tsne_emb[:idx, :]
                target = tsne_emb[idx:, :]

                plt.figure(dpi = 1000)
                sx = source[:, 0]
                sy = source[:, 1]
                blabel = label[:idx]
                scatter = plt.scatter(sx, sy, s = scale, lw = 0, color = 'C0', alpha=0.75, label = 'Source')

                tx = target[:, 0]
                ty = target[:, 1]
                tlabel = label[idx:]
                scatter = plt.scatter(tx, ty, s = scale, lw = 0, color = 'C2', alpha=0.75, label = 'Target')

                plt.legend(markerscale=10*scale)
                plt.savefig(fig_path)
                plt.close()


            prefix = f'{splt}_{gender}_layer{l}_'

            perplexiies = self.perplexiies
            lrs = self.lrs
            pbar = tqdm(total = len(perplexiies) * len(lrs))
            for pers in perplexiies:
                for lr in lrs:
                    tsne_emb = TSNE(n_components = 2, perplexity = pers, learning_rate = lr).fit_transform(emb)

                    b_tsne_emb = tsne_emb[:b_idx, :]
                    c_tsne_emb = tsne_emb[b_idx:, :]

                    b_figpath = os.path.join(self.result_dir, f'{prefix}{pca_prefix}baseline_tsne_per{pers}_lr{lr}.png')
                    plot_scatter(b_tsne_emb, b_label, b_figpath, b_st_idx)

                    c_figpath = os.path.join(self.result_dir, f'{prefix}{pca_prefix}compare_tsne_per{pers}_lr{lr}.png')
                    plot_scatter(c_tsne_emb, c_label, c_figpath, c_st_idx)

                    pbar.update(1)

            pbar.close()
