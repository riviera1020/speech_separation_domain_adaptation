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

        self.batch_size = 1
        self.num_workers = 4

        self.result_dir = config['solver']['result_dir']
        self.safe_mkdir(self.result_dir)
        #self.result_name = config['solver'].get('result_name', 'result.json')

        self.baseline = self.set_model(config['solver']['baseline'])
        self.comp = self.set_model(config['solver']['compare'])
        self.load_dset()

        self.compute_sdr = config['solver'].get('compute_sdr', True)
        self.g_mapper = GenderMapper()

    def load_dset(self):

        self.source = self.config['data']['source']
        self.target = self.config['data']['target']
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
        rname = os.path.join(self.result_dir, 'result.json')
        if not os.path.isfile(rname):
            result = self.compute_result()
        else:
            result = json.load(open(rname))

        source_baseline = self.filter_data(self.source, result['baseline'][self.source])
        target_baseline = self.filter_data(self.target, result['baseline'][self.target])

        self.plot_st(source_baseline, target_baseline, splt = 'cv', gender = 'MF', frame_num = 100000)

    def filter_data(self, dataset, result, low = -10, high = 1000):
        ret = { 'cv': [], 'tt': [] }
        for splt in result:
            uids = {'MF': [], 'MM': [], 'FF': [] }
            for uid in result[splt]:
                sisnri, sdr = result[splt][uid]
                g = self.g_mapper(uid, dataset)

                if low <= sisnri <= high:
                    uids[g].append(uid)

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

    def gather_feature(self, model, uids, dataset, splt, gender, layers, frame_num):
        ret = { l:[] for l in layers }
        cnt = 0

        with torch.no_grad(), tqdm(total = frame_num) as pbar:
            for uid in uids[splt][gender]:
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

                for l in layers:
                    f = feature[l].cpu().numpy()[0]
                    F, T = f.shape
                    ret[l].append(f)

                cnt += T
                pbar.update(T)
                if cnt >= frame_num:
                    break

        for l in ret:
            cat_tensor = np.concatenate(ret[l], axis = 1)
            cat_tensor = cat_tensor.T
            ret[l] = cat_tensor
        return ret

    def plot_st(self, source_uids, target_uids, splt, gender, frame_num = 100000):
        layers = [ 31 ]

        baseline_source_features = self.gather_feature(self.baseline, source_uids, self.source, splt, gender, layers, frame_num)
        baseline_target_features = self.gather_feature(self.baseline, target_uids, self.target, splt, gender, layers, frame_num)
        comp_source_features = self.gather_feature(self.comp, source_uids, self.source, splt, gender, layers, frame_num)
        comp_target_features = self.gather_feature(self.comp, target_uids, self.target, splt, gender, layers, frame_num)

        for l in layers:
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

            #label = np.concatenate([b_s_label, b_t_label, c_s_label, c_t_label], axis = 0)
            b_label = np.concatenate([b_s_label, b_t_label], axis = 0)
            c_label = np.concatenate([c_s_label, c_t_label], axis = 0)
            print(emb.shape)
            #print(label.shape)

            use_pca = True
            pca_prefix = ''
            if use_pca:
                pca_prefix = 'pca_'
                pca = PCA(n_components=10)
                pca.fit(emb)
                emb = pca.transform(emb)


            def plot_scatter(tsne_emb, label, fig_path):
                x = tsne_emb[:, 0]
                y = tsne_emb[:, 1]
                plt.figure()
                scatter = plt.scatter(x, y, s = 0.05, c = label, alpha=0.5)
                legned = plt.legend(*scatter.legend_elements(), loc="lower left")

                plt.savefig(fig_path)
                plt.close()

            for pers in [ 10, 20, 30, 40, 50 ]:
                for lr in [ 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
                    print(pers, lr)

                    tsne_emb = TSNE(n_components = 2, perplexity = pers, learning_rate = lr).fit_transform(emb)

                    b_tsne_emb = tsne_emb[:b_idx, :]
                    c_tsne_emb = tsne_emb[b_idx:, :]

                    b_figpath = os.path.join(self.result_dir, f'{pca_prefix}_baseline_tsne_per{pers}_lr{lr}.png')
                    plot_scatter(b_tsne_emb, b_label, b_figpath)

                    c_figpath = os.path.join(self.result_dir, f'{pca_prefix}_compare_tsne_per{pers}_lr{lr}.png')
                    plot_scatter(c_tsne_emb, c_label, c_figpath)

            exit()
