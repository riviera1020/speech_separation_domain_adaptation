import os
import time
import yaml
import json
import datetime

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.solver import Solver
from src.utils import DEV, DEBUG, NCOL, read_scale, inf_data_gen
from src.conv_tasnet import ConvTasNet
from src.adanet import ADANet
from src.slim_conv_tasnet import MyConvTasNet
from src.pit_criterion import cal_loss
from src.dataset import wsj0, wsj0_eval
from src.wham import wham, wham_eval
from src.evaluation import cal_SDR, cal_SISNRi
from src.sep_utils import remove_pad, load_mix_sdr
from src.gender_mapper import GenderMapper

class Tester(Solver):
    def __init__(self, config):
        super(Tester, self).__init__(config)

        self.tr_config = config['solver']['train_config']
        self.tr_config = yaml.load(open(self.tr_config), Loader=yaml.FullLoader)

        self.result_dir = config['solver']['result_dir']
        self.safe_mkdir(self.result_dir)

        self.checkpoint = config['solver']['checkpoint']

        self.batch_size = 1
        self.num_workers = 4

        save_dict = torch.load(self.checkpoint, map_location=torch.device('cpu'))
        self.epoch = save_dict['epoch']
        self.valid_score = save_dict['valid_score']
        self.optim_dict = save_dict['optim']

        state_dict = save_dict['state_dict']
        self.set_model(state_dict)

        self.compute_sdr = config['solver'].get('compute_sdr', False)
        self.g_mapper = GenderMapper()

        self.bn_adapt = config['solver']['bn_adapt']
        self.bn_epochs = config['solver'].get('bn_epochs', 1)
        #self.bn_batch_size = config['solver'].get('bn_batch_size', 32)
        self.bn_batch_size = config['solver'].get('bn_batch_size', 16)
        self.seg_len = config['data'].get('seg_len', 2.0)
        self.sup_dset = self.tr_config['data'].get('dset', 'wsj0')

        two_set = config['solver'].get('two_set', '')
        if two_set != '':
            tr_loader = self.load_tr_dset(two_set, self.seg_len)
            self.another_gen = inf_data_gen(tr_loader)
        self.two_set = bool(two_set)

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
        return cv_loader, tt_loader

    def load_tr_dset(self, dset, seg_len, splt = 'tr'):
        d = 'wsj' if dset == 'wsj0' else dset # stupid error
        if 'wham' in dset:
            audio_root = self.config['data'][f'wsj_root']
            tr_list = f'./data/wsj0/id_list/{splt}.pkl'

            scale = read_scale(f'./data/{dset}')
            print(f'Load wham data with scale {scale}')
            trainset = wham(tr_list,
                    audio_root = audio_root,
                    seg_len = seg_len,
                    pre_load = False,
                    one_chunk_in_utt = True,
                    mode = 'tr',
                    scale = scale)
            tr_loader = DataLoader(trainset,
                    batch_size = self.bn_batch_size,
                    shuffle = True,
                    num_workers = self.num_workers,
                    drop_last = True)
            return tr_loader
        else:
            audio_root = self.config['data'][f'{d}_root']
            tr_list = f'./data/{dset}/id_list/{splt}.pkl'
            trainset = wsj0(tr_list,
                    audio_root = audio_root,
                    seg_len = seg_len,
                    pre_load = False,
                    one_chunk_in_utt = True,
                    mode = 'tr',
                    sp_factors = None)
            tr_loader = DataLoader(trainset,
                    batch_size = self.bn_batch_size,
                    shuffle = True,
                    num_workers = self.num_workers,
                    drop_last = True)
            return tr_loader

    def set_model(self, state_dict):
        model_type = self.tr_config['model'].get('type', 'convtasnet')
        if model_type == 'adanet':
            self.model = ADANet(self.tr_config['model']).to(DEV)
        elif model_type == 'slim':
            self.model = MyConvTasNet(self.tr_config['model']).to(DEV)
        else:
            self.model = ConvTasNet(self.tr_config['model']).to(DEV)
        self.model.load_state_dict(state_dict)

    def print_info(self):
        print(f'Epoch: {self.epoch}')

    def reset_bn(self):
        print('Reset bn stat')
        for l in range(32):
            r = l // self.model.X
            x = l %  self.model.X
            #print(r, x)

            self.model.separator.network[2][r][x].net[2].reset_running_stats()
            self.model.separator.network[2][r][x].net[3].net[2].reset_running_stats()

    def exec(self):

        #self.pytorch_gather_stat()

        self.direct_compute()


    def pytorch_gather_stat(self):
        #self.reset_bn()
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
            if self.bn_adapt:
                tr_loader = self.load_tr_dset(dset, self.seg_len)

                #cv_r = self.evaluate(cv_loader, 'cv', dset, sdr0)
                #tt_r = self.evaluate(tt_loader, 'tt', dset, sdr0)
                #result_dict[dset]['no_bn'] = { 'cv': cv_r, 'tt': tt_r }

                for epoch in tqdm(range(self.bn_epochs), ncols = NCOL):

                    self.gather_batchnorm_stat(tr_loader, self.bn_epochs)

                    cv_r = self.evaluate(cv_loader, 'cv', dset, sdr0)
                    print('cv =======')
                    print(cv_r)
                    tt_r = self.evaluate(tt_loader, 'tt', dset, sdr0)
                    print('tt =======')
                    print(tt_r)

                    result_dict[dset][epoch] = { 'cv': cv_r, 'tt': tt_r }
            else:
                r = self.evaluate(cv_loader, 'cv', dset, sdr0)
                result_dict[dset]['cv'] = r

                r = self.evaluate(tt_loader, 'tt', dset, sdr0)
                result_dict[dset]['tt'] = r

        result_dict['tr_config'] = self.tr_config
        rname = os.path.join(self.result_dir, 'result.json')
        json.dump(result_dict, open(rname, 'w'), indent = 1)
        return result_dict

    def gather_batchnorm_stat(self, tr_loader, epochs):
        def print_stat(l):
            r = l // self.model.X
            x = l %  self.model.X
            print(r, x)

            bn_res = self.model.separator.network[2][r][x].net[2]
            bn_dsconv = self.model.separator.network[2][r][x].net[3].net[2]

            print(bn_res.running_mean)

        def freeze_bn():
            #adapt_ls = [ 0, 1, 2, 3, 4, 5, 6, 7 ]
            adapt_ls = [ 0, 1, 2, 3]

            print('Freeze bn running m&v')
            print('Only adapt layers: {adapt_ls}')

            for l in range(32):
                if l in adapt_ls:
                    continue
                r = l // self.model.X
                x = l %  self.model.X

                self.model.separator.network[2][r][x].net[2].eval()
                self.model.separator.network[2][r][x].net[3].net[2].eval()

        print('Gather BatchNorm Stat')
        #print_stat(0)
        self.model.train()

        #freeze_bn()

        with torch.no_grad():
            for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):
                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)

                estimate_source = self.model(padded_mixture)

                if self.two_set:
                    ns = self.another_gen.__next__()
                    padded_mixture = ns['mix'].to(DEV)
                    estimate_source = self.model(padded_mixture)

                #print_stat(0)
            #print_stat(0)

    def direct_compute(self):
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
            tr_loader = self.load_tr_dset(dset, self.seg_len)

            for l in tqdm(range(self.model.X * self.model.R), ncols = NCOL):

                self.gather_batchnorm_stat_one_layer(tr_loader, l, res_conv = True, ds_conv = True)
                cv_r = self.evaluate(cv_loader, 'cv', dset, sdr0)
                print('res cv =======')
                print(cv_r)
                tt_r = self.evaluate(tt_loader, 'tt', dset, sdr0)
                print('res tt =======')
                print(tt_r)
                result_dict[dset][f'{l}_res'] = { 'cv': cv_r, 'tt': tt_r }

                #self.gather_batchnorm_stat_one_layer(tr_loader, l, res_conv = False, ds_conv = True)
                #cv_r = self.evaluate(cv_loader, 'cv', dset, sdr0)
                #print('ds cv =======')
                #print(cv_r)
                #tt_r = self.evaluate(tt_loader, 'tt', dset, sdr0)
                #print('ds tt =======')
                #print(tt_r)
                #result_dict[dset][f'{l}_ds'] = { 'cv': cv_r, 'tt': tt_r }

        result_dict['tr_config'] = self.tr_config
        rname = os.path.join(self.result_dir, 'result.json')
        json.dump(result_dict, open(rname, 'w'), indent = 1)
        return result_dict

    def gather_batchnorm_stat_one_layer(self, tr_loader, l, res_conv, ds_conv):
        def compute_mean(tensor):
            B, F, T = tensor.size()
            tensor = tensor.permute(0, 2, 1).contiguous().view(-1, F)
            s = tensor.size(0)
            m = tensor.mean(dim = 0)
            m = m.cpu()
            return m, s

        def compute_var(tensor, m):
            m = m.cuda()
            B, F, T = tensor.size()
            tensor = tensor.permute(0, 2, 1).contiguous().view(-1, F)
            s = tensor.size(0)
            v = ((tensor - m.unsqueeze(0))**2).mean(dim = 0)
            v = v.cpu()
            return v, s

        # use source domain running avg to gather
        self.model.eval()
        save_mean = { i: { 'res_conv': [], 'ds_conv': [] } for i in range(32) }
        save_var = { i: { 'res_conv': [], 'ds_conv': [] } for i in range(32) }
        mv = { i: { 'res_conv': {'m': None, 'v': None}, 'ds_conv': {'m': None, 'v': None} } for i in range(32) }
        with torch.no_grad():
            cnt = 0
            for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):
                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)

                _, feature = self.model.bn_forward(padded_mixture)

                if res_conv:
                    res_pre = feature[l]['res_pre']
                    m, s = compute_mean(res_pre)
                    save_mean[l]['res_conv'].append((m, s))

                if ds_conv:
                    ds_pre = feature[l]['ds_pre']
                    m, s = compute_mean(ds_pre)
                    save_mean[l]['ds_conv'].append((m, s))

                del feature
                cnt += 1

                #if cnt > 5:
                #    break

            if res_conv:
                mm, ss = zip(*save_mean[l]['res_conv'])
                mm = torch.stack(mm, dim = 0)
                ss = torch.FloatTensor(ss)
                m = (mm * ss.unsqueeze(-1)).sum(dim = 0) / ss.sum()
                mv[l]['res_conv']['m'] = m

            if ds_conv:
                mm, ss = zip(*save_mean[l]['ds_conv'])
                mm = torch.stack(mm, dim = 0)
                ss = torch.FloatTensor(ss)
                m = (mm * ss.unsqueeze(-1)).sum(dim = 0) / ss.sum()
                mv[l]['ds_conv']['m'] = m

            cnt = 0
            for i, sample in enumerate(tqdm(tr_loader, ncols = NCOL)):
                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)

                _, feature = self.model.bn_forward(padded_mixture)

                if res_conv:
                    res_pre = feature[l]['res_pre']
                    v, s = compute_var(res_pre, mv[l]['res_conv']['m'])
                    save_var[l]['res_conv'].append((v, s))

                if ds_conv:
                    ds_pre = feature[l]['ds_pre']
                    v, s = compute_var(ds_pre, mv[l]['ds_conv']['m'])
                    save_var[l]['ds_conv'].append((v, s))

                del feature
                cnt += 1
                #if cnt > 5:
                #    break

            if res_conv:
                vv, ss = zip(*save_var[l]['res_conv'])
                vv = torch.stack(vv, dim = 0)
                ss = torch.FloatTensor(ss)
                v = (vv * ss.unsqueeze(-1)).sum(dim = 0) / ss.sum()
                mv[l]['res_conv']['v'] = v

            if ds_conv:
                vv, ss = zip(*save_var[l]['ds_conv'])
                vv = torch.stack(vv, dim = 0)
                ss = torch.FloatTensor(ss)
                v = (vv * ss.unsqueeze(-1)).sum(dim = 0) / ss.sum()
                mv[l]['ds_conv']['v'] = v

        r = l // self.model.X
        x = l %  self.model.X
        if res_conv:
            self.model.separator.network[2][r][x].net[2].running_mean = mv[l]['res_conv']['m'].to(DEV)
            self.model.separator.network[2][r][x].net[2].running_var = mv[l]['res_conv']['v'].to(DEV)
        if ds_conv:
            self.model.separator.network[2][r][x].net[3].net[2].running_mean = mv[l]['ds_conv']['m'].to(DEV)
            self.model.separator.network[2][r][x].net[3].net[2].running_var = mv[l]['ds_conv']['v'].to(DEV)

    def evaluate(self, loader, dset, dataset, sdr0):
        self.model.eval()
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

                estimate_source = self.model(padded_mixture)

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

