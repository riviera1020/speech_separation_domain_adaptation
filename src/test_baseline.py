import os
import time
import yaml
import json
import datetime

import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from src.solver import Solver
from src.utils import DEV, DEBUG, NCOL
from src.conv_tasnet import ConvTasNet
from src.pit_criterion import cal_loss
from src.dataset import wsj0_eval
from src.vctk import VCTK_eval
from src.evaluation import cal_SDR, cal_SISNRi
from src.sep_utils import remove_pad, load_mix_sdr

class Tester(Solver):

    def __init__(self, config, stream = None):
        super(Tester, self).__init__(config)

        self.tr_config = config['solver']['train_config']
        self.tr_config = yaml.load(open(self.tr_config), Loader=yaml.FullLoader)

        #ts = time.time()
        #st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

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

    def load_wsj0_data(self):

        audio_root = self.config['data']['wsj_root']

        devset = wsj0_eval('./data/wsj0/id_list/cv.pkl',
                audio_root = audio_root,
                pre_load = False)
        cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

        testset = wsj0_eval('./data/wsj0/id_list/tt.pkl',
                audio_root = audio_root,
                pre_load = False)
        tt_loader = DataLoader(testset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)
        return cv_loader, tt_loader

    def load_vctk_data(self):

        audio_root = self.config['data']['vctk_root']

        devset = VCTK_eval('./data/vctk/id_list/cv.pkl',
                audio_root = audio_root,
                pre_load = False)
        cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

        testset = VCTK_eval('./data/vctk/id_list/tt.pkl',
                audio_root = audio_root,
                pre_load = False)
        tt_loader = DataLoader(testset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)
        return cv_loader, tt_loader

    def load_libri_data(self):

        audio_root = self.config['data']['libri_root']

        devset = wsj0_eval('./data/libri/id_list/cv.pkl',
                audio_root = audio_root,
                pre_load = False)
        cv_loader = DataLoader(devset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)

        testset = wsj0_eval('./data/libri/id_list/tt.pkl',
                audio_root = audio_root,
                pre_load = False)
        tt_loader = DataLoader(testset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers)
        return cv_loader, tt_loader

    def set_model(self, state_dict):
        self.model = ConvTasNet(self.tr_config['model']).to(DEV)
        self.model.load_state_dict(state_dict)

    def print_info(self):
        print(f'Epoch: {self.epoch}')

    def exec(self):
        self.print_info()

        self.model.eval()
        dsets = self.config['data']['dsets']

        result_dict = {}

        for dset in dsets:
            if dset == 'wsj0':
                cv_loader, tt_loader = self.load_wsj0_data()
                sdr0 = load_mix_sdr('./data/wsj0/mix_sdr/', ['cv', 'tt'])
            elif dset == 'vctk':
                cv_loader, tt_loader = self.load_vctk_data()
                sdr0 = load_mix_sdr('./data/vctk/mix_sdr/', ['cv', 'tt'])
            elif dset == 'libri':
                cv_loader, tt_loader = self.load_libri_data()
                sdr0 = load_mix_sdr('./data/libri/mix_sdr/', ['cv', 'tt'])

            result_dict[dset] = {}

            r = self.evaluate(cv_loader, 'cv', sdr0)
            result_dict[dset]['cv'] = r

            r = self.evaluate(tt_loader, 'tt', sdr0)
            result_dict[dset]['tt'] = r

        result_dict['tr_config'] = self.tr_config
        rname = os.path.join(self.result_dir, 'result.json')
        json.dump(result_dict, open(rname, 'w'), indent = 1)

    def evaluate(self, loader, dset, sdr0):
        total_loss = 0.
        total_SISNRi = 0
        total_SDR = 0
        total_cnt = 0

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader, ncols = NCOL)):

                padded_mixture = sample['mix'].to(DEV)
                padded_source = sample['ref'].to(DEV)
                mixture_lengths = sample['ilens'].to(DEV)

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

                    total_SDR += cal_SDR(src_ref, src_est)
                    total_SISNRi += cal_SISNRi(src_ref, src_est, mix)

        total_loss /= total_cnt
        total_SDR /= total_cnt
        total_SDRi = total_SDR - sdr0[dset]
        total_SISNRi /= total_cnt

        result = { 'total_loss': total_loss, 'total_SDRi': total_SDRi, 'total_SISNRi': total_SISNRi }
        return result

