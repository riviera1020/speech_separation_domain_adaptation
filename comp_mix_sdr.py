

import os
import json
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from mir_eval.separation import bss_eval_sources

from src.dataset import wsj0_eval
from src.sep_utils import remove_pad

NCOL = 100

def load_data():
    audio_root = '/home/riviera1020/Big/Corpus/wsj0-mix/'
    batch_size = 1
    num_workers = 2
    devset = wsj0_eval('./data/wsj0/id_list/cv.pkl',
            audio_root = audio_root,
            pre_load = False)
    cv_loader = DataLoader(devset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers)

    testset = wsj0_eval('./data/wsj0/id_list/tt.pkl',
            audio_root = audio_root,
            pre_load = False)
    tt_loader = DataLoader(devset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers)

    return cv_loader, tt_loader

def comp_oneset(loader):
    result = {}

    total_sdr = 0
    total_cnt = 0

    for i, sample in enumerate(tqdm(loader, ncols = NCOL)):
        padded_mixture = sample['mix']
        padded_source = sample['ref']
        mixture_lengths = sample['ilens']
        uids = sample['uid']

        B = padded_source.size(0)
        total_cnt += B

        padded_source = remove_pad(padded_source, mixture_lengths)
        padded_mixture = remove_pad(padded_mixture, mixture_lengths)

        for b in range(B):
            mix = padded_mixture[b]
            src_ref = padded_source[b]

            src_anchor = np.stack([mix, mix], axis=0)
            sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor, False)

            sdr0 = np.mean(sdr0)
            total_sdr += sdr0

            uid = uids[b]
            result[uid] = sdr0

    total_sdr /= total_cnt
    return total_sdr, result

def dump_result(total_sdr, result, out_dir, prefix, dump_all = False):

    sdr_name = os.path.join(out_dir, prefix)
    with open(sdr_name, 'w') as f:
        f.write(str(total_sdr))

    if dump_all:
        json_name = os.path.join(out_dir, f'{prefix}.json')
        json.dump(result, open(json_name, 'w'))

def main():

    out_dir = './data/wsj0/mix_sdr/'

    cv_loader, tt_loader = load_data()

    total_sdr, result = comp_oneset(cv_loader)
    dump_result(total_sdr, result, out_dir, prefix = 'cv', dump_all = True)

    total_sdr, result = comp_oneset(tt_loader)
    dump_result(total_sdr, result, out_dir, prefix = 'tt', dump_all = True)

main()
