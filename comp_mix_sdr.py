import os
import json
import numpy as np

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from mir_eval.separation import bss_eval_sources

from src.utils import read_scale, NCOL
from src.dataset import wsj0_eval
from src.wham import wham_eval
from src.sep_utils import remove_pad
from src.gender_mapper import GenderMapper

def load_dset(audio_root, data_root, dset):
    if 'wham' not in dset:
        return load_data(audio_root, data_root)
    else:
        return load_wham(audio_root, data_root)

def load_data(audio_root, data_root):
    batch_size = 1
    num_workers = 2
    cv_list = os.path.join(data_root, 'id_list/cv.pkl')
    tt_list = os.path.join(data_root, 'id_list/tt.pkl')

    print(f'Load following list:')
    print(f'\t cv: {cv_list}')
    print(f'\t tt: {tt_list}')

    devset = wsj0_eval(cv_list,
            audio_root = audio_root,
            pre_load = False)
    cv_loader = DataLoader(devset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers)

    testset = wsj0_eval(tt_list,
            audio_root = audio_root,
            pre_load = False)
    tt_loader = DataLoader(testset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers)
    return cv_loader, tt_loader

def load_wham(audio_root, data_root):
    batch_size = 1
    num_workers = 2
    cv_list = os.path.join('./data/wsj0/id_list/cv.pkl')
    tt_list = os.path.join('./data/wsj0/id_list/tt.pkl')

    print(f'Load following list:')
    print(f'\t cv: {cv_list}')
    print(f'\t tt: {tt_list}')

    scale = read_scale(data_root)
    print(f'Load wham data with scale {scale}')

    devset = wham_eval(cv_list,
            audio_root = audio_root,
            pre_load = False,
            mode = 'cv',
            scale = scale)
    cv_loader = DataLoader(devset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers)

    testset = wham_eval(tt_list,
            audio_root = audio_root,
            pre_load = False,
            mode = 'tt',
            scale = scale)
    tt_loader = DataLoader(testset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers)
    return cv_loader, tt_loader

def comp_oneset(loader, dset):
    result = {}

    total_sdr = 0
    total_cnt = 0

    gs = [ 'MM', 'FF', 'MF' ]
    gender_sdr = { g: 0. for g in gs }
    gender_cnt = { g: 0 for g in gs }

    g_mapper = GenderMapper()

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

            g = g_mapper(uid, dset)
            gender_sdr[g] += sdr0
            gender_cnt[g] += 1

    total_sdr /= total_cnt

    for g in gender_sdr:
        gender_sdr[g] = gender_sdr[g] / gender_cnt[g]

    result['gender'] = gender_sdr
    return total_sdr, result

def dump_result(total_sdr, result, out_dir, prefix, dump_all = False):

    #sdr_name = os.path.join(out_dir, prefix)
    #with open(sdr_name, 'w') as f:
    #    f.write(str(total_sdr))

    gender_sdr = result['gender']
    for g, sdr0 in gender_sdr.items():
        sdr_name = os.path.join(out_dir, f'{prefix}_{g}')
        with open(sdr_name, 'w') as f:
            f.write(str(sdr0))

    if dump_all:
        json_name = os.path.join(out_dir, f'{prefix}.json')
        json.dump(result, open(json_name, 'w'))

def main(dset, audio_root, data_root, dump_all = False):

    out_dir = os.path.join(data_root, 'mix_sdr')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cv_loader, tt_loader = load_dset(audio_root, data_root, dset)

    total_sdr, result = comp_oneset(cv_loader, dset)
    dump_result(total_sdr, result, out_dir, prefix = 'cv', dump_all = dump_all)

    total_sdr, result = comp_oneset(tt_loader, dset)
    dump_result(total_sdr, result, out_dir, prefix = 'tt', dump_all = dump_all)

# change here
dset = 'libri'
audio_root = '/home/riviera1020/Big/Corpus/libri-mix/wav8k/min/'
data_root = './data/libri/'
dump_all = True

main(dset, audio_root, data_root, dump_all)
