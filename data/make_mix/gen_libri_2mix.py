"""
Related:
    VoiceFilter: https://arxiv.org/pdf/1810.04826.pdf
"""
import os
import json
import copy
import random

from glob import glob
from tqdm import tqdm

def read_spk_info(path):

    info = { 'M': [], 'F': [] }
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 12:
                continue

            _info = line.rstrip().split("|")
            spk = _info[0].strip()

            gender = _info[1].strip()
            info[gender].append(spk)
    return info

def is_split_valid(tr, tt, dur_info):

    tr_num = 0
    tr_time = 0.
    for spk in tr:
        utt_num = len(dur_info[spk]) - 1 # remove "total" in dict
        tr_num += utt_num
        tr_time += dur_info[spk]['total']

    tt_num = 0
    tt_time = 0.
    for spk in tt:
        utt_num = len(dur_info[spk]) - 1 # remove "total" in dict
        tt_num += utt_num
        tt_time += dur_info[spk]['total']

    tr_time /= 3600
    tt_time /= 3600

    # This time constriant is no reason, I just happy
    time_valid = tt_time >= 4.0 and tt_time <= 4.1

    if tr_num % 2 == 0 and tt_num % 2 == 0 and time_valid:
        print(f'tr+cv duration: {tr_time}')
        print(f'tt    duration: {tt_time}')
        return True
    else:
        return False

def get_utt_num(spks, spk2path):
    num = 0
    for spk in spks:
        num += len(spk2path[spk])
    return num

def draw_one(spk, spk2path, spks):
    """
    Warning: this fc will help delete empty list & dict after draw
    """
    paths = spk2path[spk]
    p = paths.pop(random.randrange(len(paths)))

    if len(paths) != 0:
        spk2path[spk] = paths
    else:
        # Deletion is here
        spk2path.pop(spk, None)
        spks.remove(spk)
    return p

def gen_mix(M, F, spk2path, strict):

    M_num = get_utt_num(M, spk2path)
    F_num = get_utt_num(F, spk2path)

    pair_num = (M_num + F_num) // 2
    #assert (M_num + F_num) % 2 == 0
    diff_num = pair_num // 2
    all_M_num = M_num - diff_num
    all_F_num = F_num - diff_num

    print(f'Male   utt: {M_num}')
    print(f'Female utt: {F_num}')
    print(f'Pair   Num: {pair_num}')

    # if sample bad, do again
    # i.e. remain same gender
    _M = M
    _F = F
    _spk2path = spk2path
    sample_good = False
    while True:
        M = copy.deepcopy(_M)
        F = copy.deepcopy(_F)
        spk2path = copy.deepcopy(_spk2path)
        all_spk = M + F

        diff = []
        all_M = []
        all_F = []

        for i in range(pair_num):

            spk1, spk2 = random.sample(all_spk, 2)

            s1 = draw_one(spk1, spk2path, all_spk)
            s2 = draw_one(spk2, spk2path, all_spk)

            if spk1 in M and spk2 in M:
                all_M.append((s1, s2))
            elif spk1 in F and spk2 in F:
                all_F.append((s1, s2))
            else:
                diff.append((s1, s2))

            if len(all_spk) == 1 and len(spk2path[all_spk[0]]) > 1:
                #print('Sample Fail')
                #print(all_spk)
                #print(spk2path[all_spk[0]])
                break

            if strict:
                if i == pair_num - 1:
                    sample_good = True
            else:
                if len(all_spk) == 1 and len(spk2path[all_spk[0]]) == 1:
                    print(all_spk)
                    print(spk2path[all_spk[0]])
                    sample_good = True

        if sample_good:
            break

    print(f'Mix: {len(diff)}')
    print(f'M  : {len(all_M)}')
    print(f'F  : {len(all_F)}')
    return diff, all_M, all_F

def sample_snr(_list):

    ret = []
    for s1, s2 in _list:
        snr = random.uniform(0, 2.5)
        ret.append((s1, s2, snr))
    return ret

def output(_list, path):
    with open(path, 'w') as f:
        for s1, s2, snr in _list:
            line = f'{s1} {snr:.6f} {s2} {-snr:.6f}\n'
            f.write(line)

def mix_subset(_root, dset, info):

    spk2path = {}
    libri_root = os.path.join(_root, dset)

    spk_paths = glob(os.path.join(libri_root, '*'))
    ret = {}
    for spk_path in spk_paths:

        spk = spk_path.split('/')[-1]
        paths = glob(os.path.join(spk_path, '*/*.flac'))
        spk2path[spk] = []
        for path in paths:
            path = path.replace(_root, '')
            spk2path[spk].append(path)

    spks = list(spk2path.keys())
    M = []
    F = []
    for spk in spks:
        if spk in info['M']:
            M.append(spk)
        elif spk in info['F']:
            F.append(spk)
        else:
            print('Damn')

    # based on check_wsj0_mix_info.py
    # ratio of MF:M:F ~= 2:1:1
    strict = False
    if dset == 'test-clean':
        strict = True

    diff, all_M, all_F = gen_mix(M, F, spk2path, strict)
    ret = diff + all_M + all_F
    return ret

def main():

    _root = '../LibriSpeech/'

    info = '../LibriSpeech/SPEAKERS.TXT'
    info = read_spk_info(info)
    dur_info = json.load(open('./libri_info/libri_duration.json'))

    # In LibriSpeech, you don't need to split corpus like VCTK or WSJ0

    print('Gen tr')
    tr = mix_subset(_root, 'train-clean-100', info)

    print('Gen cv')
    cv = mix_subset(_root, 'dev-clean', info)

    print('Gen tt')
    tt = mix_subset(_root, 'test-clean', info)

    tr = sample_snr(tr)
    cv = sample_snr(cv)
    tt = sample_snr(tt)

    tr_path = './libri_info/libri_mix_2_spk_tr.txt'
    cv_path = './libri_info/libri_mix_2_spk_cv.txt'
    tt_path = './libri_info/libri_mix_2_spk_tt.txt'

    output(tr, tr_path)
    output(cv, cv_path)
    output(tt, tt_path)

main()
