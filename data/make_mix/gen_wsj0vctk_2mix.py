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
            if i == 0:
                continue

            _info = line.rstrip().split()
            spk = _info[0]
            spk = 'p' + spk

            gender = _info[2]
            info[gender].append(spk)

    ret = {}
    for g in info:
        for spk in info[g]:
            ret[spk] = g
    return ret

def sample_snr(_list):

    ret = []
    for sw, sv in _list:
        snr = random.uniform(0, 2.5)

        flip = random.randint(0, 1)
        if flip == 0:
            s1, s2 = sw, sv
        else:
            s1, s2 = sv, sw

        ret.append((s1, s2, snr))
    return ret

def output(_list, path):
    with open(path, 'w') as f:
        for s1, s2, snr in _list:
            line = f'{s1} {snr:.6f} {s2} {-snr:.6f}\n'
            f.write(line)

def gen_mix(wsj0_wavs, vctk_wavs, wsj0_dur, vctk_dur, wsj0_gender, vctk_gender, num):
    # min is always wsj0
    min_num = len(wsj0_wavs)

    random.shuffle(wsj0_wavs)
    random.shuffle(vctk_wavs)

    total_t = 0
    g_count = { 'MF': 0, 'MM': 0, 'FF': 0 }
    ret = []
    for i in range(num):
        if i < min_num:
            sw = wsj0_wavs[i]
            sv = vctk_wavs[i]
        else:
            sw = random.choice(wsj0_wavs)
            if i < len(vctk_wavs):
                sv = vctk_wavs[i]
            else:
                sv = random.choice(vctk_wavs)

        spk_w = sw.split('/')[2]
        spk_v = sv.split('/')[1]

        wt = wsj0_dur[spk_w][sw]
        vt = vctk_dur[spk_v][sv.split('/')[-1]]
        rt = min(wt, vt)
        total_t += rt

        wg = wsj0_gender[spk_w]
        vg = vctk_gender[spk_v]
        if wg != vg:
            g_count['MF'] += 1
        elif wg == 'M':
            g_count['MM'] += 1
        else:
            g_count['FF'] += 1

        ret.append((sw, sv))

    total_hr = total_t / 3600
    print(total_hr)
    print(g_count)
    return ret, total_hr, g_count

def list_to_plain(path):
    ret = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            s1, _, s2, _ = line.split()

            ret.append(s1)
            ret.append(s2)
    ret = list(set(ret))
    return ret

def main():
    """
    wsj0 duration:
        tr: 30.38
        cv: 7.67
        tt: 4.82
    """
    # params
    splt_num = {
            'tr': 30000,
            'cv': 8000,
            'tt': 5000, }

    wsj0_list_root = '../make-wsj0-mix/'
    vctk_list_root = './vctk_info/'

    info = '../VCTK-Corpus/speaker-info.txt'
    vctk_spkinfo = read_spk_info(info)
    vctk_dur_info = json.load(open('./vctk_info/vctk_duration.json'))

    wsj0_spkinfo = json.load(open('./wsj_info/spk_info.json'))
    wsj0_dur_info = json.load(open('./wsj_info/wsj0_duration.json'))

    g_count = { 'M': 0, 'F': 0 }

    output_info = {}
    for splt in [ 'tr', 'cv', 'tt' ]:
        print(f'========={splt}===========')
        wsj0_list = f'../make-wsj0-mix/mix_2_spk_{splt}.txt'
        vctk_list = f'./vctk_info/vctk_mix_2_spk_{splt}.txt'

        wsj0_wavs = list_to_plain(wsj0_list)
        print('WSJ0 utt num:', len(wsj0_wavs))

        g_count = { 'M': 0, 'F': 0 }
        for s in wsj0_wavs:
            spk = s.split('/')[2]
            g = wsj0_spkinfo[spk]
            g_count[g] += 1
        print(g_count)

        vctk_wavs = list_to_plain(vctk_list)
        print('VCTK utt num:', len(vctk_wavs))

        g_count = { 'M': 0, 'F': 0 }
        for s in vctk_wavs:
            spk = s.split('/')[1]
            g = vctk_spkinfo[spk]
            g_count[g] += 1
        print(g_count)

        pairs, total_hr, g_info = gen_mix(wsj0_wavs, vctk_wavs, wsj0_dur_info, vctk_dur_info,
                wsj0_spkinfo, vctk_spkinfo, num = splt_num[splt])

        done = sample_snr(pairs)
        output(done, f'./wsj0vctk_info/wsj0vctk_mix_2_spk_{splt}.txt')

        output_info[splt] = { 'total_hr': total_hr, 'g_info': g_info }

    json.dump(output_info, open('./wsj0vctk_info/gen_data_info.json', 'w'), indent = 1)

main()
