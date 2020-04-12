import os
import random
import math
import torch
import numpy as np
import soundfile as sf
import _pickle as cPickle
from torch.utils.data import Dataset

def sample_seg(utt_len, seg_len):
    re = utt_len - seg_len
    if re == 0:
        s = 0
    else:
        s = random.randint(0, re - 1)
    e = s + seg_len
    return s, e

def define_maxrep(seg_len, seg_rule):
    base_len = 1.0
    if seg_len > base_len:
        print('bigger than 1.0 seg')
        print('redefine maxrep')
        exit()
    ratio = int(base_len / seg_len)
    if seg_rule == 'wsj0':
        max_rep = 4
    else:
        max_rep = 1
    return max_rep * ratio

class LimitDataset(Dataset):
    def __init__(self, id_list_path, audio_root, seg_len, spk_info, spk_num, utts_per_spk, mode = 'tr', seg_rule = 'wsj0'):
        """
        Args:
            id_list_path     : id_list
            audio_root       : root dir
            seg_len          : segment len for utt in sec
            spk_info         : spk:gender info
            spk_num          : use how many spk in dataset ( wsj0: 101, vctk: 98)
                               'all' for no sample spk
            utts_per_spk     : use how many utts per speaker
                               'all' for no sample utt
            mode             : tr/cv/tt
            seg_rule         : wsj0, vctk
        """
        super(LimitDataset, self).__init__()

        self.data = cPickle.load(open(id_list_path, 'rb'))
        self.ginfo = cPickle.load(open(spk_info, 'rb'))
        self.audio_root = audio_root
        self.sr = 8000

        self.seg_len = int(seg_len * self.sr)

        males = []
        females = []
        max_rep = define_maxrep(seg_len, seg_rule)

        drop_num = 0
        drop_len = 0.0
        for spk in self.data:
            if self.ginfo[spk] == 'M':
                males.append(spk)
            else:
                females.append(spk)
            pops = []
            for uid in self.data[spk]:
                path, utt_len, _ = self.data[spk][uid]
                if utt_len < self.seg_len:
                    pops.append(uid)
                    drop_num += 1
                    drop_len += utt_len
            for uid in pops:
                self.data[spk].pop(uid)

        drop_len = drop_len / (self.sr * 3600)
        print(f'Drop utt less than {self.seg_len}')
        print(f'Drop num: {drop_num}')
        print(f'Drop len: {drop_len:.3f} hr')

        if spk_num == 'all':
            self.spks = list(self.data.keys())
            self.spk_num = len(self.spks)
        else:
            random.shuffle(males)
            random.shuffle(females)
            self.spk_num = spk_num
            num = spk_num // 2
            self.spks = males[:num] + females[:num]

        self.spk2id = { spk:i for i, spk in enumerate(self.spks) }

        self.id_list = []
        self.spk2utts = {}
        cnt = 0
        lens = []
        max_utts = []
        for spk in self.spks:
            self.spk2utts[spk] = []
            utts = list(self.data[spk].keys())
            lens.append(len(utts))
            if utts_per_spk != 'all':
                random.shuffle(utts)
                utts = utts[:utts_per_spk]
            max_utts.append(len(utts))
            for uid in utts:
                path, utt_len, scale = self.data[spk][uid]
                r = utt_len // self.seg_len
                r = min(max_rep, r)
                nseg_len = r * self.seg_len
                info = []
                s, e = sample_seg(utt_len, nseg_len)
                for i in range(r):
                    ss = s + i * self.seg_len
                    ee = ss + self.seg_len
                    info.append([ uid, spk, ss, ee, scale ])
                    cnt += 1

                self.id_list += info
                self.spk2utts[spk] += info

        duration = float(cnt) * self.seg_len / self.sr / 3600
        max_utts_per_spk = max(max_utts)

        print(f'Speaker Num: {self.spk_num}')
        print(f'Utts per speaker: {utts_per_spk}')
        print(f'Max Utts per speaker: {max_utts_per_spk}')
        print(f'Total utt Num: {cnt}')
        print(f'Total duration: {duration}')

        self.utts_per_spk = utts_per_spk
        self.utt_num = cnt
        self.duration = duration
        self.max_utts_per_spk = max_utts_per_spk

    def get_info(self):
        ret = { 'spk_num': self.spk_num,
                'utts_per_spk': self.utts_per_spk,
                'max_utts_per_spk': self.max_utts_per_spk,
                'utt_num': self.utt_num,
                'duration': self.duration }
        return ret

    def pad_audio(self, audio, ilen):
        base = np.zeros(self.seg_len, dtype = np.float32)
        base[:ilen] = audio
        return base

    def load_audio(self, path, factor = 1.0):
        audio, _ = sf.read(path)
        return audio

    def sample_from_another_spk(self, spk):
        sid = self.spk2id[spk]
        cands = list(range(0, sid)) + list(range(sid + 1, len(self.spks)))
        cand = random.choice(cands)
        s2_spk = self.spks[cand]
        info = random.choice(self.spk2utts[s2_spk])
        return info

    def process(self, apath, s, e, scale):
        audio = self.load_audio(apath).astype(np.float32)
        audio = audio[s:e]
        w = 10 ** (scale/20)
        audio = audio / w
        return audio

    def mixing(self, s1, s2, snr):
        w1 = 10 ** (snr/20)
        w2 = 10 ** (-snr/20)
        s1 = w1 * s1
        s2 = w2 * s2
        mix = s1 + s2
        return s1, s2, mix

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        info struct: [ utt id, chunk id, start, end ]
        """
        uid1, spk1, s1, e1, scale1 = self.id_list[idx]
        uid2, spk2, s2, e2, scale2 = self.sample_from_another_spk(spk1)

        s1_path = self.data[spk1][uid1][0]
        s2_path = self.data[spk2][uid2][0]
        s1_path = os.path.join(self.audio_root, s1_path)
        s2_path = os.path.join(self.audio_root, s2_path)

        s1_audio = self.process(s1_path, s1, e1, scale1)
        s2_audio = self.process(s2_path, s2, e2, scale2)

        snr = random.uniform(0, 2.5)
        s1_audio, s2_audio, mix_audio = self.mixing(s1_audio, s2_audio, snr)
        ilen = len(mix_audio)
        sep_audio = np.stack([s1_audio, s2_audio], axis = 0)

        uid = f'{uid1}_{snr:.6f}_{uid2}_{-snr:.6f}.wav'
        sample = { 'uid': uid, 'ilens': ilen, 'mix': mix_audio, 'ref': sep_audio }

        return sample

class LimitWham(LimitDataset):
    def __init__(self, id_list_path, audio_root, seg_len, spk_info, spk_num, utts_per_spk, mode = 'tr', seg_rule = 'wsj0', scale = 1.0):
        """
        Args:
            id_list_path     : id_list
            audio_root       : root dir
            seg_len          : segment len for utt in sec
            spk_info         : spk:gender info
            spk_num          : use how many spk in dataset ( wsj0: 101, vctk: 98)
                               'all' for no sample spk
            utts_per_spk     : use how many utts per speaker
                               'all' for no sample utt
            mode             : tr/cv/tt
        """
        super(LimitWham, self).__init__(id_list_path, audio_root, seg_len, spk_info,
                                        spk_num, utts_per_spk, mode, seg_rule)

        noise_list_path = os.path.join(f'./data/wham/noise_id_list/{mode}.pkl')
        self.noise_data = cPickle.load(open(noise_list_path, 'rb'))
        self.scale = scale

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        info struct: [ utt id, chunk id, start, end ]
        """
        uid1, spk1, s1, e1, scale1 = self.id_list[idx]
        uid2, spk2, s2, e2, scale2 = self.sample_from_another_spk(spk1)

        s1_path = self.data[spk1][uid1][0]
        s2_path = self.data[spk2][uid2][0]

        nkey = s1_path.split('/')[-1]
        npath, _, ss, sn, _ = self.noise_data[nkey]['noise']

        s1_path = os.path.join(self.audio_root, s1_path)
        s2_path = os.path.join(self.audio_root, s2_path)
        npath = os.path.join(self.audio_root, npath)

        s1_audio = self.process(s1_path, s1, e1, scale1)
        s2_audio = self.process(s2_path, s2, e2, scale2)
        noise = self.process(npath, s1, e1, scale = 0.)

        snr = random.uniform(0, 2.5)
        s1_audio, s2_audio, mix_audio = self.mixing(s1_audio, s2_audio, snr)
        ilen = len(mix_audio)

        sep_audio = np.stack([s1_audio, s2_audio], axis = 0)
        mix_audio = mix_audio + self.scale + noise

        uid = f'{uid1}_{snr:.6f}_{uid2}_{-snr:.6f}.wav'
        sample = { 'uid': uid, 'ilens': ilen, 'mix': mix_audio, 'ref': sep_audio }

        return sample
