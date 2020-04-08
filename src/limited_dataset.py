import os
import random
import math
import torch
import numpy as np
import soundfile as sf
import _pickle as cPickle
from torch.utils.data import Dataset

class LimitDataset(Dataset):
    def __init__(self, id_list_path, audio_root, seg_len, spk_info, spk_num, utts_per_spk, mode = 'tr'):
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
        super(LimitDataset, self).__init__()

        self.data = cPickle.load(open(id_list_path, 'rb'))
        self.ginfo = cPickle.load(open(spk_info, 'rb'))
        self.audio_root = audio_root
        self.sr = 8000

        self.seg_len = int(seg_len * self.sr)

        males = []
        females = []

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
        for spk in self.spks:
            self.spk2utts[spk] = []
            utts = list(self.data[spk].keys())
            lens.append(len(utts))
            if utts_per_spk != 'all':
                random.shuffle(utts)
                utts = utts[:utts_per_spk]
            for uid in utts:
                path, utt_len, scale = self.data[spk][uid]
                re = utt_len - self.seg_len
                if re == 0:
                    s = 0
                else:
                    s = random.randint(0, re - 1)
                e = s + self.seg_len
                info = [ uid, spk, s, e, scale ]
                self.id_list.append(info)
                self.spk2utts[spk].append(info)
                cnt += 1
        duration = float(cnt) * self.seg_len / self.sr / 3600
        print(f'Speaker Num: {self.spk_num}')
        print(f'Utts per speaker: {utts_per_spk}')
        print(f'Total utt Num: {cnt}')
        print(f'Total duration: {duration}')

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

