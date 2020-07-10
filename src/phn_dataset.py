
import os
import random

import torch
import numpy as np
import soundfile as sf
import _pickle as cPickle

from tqdm import tqdm
from torch.utils.data import Dataset

class PhnSepDataset(Dataset):

    def __init__(self, id_list_path, audio_root, phone_mapper, sample_rate, seg_len = 4.0, pre_load = True, one_chunk_in_utt = True, mode = 'tr'):
        """
        Args:
            id_list_path     : id_list from data/*/preprocess.py
            audio_root       : root dir for dataset
            phone_mapper     : phone mapper for phn label
            seg_len          : segment len for utt in sec
            pre_load         : pre load all audio into RAM
            one_chunk_in_utt : T -> random select one chunk in one utt
                               F -> split and access all chunk, (must false in cv and tt)
            mode             : tr/cv/tt
        """
        super(PhnSepDataset, self).__init__()

        self.data = cPickle.load(open(id_list_path, 'rb'))
        self.audio_root = audio_root
        self.sr = sample_rate
        self.phone_mapper = phone_mapper
        self.mode = mode

        if seg_len != -1:
            self.seg_len = int(seg_len * self.sr)

        self.pre_load = pre_load
        self.one_chunk = one_chunk_in_utt

        self.id_list = []
        drop_num = 0
        drop_len = 0.0
        for uid in self.data:
            path, utt_len = self.data[uid]['mix']
            if mode == 'tr':
                if utt_len >= self.seg_len:
                    if self.one_chunk:
                        info = [ uid, uid, -1, -1 ]
                        self.id_list.append(info)
                    else:
                        seg_num = utt_len // self.seg_len
                        if utt_len % self.seg_len > 0:
                            seg_num += 1
                        for i in range(seg_num):
                            s = int(i * self.seg_len)
                            e = int((i + 1) * self.seg_len)
                            if i == seg_num - 1:
                                e = min([e, utt_len])
                            info = [ uid, f'{uid}_{i}', s, e ]
                            self.id_list.append(info)
                else:
                    drop_num += 1
                    drop_len += utt_len

            else: # in cv and tt, don't seg
                # Same template with wsj0
                info = [ uid, uid, -1, -1 ]
                self.id_list.append(info)

        if mode == 'tr':
            drop_len = drop_len / (self.sr * 3600)
            print(f'Drop utt less than {self.seg_len}')
            print(f'Drop num: {drop_num}')
            print(f'Drop len: {drop_len:.3f} hr')

        if self.pre_load:
            print('Start pre-loading audio')
            self.audios = {}
            for uid in tqdm(self.data):
                self.audios[uid] = {}
                for speaker in self.data[uid]:
                    path, _ = self.data[uid][speaker]
                    path = os.path.join(audio_root, path)
                    audio, _ = sf.read(path)
                    audio = audio.astype(np.float32)
                    self.audios[uid][speaker] = audio

    def pad_audio(self, audio, ilen):
        base = np.zeros(self.seg_len, dtype = np.float32)
        base[:ilen] = audio
        return base

    def __len__(self):
        return len(self.id_list)

    def get_complete_utt(self, idx):
        uid, cid, s, e = self.id_list[idx]
        if self.pre_load:
            mix_audio = self.audios[uid]['mix']
            s1_audio = self.audios[uid]['s1']
            s2_audio = self.audios[uid]['s2']
        else:
            mix_path = os.path.join(self.audio_root, self.data[uid]['mix'][0])
            s1_path = os.path.join(self.audio_root, self.data[uid]['s1'][0])
            s2_path = os.path.join(self.audio_root, self.data[uid]['s2'][0])

            mix_audio, _ = sf.read(mix_path)
            s1_audio, _ = sf.read(s1_path)
            s2_audio, _ = sf.read(s2_path)

            mix_audio = mix_audio.astype(np.float32)
            s1_audio = s1_audio.astype(np.float32)
            s2_audio = s2_audio.astype(np.float32)

        ilen = len(mix_audio)

        s = 0
        e = ilen
        s1_phns, s2_phns = self.phone_mapper.get_label(uid, s, e)
        ref_phns = np.stack([s1_phns, s2_phns])
        ref_phns = ref_phns.astype(np.int)

        sep_audio = np.stack([s1_audio, s2_audio], axis = 0)

        sample = { 'uid': uid, 'cid': cid, 'ilens': ilen,
                   'mix': mix_audio, 'ref': sep_audio,
                   'ref_phns': ref_phns }
        return sample

    def __getitem__(self, idx):
        """
        info struct: [ utt id, chunk id, start, end ]
        """
        if self.mode != 'tr':
            return self.get_complete_utt(idx)

        uid, cid, s, e = self.id_list[idx]
        if self.pre_load:
            mix_audio = self.audios[uid]['mix']
            s1_audio = self.audios[uid]['s1']
            s2_audio = self.audios[uid]['s2']
        else:
            mix_path = os.path.join(self.audio_root, self.data[uid]['mix'][0])
            s1_path = os.path.join(self.audio_root, self.data[uid]['s1'][0])
            s2_path = os.path.join(self.audio_root, self.data[uid]['s2'][0])

            mix_audio, _ = sf.read(mix_path)
            s1_audio, _ = sf.read(s1_path)
            s2_audio, _ = sf.read(s2_path)

            mix_audio = mix_audio.astype(np.float32)
            s1_audio = s1_audio.astype(np.float32)
            s2_audio = s2_audio.astype(np.float32)

        if self.one_chunk:
            L = len(mix_audio)
            re = L - self.seg_len
            if re == 0:
                s = 0
            else:
                s = random.randint(0, re - 1)
            e = s + self.seg_len

        s1_phns, s2_phns = self.phone_mapper.get_label(uid, s, e)
        ref_phns = np.stack([s1_phns, s2_phns])
        ref_phns = ref_phns.astype(np.int)

        mix_audio = mix_audio[s:e]
        s1_audio = s1_audio[s:e]
        s2_audio = s2_audio[s:e]

        ilen = len(mix_audio)
        if ilen < self.seg_len:
            mix_audio = self.pad_audio(mix_audio, ilen)
            s1_audio = self.pad_audio(s1_audio, ilen)
            s2_audio = self.pad_audio(s2_audio, ilen)

        sep_audio = np.stack([s1_audio, s2_audio], axis = 0)

        sample = { 'uid': uid, 'cid': cid, 'ilens': ilen,
                   'mix': mix_audio, 'ref': sep_audio,
                   'ref_phns': ref_phns }
        return sample
