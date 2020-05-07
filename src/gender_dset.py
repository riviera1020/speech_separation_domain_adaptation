
import os
import random
import math
import torch
import numpy as np
import subprocess
import soundfile as sf
import _pickle as cPickle

from io import BytesIO
from tqdm import tqdm
from torch.utils.data import Dataset
from src.gender_mapper import GenderMapper

class wsj0_gender(Dataset):

    def __init__(self, id_list_path, audio_root, seg_len = 4.0, pre_load = True, one_chunk_in_utt = True, mode = 'tr', gender = None):
        """
        Args:
            id_list_path     : id_list from data/wsj0/preprocess.py
            audio_root       : root dir for wsj0 dataset
            seg_len          : segment len for utt in sec
            pre_load         : pre load all audio into RAM
            one_chunk_in_utt : T -> random select one chunk in one utt
                               F -> split and access all chunk, (must false in cv and tt)
            mode             : tr/cv/tt
            sp_factors       : support speed augm with list of factor [ 0.9, 1.0, 1.1 ]
        """
        super(wsj0_gender, self).__init__()

        self.data = cPickle.load(open(id_list_path, 'rb'))
        self.audio_root = audio_root
        self.sr = 8000

        if seg_len != -1:
            self.seg_len = int(seg_len * self.sr)

        self.pre_load = pre_load
        self.one_chunk = one_chunk_in_utt

        self.gender_mapper = GenderMapper()
        if gender != None:
            print(f'Gather gender {gender} utt')
            self.gender_mapper = GenderMapper()
        else:
            print('Specify gender')
            exit()

        self.id_list = []
        drop_num = 0
        drop_len = 0.0
        dset_num = 0
        dset_len = 0.0
        for uid in self.data:
            path, utt_len = self.data[uid]['mix']
            g = self.gender_mapper(uid, 'wsj0')
            if utt_len >= self.seg_len and g == gender:
                info = [ uid, uid, -1, -1 ]
                self.id_list.append(info)
                dset_num += 1
                dset_len += utt_len
            else:
                drop_num += 1
                drop_len += utt_len

        drop_len = drop_len / (self.sr * 3600)
        dset_len = dset_len / (self.sr * 3600)
        print(f'Drop utt less than {self.seg_len}')
        print(f'Drop num: {drop_num}')
        print(f'Drop len: {drop_len:.3f} hr')

        print(f'Dset num: {dset_num}')
        print(f'Dset len: {dset_len:.3f} hr')

    def pad_audio(self, audio, ilen):
        base = np.zeros(self.seg_len, dtype = np.float32)
        base[:ilen] = audio
        return base

    def load_audio(self, path, factor = 1.0):
        if factor == 1.0:
            audio, _ = sf.read(path)
        else:
            cmd = f'sox {path} -t wav - speed {factor}'.split()
            result = subprocess.Popen(args = cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            bio = BytesIO(result.stdout.read())
            audio, _ = sf.read(bio)
            bio.close()
        return audio

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        info struct: [ utt id, chunk id, start, end ]
        """
        uid, cid, s, e = self.id_list[idx]
        mix_path = os.path.join(self.audio_root, self.data[uid]['mix'][0])
        s1_path = os.path.join(self.audio_root, self.data[uid]['s1'][0])
        s2_path = os.path.join(self.audio_root, self.data[uid]['s2'][0])

        factor = 1.0
        mix_audio = self.load_audio(mix_path, factor)
        s1_audio = self.load_audio(s1_path, factor)
        s2_audio = self.load_audio(s2_path, factor)

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
                   'mix': mix_audio, 'ref': sep_audio }
        return sample
