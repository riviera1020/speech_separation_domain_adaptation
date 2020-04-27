
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

class wsj0(Dataset):

    def __init__(self, id_list_path, audio_root, seg_len = 4.0, pre_load = True, one_chunk_in_utt = True, mode = 'tr', sp_factors = None):
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
        super(wsj0, self).__init__()

        self.data = cPickle.load(open(id_list_path, 'rb'))
        self.audio_root = audio_root
        self.sr = 8000

        if seg_len != -1:
            self.seg_len = int(seg_len * self.sr)

        self.pre_load = pre_load
        self.one_chunk = one_chunk_in_utt

        if sp_factors != None:
            if 1.0 not in sp_factors:
                sp_factors.append(1.0)
            #TODO, pre_load now is useless, not support
            if pre_load:
                print("Error: pre_load don't support speed perturb now")
                exit()
        self.sp_factors = sp_factors

        self.id_list = []
        drop_num = 0
        drop_len = 0.0
        for uid in self.data:
            path, utt_len = self.data[uid]['mix']
            if self.sp_factors != None:
                # fast speed will shrink the len of audio
                mf = max(self.sp_factors)
                utt_len = math.floor(float(utt_len) / mf)
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
        if self.pre_load:
            mix_audio = self.audios[uid]['mix']
            s1_audio = self.audios[uid]['s1']
            s2_audio = self.audios[uid]['s2']
        else:
            mix_path = os.path.join(self.audio_root, self.data[uid]['mix'][0])
            s1_path = os.path.join(self.audio_root, self.data[uid]['s1'][0])
            s2_path = os.path.join(self.audio_root, self.data[uid]['s2'][0])

            if self.sp_factors != None:
                factor = random.choice(self.sp_factors)
            else:
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

class wsj0_eval(Dataset):

    def __init__(self, id_list_path, audio_root, pre_load = True):
        """
        Args:
            id_list_path     : id_list from data/wsj0/preprocess.py
            audio_root       : root dir for wsj0 dataset
            pre_load         : pre load all audio into RAM
        """
        super(wsj0_eval, self).__init__()

        self.data = cPickle.load(open(id_list_path, 'rb'))
        self.audio_root = audio_root
        self.sr = 8000

        # get maxlen
        self.maxlen = 0
        for uid in self.data:
            l = self.data[uid]['mix'][1]
            if l > self.maxlen:
                self.maxlen = l

        self.pre_load = pre_load

        self.id_list = []
        for uid in self.data:
            # Same template with wsj0
            l = self.data[uid]['mix'][1]
            info = [ uid, uid, 0, l ]
            self.id_list.append(info)

        # sort list so that dataloader load from long to short
        # also more effiencient when using batch
        self.id_list.sort(key = lambda x: x[3], reverse = True)

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
        base = np.zeros(self.maxlen, dtype = np.float32)
        base[:ilen] = audio
        return base

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        info struct: [ utt id, chunk id, start, end ]
        """
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
        if ilen < self.maxlen:
            mix_audio = self.pad_audio(mix_audio, ilen)
            s1_audio = self.pad_audio(s1_audio, ilen)
            s2_audio = self.pad_audio(s2_audio, ilen)

        sep_audio = np.stack([s1_audio, s2_audio], axis = 0)

        sample = { 'uid': uid, 'cid': cid, 'ilens': ilen,
                   'mix': mix_audio, 'ref': sep_audio }
        return sample

    def get_sample_by_uid(self, uid):
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
        if ilen < self.maxlen:
            mix_audio = self.pad_audio(mix_audio, ilen)
            s1_audio = self.pad_audio(s1_audio, ilen)
            s2_audio = self.pad_audio(s2_audio, ilen)

        sep_audio = np.stack([s1_audio, s2_audio], axis = 0)

        mix_audio = torch.FloatTensor([mix_audio])
        sep_audio = torch.FloatTensor([sep_audio])
        ilen = torch.LongTensor([ilen])

        sample = { 'uid': uid, 'ilens': ilen,
                   'mix': mix_audio, 'ref': sep_audio }
        return sample

if __name__ == '__main__':

    from torch.utils.data import DataLoader

    trainset = wsj0('./data/wsj0/id_list/tr.pkl', pre_load = False, one_chunk_in_utt = True)
    print(len(trainset))

    print(trainset[0])

    trainloader = DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
    print(len(trainloader))

    for sample in trainloader:
        print(sample)
        mix = sample['mix']
        ref = sample['ref']

        print(mix.size())
        print(ref.size())
        break

    testset = wsj0('./data/wsj0/id_list/cv.pkl', pre_load = False, one_chunk_in_utt = False, mode = 'tt')
    print(len(testset))
    print(testset[0])

    testloader = DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 2)
    print(len(testloader))
    for sample in testloader:
        print(sample)
        mix = sample['mix']
        ref = sample['ref']

        print(mix.size())
        print(ref.size())
        break
