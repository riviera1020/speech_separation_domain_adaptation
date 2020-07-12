import os
import json
import soundfile as sf

from glob import glob
from tqdm import tqdm

_root = '../LibriSpeech/'

dsets = [ 'train-clean-100', 'dev-clean', 'test-clean' ]
spk_paths = []
for dset in dsets:
    libri_root = os.path.join(_root, dset)
    spk_paths += glob(os.path.join(libri_root, '*'))

ret = {}
total_t = 0
for spk_path in tqdm(spk_paths):
    paths = glob(os.path.join(spk_path, '*/*.flac'))
    spk = spk_path.split('/')[-1]
    ret[spk] = {}
    spk_t = 0

    for path in paths:

        uid = path.split('/')[-1]

        audio, sr = sf.read(path)

        l = len(audio)
        t = float(l) / sr

        ret[spk][uid] = t
        spk_t += t

    ret[spk]['total'] = spk_t
    total_t += spk_t

ret['total'] = total_t
ret['total_hr'] = total_t / 3600

json.dump(ret, open('./libri_info/libri_duration.json', 'w'), indent = 1)

