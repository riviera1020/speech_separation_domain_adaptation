import os
import json
import soundfile as sf

from glob import glob
from tqdm import tqdm

_root = '../VCTK-Corpus/'
vctk_root = '../VCTK-Corpus/wav48/'

spk_paths = glob(os.path.join(vctk_root, '*'))
ret = {}

total_t = 0
for spk_path in tqdm(spk_paths):
    paths = glob(os.path.join(spk_path, '*.wav'))
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

json.dump(ret, open('./vctk_duration.json', 'w'), indent = 1)

