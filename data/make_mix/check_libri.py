import os
import json
import soundfile as sf

from glob import glob
from tqdm import tqdm

_root = '../LibriSpeech/'

dsets = [ 'train-clean-100', 'dev-clean', 'test-clean' ]
spkss = []
for dset in dsets:
    libri_root = os.path.join(_root, dset)
    spks = glob(os.path.join(libri_root, '*'))

    spks = [ spk_p.split('/')[-1] for spk_p in spks ]
    spkss.append(set(spks))

tr, cv, tt = spkss

print(tr & cv)
print(tr & tt)
print(cv & tr)

"""
Result:
    LibriSpeech doesn't have same spk in diff subset
"""
