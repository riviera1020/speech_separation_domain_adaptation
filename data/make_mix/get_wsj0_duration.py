import os
import json
import soundfile as sf

from glob import glob
from tqdm import tqdm

ret = {}
for splt in [ 'tr', 'cv', 'tt' ]:

    wavs = glob(f'../make-wsj0-mix/wav8k/min/{splt}/mix/*.wav')
    total_t = 0
    for upath in tqdm(wavs):
        audio, sr = sf.read(upath)
        l = len(audio)
        t = float(l) / sr

        total_t += t

    ret[splt] = total_t / 3600

json.dump(ret, open('./wsj_info/wsj0_dset_duration.json', 'w'), indent = 1)
exit()


wsj0_root = '../wsj0-clean-wav/'
all_wavs = glob('../make-wsj0-mix/mix_2_spk_*.txt')
print(all_wavs)

ret = {}
for wav_list in all_wavs:
    with open(wav_list) as f:
        for line in tqdm(f.readlines()):
            line = line.rstrip()

            s1, _, s2, _ = line.split()

            for s in [ s1, s2 ]:
                spk = s.split('/')[2]
                if spk not in ret:
                    ret[spk] = {}

                upath = os.path.join(wsj0_root, s)

                audio, sr = sf.read(upath)
                l = len(audio)
                t = float(l) / sr

                ret[spk][s] = t

total_t = 0
for spk in ret:
    spk_t = 0
    for uid in ret[spk]:
        spk_t += ret[spk][uid]
        total_t += ret[spk][uid]

    ret[spk]['total'] = spk_t

ret['total'] = total_t
ret['total_hr'] = total_t / 3600

json.dump(ret, open('./wsj_info/wsj0_duration.json', 'w'), indent = 1)
