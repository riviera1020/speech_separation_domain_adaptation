
import os
import soundfile as sf
import _pickle as cPickle
from tqdm import tqdm

root = '/home/riviera1020/Big/Corpus/vctk-mix/wav8k/min/'

for splt in [ 'tr', 'cv', 'tt' ]:
    data = cPickle.load(open(f'./data/vctk/id_list/{splt}.pkl', 'rb'))

    utt_num = len(data)
    total_len = 0
    for uid, info in tqdm(data.items()):
        apath = info['mix'][0]
        apath = os.path.join(root, apath)

        audio, sr = sf.read(apath)
        total_len += ( float(len(audio)) / sr )


    total_d = total_len / 3600

    print(splt)
    print(utt_num)
    print(total_d)

