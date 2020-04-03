
import _pickle as cPickle

spk_info = '/home/riviera1020/Big/Corpus/VCTK-Corpus/speaker-info.txt'
info = {}
with open(spk_info) as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.rstrip()
        spk, _, gender  = line.split()[:3]
        spk = 'p' + spk
        info[spk] = gender

cPickle.dump(info, open('./data/vctk/spk_gender.pkl', 'wb'))
