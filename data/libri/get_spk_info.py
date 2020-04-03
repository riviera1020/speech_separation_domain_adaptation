
import _pickle as cPickle

spk_info = '/home/riviera1020/Big/Corpus/LibriSpeech/SPEAKERS.TXT'
info = {}

# only use these sets for data generation
used_sets = [ 'train-clean-100', 'dev-clean', 'test-clean']

with open(spk_info) as f:
    for i, line in enumerate(f):
        if line[0] == ';':
            continue
        line = line.rstrip()
        line = line.split('|')[:3]
        spk, gender, subset = [ s.strip() for s in line ]
        if subset in used_sets:
            info[spk] = gender

cPickle.dump(info, open('./data/libri/spk_gender.pkl', 'wb'))
