
import _pickle as cPickle

tr_info = '/home/riviera1020/Big/Corpus/WSJ/wsj0-train-spkrinfo.txt'
cv_info = '/home/riviera1020/Big/Corpus/WSJ/wsj0-eval-spkrinfo.txt'

info = {}

for p in [ tr_info, cv_info ]:
    with open(p) as f:
        for line in f:
            line = line.rstrip()
            if line[0] != ';' and line[0] != '-':
                spk, gender  = line.split()[:2]
                spk = spk.lower() # 01L to 01l
                info[spk] = gender

cPickle.dump(info, open('./data/wsj0/spk_gender.pkl', 'wb'))
