
import os
import soundfile as sf
import _pickle as cPickle

ori = './data/wsj0/id_list/tr.pkl.bak'
data = cPickle.load(open(ori, 'rb'))

wsj_root = '/home/riviera1020/Big/Corpus/wsj0-mix/'

l = []
with open('./data/wsj0/miss_list.txt') as f:
    for line in f:
        line = line.rstrip()
        rm, add = line.split(',')
        l.append((rm, add))

for rm, add in l:

    rm_info = data[rm]

    add_info = { 'mix': [], 's1': [], 's2': [] }
    for speaker in [ 'mix', 's1', 's2' ]:
        path = os.path.join(wsj_root, 'tr', speaker, add)
        samples, sr = sf.read(path)

        apath = path.replace(wsj_root, '')
        add_info[speaker] = [ apath, len(samples) ]

    print(rm_info)
    print(add_info)

    data.pop(rm)
    data[add] = add_info

print(len(data))
cPickle.dump(data, open('./data/wsj0/id_list/tr.pkl', 'wb'))
