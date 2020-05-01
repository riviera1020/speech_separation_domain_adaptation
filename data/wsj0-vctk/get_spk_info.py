
import _pickle as cPickle

w_info = './data/wsj0/spk_gender.pkl'
v_info = './data/vctk/spk_gender.pkl'

info = {}

for p in [ w_info, v_info ]:
    data = cPickle.load(open(p, 'rb'))
    for spk, gender in data.items():
        spk = spk.lower()
        if spk in info:
            print('Warning!!! Dup Spk in WSJ0, VCTK')
            print(spk)
        info[spk] = gender

print(info)
cPickle.dump(info, open('./data/wsj0-vctk/spk_gender.pkl', 'wb'))
