import os
import _pickle as cPickle

def get_sep_uid(uid, dataset):
    if dataset == 'wsj0' or dataset == 'libri':
        s1, _, s2, _ = uid[:-4].split('_')
    elif dataset == 'vctk':
        s11, s12, _, s21, s22, _ = uid[:-4].split('_')
        s1 = f'{s11}_{s12}'
        s2 = f'{s21}_{s22}'
    elif dataset == 'wsj0-vctk':
        w1, w2, w3, w4, w5 = uid[:-4].split('_')
        if w1[0] == 'p':
            s1 = f'{w1}_{w2}'
            s2 = w4
        else:
            s1 = w1
            s2 = f'{w3}_{w4}'
    return s1, s2

def get_spk(uid, dataset):
    if dataset == 'wsj0':
        return uid[:3]
    elif dataset == 'vctk':
        return uid.split('_')[0]
    elif dataset == 'libri':
        return uid.split('-')[0]
    elif dataset == 'wsj0-vctk':
        if uid[0] == 'p':
            return uid.split('_')[0]
        else:
            return uid[:3]

class GenderMapper():
    def __init__(self):

        # all dsets
        self.dsets = [ 'wsj0', 'vctk', 'libri', 'wsj0-vctk' ]
        #['wham', 'wham-easy' ]

        self.info = {}
        for dset in self.dsets:
            pkl = os.path.join(f'./data/{dset}/spk_gender.pkl')
            data = cPickle.load(open(pkl, 'rb'))
            self.info[dset] = data

    def __call__(self, uid, dset):
        if 'wham' in dset:
            dset = 'wsj0'
        s1, s2 = get_sep_uid(uid, dset)
        s1 = get_spk(s1, dset)
        s2 = get_spk(s2, dset)

        g1 = self.info[dset][s1]
        g2 = self.info[dset][s2]

        if g1 != g2:
            r = 'MF'
        elif g1 == 'M':
            r = 'MM'
        else:
            r = 'FF'
        return r

if __name__ == '__main__':
    m = GenderMapper()

    splts = ['cv', 'tt']
    #dsets = [ 'wsj0', 'vctk', 'libri' ]
    dsets = [ 'wsj0-vctk']

    for dset in dsets:
        for splt in splts:
            p = os.path.join(f'./data/{dset}/id_list/{splt}.pkl')
            data = cPickle.load(open(p, 'rb'))

            for uid in data:
                g = m(uid, dset)
                print(g)

