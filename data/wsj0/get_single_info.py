
import _pickle as cPickle

def get_sep_uid(uid, dataset):
    if dataset == 'wsj0' or dataset == 'libri':
        s1, scale1, s2, scale2 = uid[:-4].split('_')
    elif dataset == 'vctk':
        s11, s12, scale1, s21, s22, scale2 = uid[:-4].split('_')
        s1 = f'{s11}_{s12}'
        s2 = f'{s21}_{s22}'

    scale1 = float(scale1)
    scale2 = float(scale2)
    return s1, s2, scale1, scale2

def get_spk(uid, dataset):
    if dataset == 'wsj0':
        return uid[:3]
    elif dataset == 'vctk':
        return uid.split('_')[0]
    elif dataset == 'libri':
        return uid.split('-')[0]

def update_spk(out, s):
    if s not in out:
        out[s] = {}

def main(data, dataset):
    data = cPickle.load(open(data, 'rb'))
    out = {}
    for uid in data:
        sample = data[uid]

        u1, u2, scale1, scale2 = get_sep_uid(uid, dataset)
        s1 = get_spk(u1, dataset)
        s2 = get_spk(u2, dataset)

        update_spk(out, s1)
        update_spk(out, s2)

        out[s1][u1] = sample['s1'] + [ scale1 ]
        out[s2][u2] = sample['s2'] + [ scale2 ]

    tr_spks = list(out.keys())
    tr_spk_num = len(tr_spks)

    lens = []
    for spk in out:
        lens.append(len(out[spk]))

    print(tr_spk_num)
    print(lens)

    cPickle.dump(out, open(f'./data/{dataset}/single_list/tr.pkl', 'wb'))

dataset = 'wsj0'
data = f'./data/{dataset}/id_list/tr.pkl'

main(data, dataset)
