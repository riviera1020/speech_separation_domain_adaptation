
import os
import argparse
import _pickle as cPickle

def split_uids(mix_uid, dset):
    if dset == 'vctk':
        text = mix_uid.split('_')
        s1 = text[0] + '_' + text[1]
        s2 = text[3] + '_' + text[4]
    elif dset == 'wsj0':
        text = mix_uid.split('_')
        s1 = text[0]
        s2 = text[2]
    return s1, s2

def parse_args():
    parser = argparse.ArgumentParser("Check and gen those uid has force align info")
    parser.add_argument('--dset', type=str, default=None,
                        help='Dataset: wsj0/vctk')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Path write uids')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    split = [ 'tr', 'tt', 'cv' ]

    for s in split:
        path = f'./data/{args.dset}/id_list/{s}.pkl'
        data = cPickle.load(open(path, 'rb'))

        phn_data = cPickle.load(open(f'./data/{args.dset}/phn.pkl', 'rb'))
        out = {}
        cnt = 0
        for uid in data:
            d = data[uid]

            uid = uid.rsplit('.', 1)[0]
            s1, s2 = split_uids(uid, args.dset)

            if s1 in phn_data and s2 in phn_data:
                out[uid] = d
            else:
                cnt += 1

        cPickle.dump(out, open(os.path.join(args.out_dir, f'{s}.pkl'), 'wb'))

        p = float(cnt) / len(data)
        print(f'{args.dset} {s} | remove percentage {p}')

main()
