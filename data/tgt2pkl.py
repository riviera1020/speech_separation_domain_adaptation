
import os
import tgt
import argparse
import _pickle as cPickle

from glob import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("TextGrid Data process")
    parser.add_argument('--in_dir', type=str, default=None,
                        help='Directory path of textgrid file')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Directory path to put output files')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    paths = glob(os.path.join(args.in_dir, '*.TextGrid'))
    out = {}

    for path in tqdm(paths):

        uid = path.split('/')[-1].split('.')[0]
        t = tgt.io.read_textgrid(path)
        phns = []

        for p in t.get_tier_by_name('phones'):

            phn = p.text
            if p.text == 'sp':
                phn = 'sil'
            phn = phn.upper()

            s = float(p.start_time)
            e = float(p.end_time)
            phns.append([phn, s, e])

        out[uid] = phns

    cPickle.dump(out, open(args.out_path, 'wb'))

main()
