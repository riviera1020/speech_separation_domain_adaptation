import os
import argparse
import soundfile as sf
import _pickle as cPickle
from glob import glob
from tqdm import tqdm

def preprocess(args):
    for data_type in ['tr', 'cv', 'tt']:
        out_name = os.path.join(args.out_dir, f'{data_type}.pkl')
        data = {}

        for speaker in ['mix', 's1', 's2']:
            paths = glob(os.path.join(args.in_dir, data_type, speaker, '*.wav'))

            for path in tqdm(paths):
                uid = path.split('/')[-1]
                if uid not in data:
                    data[uid] = { 'mix': [], 's1': [], 's2': [] }

                samples, sr = sf.read(path)

                if sr != args.sample_rate:
                    print('Error')
                    exit()

                apath = path.replace(args.in_dir, '')
                data[uid][speaker] = [ apath, len(samples) ]

        cPickle.dump(data, open(out_name, 'wb'))

def parse_args():
    parser = argparse.ArgumentParser("VCTK data preprocessing")
    parser.add_argument('--in_dir', type=str, default=None,
                        help='Directory path of vctk-mix including tr, cv and tt')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory path to put output files')
    parser.add_argument('--sample_rate', type=int, default=8000,
                        help='Sample rate of audio file')
    args = parser.parse_args()
    return args

args = parse_args()
preprocess(args)
