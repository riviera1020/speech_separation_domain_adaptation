import os
import argparse
import numpy as np
import soundfile as sf
import _pickle as cPickle
from tqdm import tqdm

def fetch_info(args):
    sr_dirs = [ '8k' ]
    datalen_dirs = [ 'min' ]

    for splt in [ 'tr', 'cv', 'tt' ]:
        data_list = os.path.join('./data/wsj0/id_list/', f'{splt}.pkl')
        wsj_datas = cPickle.load(open(data_list, 'rb'))

        scaling_npz_path = os.path.join(args.noise_dir, 'metadata', f'scaling_{splt}.npz')
        scaling_npz = np.load(scaling_npz_path, allow_pickle=True)

        # wsj0 dir must contain noise/ under tr/, cv/, tt
        # noise/ create from wham_noise and downsample to 8k without scaling
        noise_root = os.path.join(args.wsj0_dir, splt, 'noise/')

        for sr_dir in sr_dirs:
            if sr_dir == '8k':
                sr = 8000
            else:
                sr = 16000

            for datalen_dir in datalen_dirs:
                wsjmix_key = 'scaling_wsjmix_{}_{}'.format(sr_dir, datalen_dir)
                wham_speech_key = 'scaling_wham_speech_{}_{}'.format(sr_dir, datalen_dir)
                wham_noise_key = 'scaling_wham_noise_{}_{}'.format(sr_dir, datalen_dir)

                utt_ids = scaling_npz['utterance_id']
                # scale for noise
                scaling_noise = scaling_npz[wham_noise_key]

                # scale for speech
                scaling_speech = scaling_npz[wham_speech_key]

                # scale for wsjmix (from scratch)
                scaling_wsjmix = scaling_npz[wsjmix_key]

                data = {}
                data['header'] = [ 'apath', 'len', 'scale_speech', 'scale_noise', 'scale_wsjmix' ]
                for i_utt, output_name in enumerate(tqdm(utt_ids)):
                    mix_info = wsj_datas[output_name]
                    wsj_len = mix_info['mix'][1]

                    path = os.path.join(noise_root, output_name)
                    samples, sr = sf.read(path)
                    l = len(samples)

                    if wsj_len > l:
                        print(splt, output_name)
                        print(wsj_len, l)

                    ss = scaling_speech[i_utt]
                    sn = scaling_noise[i_utt]
                    sw = scaling_wsjmix[i_utt]

                    apath = path.replace(args.wsj0_dir, '')
                    data[output_name] = { 'noise': [ apath, l, ss, sn, sw ] }

                out_name = os.path.join(args.out_dir, f'{splt}.pkl')
                cPickle.dump(data, open(out_name, 'wb'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_dir', type=str, default=None,
                        help='Directory path of wham_noise')
    parser.add_argument('--wsj0_dir', type=str, default=None,
                        help='Directory path of wsj0')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory path to put output files')
    parser.add_argument('--sample_rate', type=int, default=8000,
                        help='Sample rate of audio file')
    args = parser.parse_args()
    return args

args = parse_args()
fetch_info(args)
