import os
import yaml
import torch
import numpy as np
import soundfile as sf
import _pickle as cPickle
import matplotlib.pyplot as plt

def get_root(d):
    d = 'wsj' if d == 'wsj0' else d
    c = './config/path.yaml'
    c = yaml.load(open(c), Loader=yaml.FullLoader)
    r = c[f'{d}_root']
    return r

def get_data(root, uid, data):
    def load_audio(path):
        audio, _ = sf.read(path)
        audio = audio.astype(np.float32)
        return audio

    d = data[uid]
    mix_audio = load_audio(os.path.join(root, d['mix'][0]))
    s1_audio = load_audio(os.path.join(root, d['s1'][0]))
    s2_audio = load_audio(os.path.join(root, d['s2'][0]))
    ref_audio = np.stack([s1_audio, s2_audio], axis = 0)
    l = len(mix_audio)

    sample = { 'uid': uid, 'ilens': l, 'mix': mix_audio, 'ref': ref_audio }
    return sample

def ps(s, name):
    plt.plot(s)
    plt.savefig(f'./plot/{name}.png')

def main():
    dset = 'wsj0'
    root = get_root(dset)
    data_list = f'./data/{dset}/id_list/tt.pkl'
    data = cPickle.load(open(data_list, 'rb'))

    uid = list(data.keys())[0]
    sample = get_data(root, uid, data)
    ref = sample['ref']

    s1 = ref[0]
    s2 = ref[1]
    s1 = s1[:42500]
    s2 = s2[8100:]

    uid = list(data.keys())[1]
    sample = get_data(root, uid, data)
    ref = sample['ref']
    s3 = ref[0][:42500]

    plt.figure(figsize= [12.8, 9.6])
    plt.axis('off')
    plt.plot(s1, 'b', alpha = 0.75)
    plt.plot(s2, 'r', alpha = 0.75)
    plt.plot(s3, color = 'orange', alpha = 0.75)
    #plt.show()
    plt.savefig(f'./plot/wav_png/mix3.png', transparent = True)
    exit()

    s1 = ref[0]
    s2 = ref[1]
    s1 = s1[:42500]
    s2 = s2[8100:]

    s = 20000
    x2 = list(range(s, s + s2.shape[0]))

    x = list(range(x2[-1] + 1))
    y = np.zeros(len(x))

    l = s1.shape[0] - s
    s2 = s2[l:]
    s1 = s1[:20000]

    s = 1500
    x2 = list(range(s, s + s2.shape[0]))

    plt.figure(figsize= [12.8, 9.6])
    plt.axis('off')
    #plt.plot(s1, 'b', alpha = 0.75)
    plt.plot(x, y, alpha = 0)
    plt.plot(s1, 'b', alpha = 0.75)
    plt.plot(x2, s2, 'r', alpha = 0.75)
    #plt.show()
    plt.savefig(f'./plot/wav_png/overlap_remix.png', transparent = True)


def main_badresult():
    dset = 'wsj0'
    root = get_root(dset)
    data_list = f'./data/{dset}/id_list/tt.pkl'
    data = cPickle.load(open(data_list, 'rb'))

    uid = list(data.keys())[0]
    sample = get_data(root, uid, data)
    ref = sample['ref']
    scale = 0.3

    s1 = ref[0]
    s2 = ref[1]
    s1 = s1[:42500]
    s2 = s2[8100:]

    #s = 1500
    #x2 = list(range(s, s + s2.shape[0]))

    #x = list(range(x2[-1] + 1))
    #y = np.zeros(len(x))

    #l = s1.shape[0] - s
    #s2 = s2[l:]
    #s1 = s1[:1500]

    plt.figure(figsize= [12.8, 9.6])
    plt.axis('off')
    plt.plot(s2, 'r', alpha = 0)
    plt.plot(scale * s2, 'r', alpha = 0.75)
    plt.savefig(f'./plot/wav_png/red_nogood.png', transparent = True)
    plt.close()

    plt.figure(figsize= [12.8, 9.6])
    plt.axis('off')
    plt.plot(s1, 'b', alpha = 0)
    plt.plot(scale * s1, 'b', alpha = 0.75)
    plt.savefig(f'./plot/wav_png/blue_nogood.png', transparent = True)
    plt.close()

#main()

main_badresult()


