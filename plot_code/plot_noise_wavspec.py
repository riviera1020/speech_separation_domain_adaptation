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


def p_nwav():
    nwav = np.random.randn(500)

    plt.figure(figsize= [12.8, 9.6])
    plt.axis('off')
    #plt.plot(s1, 'b', alpha = 0.75)
    plt.plot(nwav, c = 'black')
    #plt.show()
    plt.savefig(f'./plot/noise_wav.png', transparent = True)

def p_nspec():
    nspec = np.random.randn(256, 700)

    #plt.figure(figsize= [12.8, 9.6])
    plt.figure()
    plt.axis('off')
    plt.imshow(nspec)
    #plt.show()
    plt.savefig(f'./plot/noise_spec.png', transparent = True)

def main():
    #p_nwav()
    p_nspec()
main()


