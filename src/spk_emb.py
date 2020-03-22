import torch
import torch.nn as nn
import librosa
import numpy as np
from torchaudio.transforms import Spectrogram

class Audio():
    def __init__(self):
        # change to 8000 and adjust hop&win length
        self.sr = 8000
        self.n_fft = 512
        self.n_mels = 40
        self.win_length = 200
        self.hop_length = 80
        self.mel_basis = librosa.filters.mel(sr=8000,
                                             n_fft=self.n_fft,
                                             n_mels=self.n_mels)

    def get_mel(self, y):
        y = librosa.core.stft(y=y, n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length,
                              window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        return mel

class Mel(nn.Module):
    def __init__(self):
        super(Mel, self).__init__()
        self.sr = 8000
        self.n_fft = 512
        self.n_mels = 40
        self.win_length = 200
        self.hop_length = 80
        self.mel_basis = librosa.filters.mel(sr=8000,
                                             n_fft=self.n_fft,
                                             n_mels=self.n_mels)
        self.mel_basis = nn.Parameter(torch.FloatTensor(self.mel_basis), requires_grad = False)
        self.spectrogram = Spectrogram(n_fft=self.n_fft,
                                       win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       pad=0,
                                       power=2)

    def forward(self, y):
        y = self.spectrogram(y)
        y = torch.mm(self.mel_basis, y)
        mel = torch.log10(y + 1e-6)
        return mel

class LinearNorm(nn.Module):
    def __init__(self, hidden, emb_dim):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(hidden, emb_dim)

    def forward(self, x):
        return self.linear_layer(x)

class SpeechEmbedder(nn.Module):
    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        # use pretrain so fix
        self.num_mels = 40
        self.lstm_hidden = 768
        self.lstm_layers = 3
        self.emb_dim = 256
        self.lstm = nn.LSTM(self.num_mels,
                            self.lstm_hidden,
                            num_layers=self.lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(self.lstm_hidden, self.emb_dim)

        self.window = 80
        self.stride = 40

    def forward(self, mel):
        # (num_mels, T)
        # TODO, change to access length, and batchify
        mels = mel.unfold(1, self.window, self.stride) # (num_mels, T', window)
        mels = mels.permute(1, 2, 0) # (T', window, num_mels)
        x, _ = self.lstm(mels) # (T', window, lstm_hidden)
        x = x[:, -1, :] # (T', lstm_hidden), use last frame only
        x = self.proj(x) # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True) # (T', emb_dim)
        x = x.sum(0) / x.size(0) # (emb_dim), average pooling over time frames
        return x

if __name__ == '__main__':
    import soundfile as sf
    from torchaudio.transforms import MelSpectrogram

    pt = './embedder.pt'
    pt = torch.load(pt)
    embedder = SpeechEmbedder().cuda()
    embedder.load_state_dict(pt)
    embedder.eval()

    mel_ext = Audio()

    apath = '/home/riviera1020/Big/Corpus/wsj0-mix/cv/mix/011a010d_0.54422_20do010c_-0.54422.wav'

    audio, _= sf.read(apath)

    #mel = mel_ext.get_mel(audio)
    #print(mel.shape)

    audio = torch.from_numpy(audio).float().cuda()
    torch_mel = Mel().cuda()
    mel = torch_mel(audio)

    print(mel.size())

    dvec = embedder(mel)
    print(dvec.size())

