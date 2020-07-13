import numpy as np
from torch.utils.data import Dataset

class Combination(Dataset):

    def __init__(self, dsets):
        """
        Args:
            dsets: list of Dataset object
        """
        super(Combination, self).__init__()

        self.dsets = dsets
        self.id_list = []
        seg_lens = []
        for i_dset, dset in enumerate(dsets):
            seg_lens.append(dset.seg_len)

            for idx in range(len(dset)):
                self.id_list.append((i_dset, idx))

        self.seg_len = max(seg_lens)

    def pad_audio(self, audio, ilen):
        base = np.zeros(self.seg_len, dtype = np.float32)
        base[:ilen] = audio
        return base

    def pad_ref_audio(self, audio, c, ilen):
        base = np.zeros((c, self.seg_len), dtype = np.float32)
        base[:, :ilen] = audio
        return base

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        info struct: [ utt id, chunk id, start, end ]
        """

        i_dset, sub_id = self.id_list[idx]
        sample = self.dsets[i_dset][sub_id]
        ilens = sample['ilens']
        if ilens < self.seg_len:
            mix_audio = sample['mix']
            mix_audio = self.pad_audio(mix_audio, ilens)

            ref_audio = sample['ref']
            ref_audio = self.pad_ref_audio(ref_audio, ref_audio.shape[0], ilens)

            sample['mix'] = mix_audio
            sample['ref'] = ref_audio

        return sample

