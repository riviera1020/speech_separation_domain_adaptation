
import _pickle as cPickle

class PhoneVocab():
    def __init__(self):

        self.vocab = {}
        with open('./data/cmu.phones') as f:
            for i, line in enumerate(f):
                phn = line.rstrip()
                self.vocab[phn] = i

        self.stress = {}
        for phn in self.vocab.keys():
            self.stress[phn] = phn
            for i in range(3):
                ps = f'{phn}{i}'
                self.stress[ps] = phn

    def index(self, ps):
        phn = self.stress[ps]
        return self.vocab[phn]
vocab = PhoneVocab()

class PhoneMapper():

    def __init__(self, phn_path, sample_rate, dset, win_length, hop_length):
        """
        n_fft : always == win_length
        center: stft center arg (Always True because torch fix this)
        """

        self.phns = cPickle.load(open(phn_path, 'rb'))

        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = True

        self.dset = dset

    def split_uids(self, mix_uid):
        if self.dset == 'vctk':
            text = mix_uid.split('_')
            s1 = text[0] + '_' + text[1]
            s2 = text[3] + '_' + text[4]
        elif self.dset == 'wsj0':
            text = mix_uid.split('_')
            s1 = text[0]
            s2 = text[2]
        return s1, s2

    def get_frame_num(self, duration):
        if self.center:
            return (duration // self.hop_length) + 1
        else:
            return duration // self.hop_length

    def get_label_by_uid(self, uid, s, e):
        def in_interval(t, phn_info):
            s = phn_info[1]
            e = phn_info[2]
            if s <= t and t < e:
                return True
            else:
                return False

        duration = e - s
        frame_num = self.get_frame_num(duration)
        phns = self.phns[uid]
        ret = []

        #print(s, e)
        #print(float(s) / self.sample_rate, float(e) / self.sample_rate)
        #print(phns)

        all_sil = True

        p_idx = 0
        out_of_bound = False
        for f_idx in range(frame_num):
            t = float(f_idx * self.hop_length + s) / self.sample_rate

            while not in_interval(t, phns[p_idx]):
                if p_idx == 0 and t < phns[p_idx][1]:
                    out_of_bound = True
                    break
                elif p_idx == len(phns) - 1:
                    out_of_bound = True
                    break
                else:
                    p_idx += 1
                    out_of_bound = False

            if out_of_bound:
                phn = 'SIL'
            else:
                phn = phns[p_idx][0]

            phn = vocab.index(phn)

            if phn != 39:
                all_sil = False

            ret.append(phn)

        #if all_sil:
        #    print(s, e)
        #    print(float(s) / self.sample_rate, float(e) / self.sample_rate)
        #    print(phns)
        #    print(ret)
        #    exit()

        return ret

    def get_label(self, uid, s, e):
        """
        This uid is mix uid in separation corpus
        """
        s1, s2 = self.split_uids(uid)

        s1_phns = self.get_label_by_uid(s1, s, e)
        s2_phns = self.get_label_by_uid(s2, s, e)
        return s1_phns, s2_phns
