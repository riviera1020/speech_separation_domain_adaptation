
import json

def get_gender(spk, gender_info):
    if spk in gender_info:
        return gender_info[spk]
    else:
        spk = spk.upper()
        return gender_info[spk]

def read_mix(path):

    ret = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()

            s1, _, s2, _ = line.split()
            ret.append((s1, s2))
    return ret

def check_info(mix_path, gender_info, mode):
    data = read_mix(mix_path)

    diff = []
    all_M = []
    all_F = []
    same_spk = []
    for s1, s2 in data:

        s1_spk = s1.split('/')[2]
        s2_spk = s2.split('/')[2]
        s1_g = get_gender(s1_spk, gender_info)
        s2_g = get_gender(s2_spk, gender_info)

        assert s1_g in [ 'M', 'F' ]
        assert s2_g in [ 'M', 'F' ]

        if s1_g != s2_g:
            diff.append((s1, s2))
        else:
            if s1_spk == s2_spk:
                same_spk.append((s1, s2))
            if s1_g == 'M':
                all_M.append((s1, s2))
            else:
                all_F.append((s1, s2))

    # print info
    print('====================')
    print(f'Mode     : {mode}')
    print(f'Total    : {len(data)}')
    print(f'Diff     : {len(diff)}')
    print(f'M        : {len(all_M)}')
    print(f'F        : {len(all_F)}')
    print(f'Same spk : {len(same_spk)}')


gender_info = json.load(open('./wsj_info/spk_info.json'))

tr_2mix = '../make-wsj0-mix/mix_2_spk_tr.txt'
check_info(tr_2mix, gender_info, 'tr')

cv_2mix = '../make-wsj0-mix/mix_2_spk_cv.txt'
check_info(cv_2mix, gender_info, 'cv')

tt_2mix = '../make-wsj0-mix/mix_2_spk_tt.txt'
check_info(tt_2mix, gender_info, 'tt')
