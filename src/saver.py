
import os
import torch

from glob import glob
from functools import cmp_to_key

class Saver(object):

    def __init__(self, max_save_num, save_dir, keep, resume = False, resume_score_fn = None):
        # keep: min, max
        assert keep in ['min', 'max']

        self.max_save_num = max_save_num
        self.save_dir = save_dir
        self.save_list = []
        self.keep = keep

        if keep == 'min':
            self.reverse = True
        else:
            self.reverse = False

        if resume:
            # resume_score_fn: a callable function to get score for sorting
            if resume_score_fn == None:
                print('Specify function to get sorting score')

            # exclude latest.pth
            paths = glob(os.path.join(self.save_dir, '[!latest]*.pth'))
            save_list = []
            for path in paths:
                info_dict = torch.load(path)
                score = resume_score_fn(info_dict)
                save_list.append({'score': score, 'path': path})
            save_list = sorted(save_list, key = lambda x: x['score'], reverse = self.reverse)
            self.save_list = save_list[:self.max_save_num]

    def force_save(self, model, model_name, info_dict = None):
        path = os.path.join(self.save_dir, model_name)
        self.save(model, path, info_dict)

    def logging(self):
        with open(os.path.join(self.save_dir, 'save.log'), 'w') as f:
            for item in self.save_list[::-1]:
                s = item['score']
                p = item['path']
                f.write(f'{p}: {s}\n')

    def save(self, model, path, info_dict = None):

        if info_dict is None:
            info_dict = { 'state_dict': model.state_dict() }
        else:
            info_dict['state_dict'] = model.state_dict()

        torch.save(info_dict, path)

    def update(self, model, score, model_name, info_dict = None):

        path = os.path.join(self.save_dir, model_name)

        if len(self.save_list) < self.max_save_num:
            item = { 'score': score, 'path': path }
            self.save_list.append(item)
            self.save(model, path, info_dict)

            if len(self.save_list) == self.max_save_num:
                self.save_list = sorted(self.save_list, key = lambda x: x['score'], reverse = self.reverse)
        else:
            m_score = self.save_list[0]['score']

            if self.keep == 'min':
                flag = score < m_score
            else:
                flag = score > m_score

            if flag:
                os.remove(self.save_list[0]['path'])
                self.save_list[0] = { 'score': score, 'path': path }
                self.save(model, path, info_dict)
                self.save_list = sorted(self.save_list, key = lambda x: x['score'], reverse = self.reverse)

        self.logging()

    @staticmethod
    def simple_comp(x, y):
        x, y = x['score'], y['score']
        if x > y:
            return 1
        elif x == y:
            return 0
        else:
            return -1

if __name__ == '__main__':

    import torch.nn as nn
    import random

    class m(nn.Module):
        def __init__(self):
            super(m, self).__init__()
            self.nn = nn.Linear(100, 100)

    saver = Saver(5, './temp/', 'max')
    model = m()

    check = []

    for i in range(20):

        score = random.randint(0, 100)
        model_name = str(i) + '.pth'
        saver.update(model, score, model_name)

        check.append((score, model_name))

    check = sorted(check, key = lambda x: x[0], reverse = True)
    for c in check:
        print(c)


