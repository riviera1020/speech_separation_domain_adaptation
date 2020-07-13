
import _pickle as cPickle

splts = ['tr', 'cv', 'tt']

for splt in splts:

    p = f'./data/wsj0-vctk/id_list/{splt}.pkl'

    data = cPickle.load(open(p, 'rb'))

    print(len(data))
