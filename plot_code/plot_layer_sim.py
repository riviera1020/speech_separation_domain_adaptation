import os
import json
import matplotlib.pyplot as plt


dset = 'wham'
baseline = './result/dagan_but_wsj0baseline_wham_iter1/result.json'
#comp = './result/chapter4/dagan-wham_easy-ln-loadpre80-10it-g00001_05-try2528_trycca_2/result.json'
comp = './result/chapter4/dagan-wham-dcgan-ln-g05/result.json'

layer_order = [ 'enc' ] + [ str(i) for i in range(32) ]

b_result = json.load(open(baseline))
c_result = json.load(open(comp))

b_result = b_result[dset]
c_result = c_result[dset]

#splts = [ 'cv', 'tt' ]
#sim_methods = [ 'cos_sim', 'pwcca', 'cka' ]

splts = [ 'cv' ]
sim_methods = [ 'pwcca', 'cka' ]

#for splt in splts:
#    for method in sim_methods:
#        k = f'total_layer_{method}'
#        b_score = b_result[splt][k]
#        b_score = [ b_score[l] for l in layer_order ]
#
#        c_score = c_result[splt][k]
#        c_score = [ c_score[l] for l in layer_order ]
#
#        ylabel = 'Cosine Similarity' if method == 'cos_sim' else method.upper()
#
#        plt.figure(figsize=[12.8, 9.6])
#        plt.title(f'{ylabel} on {splt}')
#        #plt.ylim([0.4, 1])
#        plt.xticks(range(33), layer_order)
#        plt.ylabel(ylabel)
#        plt.plot(b_score, label='Baseline')
#        plt.plot(c_score, label='Dagan')
#        plt.legend()
#        #plt.show()
#        plt.savefig(f'./plot/sim_png/{dset}_{splt}_{method}.png')
#        plt.close()

for splt in splts:
    plt.figure(figsize=[12.8, 9.6])
    plt.title(f'PWCCA / CKA on {splt}')
    plt.xticks(range(33), layer_order)

    for method in sim_methods:
        k = f'total_layer_{method}'
        b_score = b_result[splt][k]
        b_score = [ b_score[l] for l in layer_order ]

        c_score = c_result[splt][k]
        c_score = [ c_score[l] for l in layer_order ]

        plt.ylabel('Similarity')

        c1 = 'b' if method == 'pwcca' else '--b'
        c2 = 'r' if method == 'pwcca' else '--r'
        plt.plot(b_score, c1, label=f'Baseline {method.upper()}')
        plt.plot(c_score, c2, label=f'Dagan {method.upper()}')

    plt.legend()
    plt.savefig(f'./plot/sim_png/progress.png')
    plt.close()
