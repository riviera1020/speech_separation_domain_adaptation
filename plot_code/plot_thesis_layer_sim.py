
import os
import json
import matplotlib.pyplot as plt

from glob import glob

plt.rcParams['font.sans-serif'] = [ 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-colorblind')

lw = 1.5
alpha = 0.8
baseline = './layer_sim/baseline_result.json'
splts = [ 'cv', 'tt' ]
layer_order = [ 'enc' ] + [ str(i) for i in range(32) ]
layer_index = [ '編碼器' ] + [ str(i) for i in range(1, 33) ]

def get_sim(r, dset, splt):
    result = json.load(open(r))
    sim = result[dset][splt]['total_layer_cos_sim']
    sim = [ sim[l] for l in layer_order ]
    return sim

def plot_diff_norm():
    dsets = [ 'wham', 'wham-easy' ]
    for dset in dsets:
        for splt in splts:

            d = 'wham_easy' if dset == 'wham-easy' else dset
            ln = f'./layer_sim/diff_norm/{d}-ln-g05.json'
            sn = f'./layer_sim/diff_norm/{d}-sn-g05.json'
            wn = f'./layer_sim/diff_norm/{d}-wn-g05.json'

            bs_sim = get_sim(baseline, dset, splt)
            ln_sim = get_sim(ln, dset, splt)
            sn_sim = get_sim(sn, dset, splt)
            wn_sim = get_sim(wn, dset, splt)

            plt.figure(figsize = [12.8, 9.6])
            plt.title('不同正規化對層相似度的影響', fontsize = 'xx-large')
            plt.xlabel('層索引', fontsize = 'x-large')
            plt.ylabel('特徵相似度', fontsize = 'x-large')
            plt.xticks(range(33), layer_index, fontsize = 'small')
            plt.ylim([0, 1.1])

            plt.plot(sn_sim, alpha = alpha, linewidth = lw, label = '譜正規化')
            plt.plot(ln_sim, alpha = alpha, linewidth = lw, label = '全域層正規化')
            plt.plot(wn_sim, alpha = alpha, linewidth = lw, label = '權重正規化')

            plt.plot(bs_sim, alpha = alpha, linewidth = lw, label = '基礎模型')
            plt.legend(loc = 0, fontsize = 'x-large')

            #plt.show()
            plt.savefig(f'./plot/c4/norm_layer_sim_{dset}_{splt}.pdf', format = 'pdf')

def plot_diff_weight():
    dsets = [ 'wham', 'wham-easy' ]
    for dset in dsets:
        for splt in splts:

            d = 'wham_easy' if dset == 'wham-easy' else dset
            g1 = f'./layer_sim/diff_weight/{d}-ln-g1.json'
            g05 = f'./layer_sim/diff_weight/{d}-ln-g05.json'
            g01 = f'./layer_sim/diff_weight/{d}-ln-g01.json'

            bs_sim = get_sim(baseline, dset, splt)
            g1_sim = get_sim(g1, dset, splt)
            g05_sim = get_sim(g05, dset, splt)
            g01_sim = get_sim(g01, dset, splt)

            plt.figure(figsize = [12.8, 9.6])
            plt.title(r'不同$\lambda_G$對層相似度的影響', fontsize = 'xx-large')
            plt.xlabel('層索引', fontsize = 'x-large')
            plt.ylabel('特徵相似度', fontsize = 'x-large')
            plt.xticks(range(33), layer_index, fontsize = 'small')
            plt.ylim([0, 1.1])

            plt.plot(g1_sim, alpha = alpha, linewidth = lw, label = r'$\lambda_G = 1$')
            plt.plot(g05_sim, alpha = alpha, linewidth = lw, label = r'$\lambda_G = 0.5$')
            plt.plot(g01_sim, alpha = alpha, linewidth = lw, label = f'$\lambda_G = 0.1$')

            plt.plot(bs_sim, alpha = alpha, linewidth = lw, label = '基礎模型')
            plt.legend(loc = 2, fontsize = 'x-large')
            #plt.show()
            plt.savefig(f'./plot/c4/weight_layer_sim_{dset}_{splt}.pdf', format = 'pdf')

def plot_diff_layer_pos():
    dsets = [ 'wham', 'wham-easy' ]
    for dset in dsets:
        for splt in splts:

            d = 'wham_easy' if dset == 'wham-easy' else dset
            l4 = f'./layer_sim/diff_layer_pos/{d}-ln-g05-2932.json'
            p4 = f'./layer_sim/diff_layer_pos/{d}-ln-g05-2528.json'
            al = f'./layer_sim/diff_layer_pos/{d}-ln-g05-alllayer.json'

            bs_sim = get_sim(baseline, dset, splt)
            l4_sim = get_sim(l4, dset, splt)
            p4_sim = get_sim(p4, dset, splt)
            al_sim = get_sim(al, dset, splt)

            plt.figure(figsize = [12.8, 9.6])
            plt.title('生成器選擇的層位置對層相似度的影響', fontsize = 'xx-large')
            plt.xlabel('層索引', fontsize = 'x-large')
            plt.ylabel('特徵相似度', fontsize = 'x-large')
            plt.xticks(range(33), layer_index, fontsize = 'small')
            plt.ylim([0, 1.1])

            plt.plot(l4_sim, alpha = alpha, linewidth = lw, label = '29 - 32')
            plt.plot(p4_sim, alpha = alpha, linewidth = lw, label = '25 - 28')
            plt.plot(al_sim, alpha = alpha, linewidth = lw, label = '全部')

            plt.plot(bs_sim, alpha = alpha, linewidth = lw, label = '基礎模型')
            plt.legend(loc = 4, fontsize = 'x-large')

            #plt.show()
            plt.savefig(f'./plot/c4/pos_layer_sim_{dset}_{splt}.pdf', format = 'pdf')

#plot_diff_norm()
#plot_diff_weight()
plot_diff_layer_pos()
