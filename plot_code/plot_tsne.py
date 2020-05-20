
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

plt.rcParams['font.sans-serif'] = [ 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-colorblind')

alpha = 0.75
scale = 0.75

label_fz = 32

slabel = '源域'
tlabel = '目標域'

def get_xy(tensor):
    x = tensor[:, 0]
    y = tensor[:, 1]
    return x, y

def plot_wham_last():

    s = 'WSJ'
    t = 'Wham'
    sl = f'{slabel}({s})'
    tl = f'{tlabel}({t})'
    fig_path = './plot/thesis_tsne/tsne_wham_last.pdf'

    data = np.load('./result/chapter4/da_cluster_test_wham_alllayer/cv_all_layer31_pca_tsne_per40_lr700.npz')

    bs = data['bs']
    bt = data['bt']
    cs = data['cs']
    ct = data['ct']

    plt.figure(figsize = [12.8, 4.8], dpi = 1000)

    plt.subplot(1, 2, 1)
    #plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    bsx, bsy = get_xy(bs)
    plt.scatter(bsx, bsy, s = scale, lw = 0, color = 'C0', alpha=alpha, label = sl)

    btx, bty = get_xy(bt)
    plt.scatter(btx, bty, s = scale, lw = 0, color = 'C2', alpha=alpha, label = tl)
    plt.xlabel('基礎模型', fontsize = label_fz)
    plt.legend(markerscale=15*scale, fontsize = 'x-large')

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    csx, csy = get_xy(cs)
    plt.scatter(csx, csy, s = scale, lw = 0, color = 'C0', alpha=alpha, label = sl)

    ctx, cty = get_xy(ct)
    plt.scatter(ctx, cty, s = scale, lw = 0, color = 'C2', alpha=alpha, label = tl)
    plt.xlabel('域對抗式方法', fontsize = label_fz)
    plt.legend(markerscale=15*scale, fontsize = 'x-large')
    plt.tight_layout()
    #plt.show()
    plt.savefig(fig_path, format = 'pdf')
    plt.savefig(fig_path.replace('pdf', 'jpeg'), format = 'jpeg')
    plt.close()

def plot_vctk_last():
    s = 'WSJ'
    t = 'VCTK'
    sl = f'{slabel}({s})'
    tl = f'{tlabel}({t})'
    fig_path = './plot/thesis_tsne/tsne_vctk_last.pdf'

    data = np.load('./result/chapter4/da_cluster_test/cv_all_layer31_pca_tsne_per40_lr700.npz')

    bs = data['bs']
    bt = data['bt']
    cs = data['cs']
    ct = data['ct']

    plt.figure(figsize = [12.8, 4.8], dpi = 1000)

    plt.subplot(1, 2, 1)
    #plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    bsx, bsy = get_xy(bs)
    plt.scatter(bsx, bsy, s = scale, lw = 0, color = 'C0', alpha=alpha, label = sl)

    btx, bty = get_xy(bt)
    plt.scatter(btx, bty, s = scale, lw = 0, color = 'C2', alpha=alpha, label = tl)
    plt.xlabel('基礎模型', fontsize = label_fz)
    plt.legend(markerscale=15*scale, fontsize = 'x-large')

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    csx, csy = get_xy(cs)
    plt.scatter(csx, csy, s = scale, lw = 0, color = 'C0', alpha=alpha, label = sl)

    ctx, cty = get_xy(ct)
    plt.scatter(ctx, cty, s = scale, lw = 0, color = 'C2', alpha=alpha, label = tl)
    plt.xlabel('域對抗式方法', fontsize = label_fz)
    plt.legend(markerscale=15*scale, fontsize = 'x-large')
    plt.tight_layout()
    #plt.show()
    plt.savefig(fig_path, format = 'pdf')
    plt.savefig(fig_path.replace('pdf', 'jpeg'), format = 'jpeg')
    plt.close()

def plot_vctk_mid():
    s = 'WSJ'
    t = 'VCTK'
    sl = f'{slabel}({s})'
    tl = f'{tlabel}({t})'
    fig_path = './plot/thesis_tsne/tsne_vctk_23.pdf'

    data = np.load('./result/chapter4/da_cluster_test/cv_all_layer23_pca_tsne_per40_lr700.npz')

    bs = data['bs']
    bt = data['bt']
    cs = data['cs']
    ct = data['ct']

    plt.figure(figsize = [12.8, 4.8], dpi = 1000)

    plt.subplot(1, 2, 1)
    #plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    bsx, bsy = get_xy(bs)
    plt.scatter(bsx, bsy, s = scale, lw = 0, color = 'C0', alpha=alpha, label = sl)

    btx, bty = get_xy(bt)
    plt.scatter(btx, bty, s = scale, lw = 0, color = 'C2', alpha=alpha, label = tl)
    plt.xlabel('基礎模型', fontsize = label_fz)
    plt.legend(markerscale=15*scale, fontsize = 'x-large')

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    csx, csy = get_xy(cs)
    plt.scatter(csx, csy, s = scale, lw = 0, color = 'C0', alpha=alpha, label = sl)

    ctx, cty = get_xy(ct)
    plt.scatter(ctx, cty, s = scale, lw = 0, color = 'C2', alpha=alpha, label = tl)
    plt.xlabel('域對抗式方法', fontsize = label_fz)
    plt.legend(markerscale=15*scale, fontsize = 'x-large')
    plt.tight_layout()
    #plt.show()
    plt.savefig(fig_path, format = 'pdf')
    plt.savefig(fig_path.replace('pdf', 'jpeg'), format = 'jpeg')
    plt.close()

def plot_vctk_mid_alllayer():
    s = 'WSJ'
    t = 'VCTK'
    sl = f'{slabel}({s})'
    tl = f'{tlabel}({t})'
    fig_path = './plot/thesis_tsne/tsne_vctk_23_alllayer.pdf'

    data = np.load('./result/chapter4/da_cluster_vctk_all/cv_MF_layer23_pca_tsne_per40_lr700.npz')

    bs = data['bs']
    bt = data['bt']
    cs = data['cs']
    ct = data['ct']

    plt.figure(figsize = [12.8, 4.8], dpi = 1000)

    plt.subplot(1, 2, 1)
    #plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    bsx, bsy = get_xy(bs)
    plt.scatter(bsx, bsy, s = scale, lw = 0, color = 'C0', alpha=alpha, label = sl)

    btx, bty = get_xy(bt)
    plt.scatter(btx, bty, s = scale, lw = 0, color = 'C2', alpha=alpha, label = tl)
    plt.xlabel('基礎模型', fontsize = label_fz)
    plt.legend(markerscale=15*scale, fontsize = 'x-large')

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    csx, csy = get_xy(cs)
    plt.scatter(csx, csy, s = scale, lw = 0, color = 'C0', alpha=alpha, label = sl)

    ctx, cty = get_xy(ct)
    plt.scatter(ctx, cty, s = scale, lw = 0, color = 'C2', alpha=alpha, label = tl)
    plt.xlabel('域對抗式方法', fontsize = label_fz)
    plt.legend(markerscale=15*scale, fontsize = 'x-large')
    plt.tight_layout()
    #plt.show()
    plt.savefig(fig_path, format = 'pdf')
    plt.savefig(fig_path.replace('pdf', 'jpeg'), format = 'jpeg')
    plt.close()


plot_wham_last()
plot_vctk_last()
plot_vctk_mid()
plot_vctk_mid_alllayer()
