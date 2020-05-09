
import comet_ml
from comet_ml import API
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = [ 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-colorblind')

def sort(metrics, s = 'step'):
    """
    s : 'step' or 'epoch'
    """

    steps_value = [ (m[s], float(m['metricValue'])) for m in metrics ]
    steps_value.sort(key = lambda x: x[0])
    return steps_value

def get_exp(exp_id):
    comet_api = API()
    exp = comet_api.get('riviera1020', 'semi-separation', exp_id)
    return exp

sn = '106d5401a8a34f0bb646889fb6097154'
ln = '93be94113dbf4fbcadfe4045396998e7'
wn = '815efc81886948b89bd8775b35958693'

sn_exp = get_exp(sn)
ln_exp = get_exp(ln)
wn_exp = get_exp(wn)

mname = 'train_pretrain_dis_domain_acc'

sn_acc = sn_exp.get_metrics(mname)
ln_acc = ln_exp.get_metrics(mname)
wn_acc = wn_exp.get_metrics(mname)

sn_acc = sort(sn_acc)
ln_acc = sort(ln_acc)
wn_acc = sort(wn_acc)

steps, sn_acc = zip(*sn_acc)
_, ln_acc = zip(*ln_acc)
_, wn_acc = zip(*wn_acc)

lw = 1.5
alpha = 0.8
plt.figure()
plt.title('不同正規化對鑑別器的影響', fontsize = 'x-large')
plt.xlabel('更新次數', fontsize = 'large')
plt.ylabel('域辨識準確率', fontsize = 'large')
plt.ylim([0, 1.1])
plt.plot(steps, sn_acc, alpha = alpha, linewidth = lw, label = '譜正規化')
plt.plot(steps, ln_acc, alpha = alpha, linewidth = lw, label = '全域層正規化')
plt.plot(steps, wn_acc, alpha = alpha, linewidth = lw, label = '權重正規化')
plt.legend(loc = 4, fontsize = 'large')
#plt.show()
plt.savefig('./plot/c4/norm_domain_acc.pdf', format = 'pdf')
