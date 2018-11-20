import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import os

plot = 2

def smooth(line, N=10):
    smoothen = np.convolve(line, np.ones((N,))/N, mode='valid')
    return smoothen

with open('training_log.txt', 'r') as ifile:
    lines = ifile.readlines()

precisions, recalls, fscores = [], [], []
lossxs, lossys, lossws, losshs, loss_confs, loss_clss, loss_totals = [], [], [], [], [], [], []

for line in lines:
    if 'precision' in line:     # Test precision
        # 2018-07-27 12:02:20 precision: 0.484170, recall: 0.498014, fscore: 0.490990
        split = line.strip().split(' ')
        precision = float(split[3][:-1])
        recall = float(split[5][:-1])
        fscore = float(split[-1])
        # print(precision, recall, fscore)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
    elif 'proposals' in line:
        # 17962: nGT 527, recall 231, proposals 414, loss: x 4.572482, y 4.130553, w 6.640141, h 2.081057, conf 43.656105, cls 2.181237, total 63.261574
        split = line.strip().split(' ')
        lossx = float(split[9][:-1])
        lossy = float(split[11][:-1])
        lossw = float(split[13][:-1])
        lossh = float(split[15][:-1])
        loss_conf = float(split[17][:-1])
        loss_cls = float(split[19][:-1])
        loss_total = float(split[-1])
        # print(precision, recall, fscore)
        lossxs.append(lossx)
        lossys.append(lossy)
        lossws.append(lossw)
        losshs.append(lossh)
        loss_confs.append(loss_conf)
        loss_clss.append(loss_cls)
        loss_totals.append(loss_total)

plt.clf()

if plot == 1:

    plt.plot(precisions)
    plt.plot(recalls)
    plt.plot(fscores)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(['Precision', 'Recall', 'Fscore'])
    plt.title('Validation performance')
    plt.grid(True)
    plt.show()

elif plot == 2:

    plt.plot(smooth(loss_confs))
    plt.title('Training loss of confidence')
    plt.xlabel('Round')
    plt.ylabel('Loss')

elif plot == 3:

    plt.plot(smooth(loss_clss))
    plt.title('Training loss of class')
    plt.xlabel('Round')
    plt.ylabel('Loss')

elif plot == 4:

    plt.plot(smooth(loss_totals))
    plt.title('Training total loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')

else:

    assert 0, 'Invalid plot argument %d' % plot

plt.grid(True)
# plt.savefig(os.path.join('Figs', '%04d.png' % id_player))
plt.show()