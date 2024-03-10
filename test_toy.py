
import sys
#sys.path.append("..")

from consensus_spc import *
import numpy as np
import matplotlib.pyplot as plt
from util import *
import seaborn as sns

import numpy as np
import sklearn.datasets

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import scale,MinMaxScaler
from sklearn.datasets import load_wine, load_breast_cancer
from scipy.cluster import hierarchy
import time


def main(x,labels, L=2, K=5, rep=100, rep_samp=0.8, file_path="./"):

    Ak = np.zeros(K - L + 1)
    cm = []
    cl_all = []

    nmi = []
    arc = []
    for k in range(L, K + 1):
        i = k - L
        ckm = CKmeans2(k=k, n_rep=rep, p_samp=rep_samp, p_feat=1.0)
        ckm.fit(x)
        ckm_res = ckm.predict(x)
        # Compute area under CDF
        m_upper = ckm_res.cmatrix[np.triu_indices(ckm_res.cmatrix.shape[0], k=1)]
        hist, bin_edges = np.histogram(ckm_res.cmatrix.ravel(), bins=np.linspace(0, 1, 100), density=True)
        cdf = np.cumsum(hist * np.diff(bin_edges))
        Ak[i] = np.sum(np.cumsum(hist * np.diff(bin_edges)) * np.diff(bin_edges))
        cm.append((bin_edges, cdf))
        cl_all.append(ckm_res.cl)
        nmi.append(normalized_mutual_info_score(labels, ckm_res.cl))
        arc.append(adjusted_rand_score(labels, ckm_res.cl))
        draw_consus(ckm_res.cmatrix,ckm_res.cl, k, ckm_res.order, file_path)

    plot_cdf(cm, L, file_path)
    deltaK = (Ak[1:] - Ak[:-1]) / Ak[:-1]
    deltaK = np.insert(deltaK, 0, Ak[0])
    bestK = np.argmax(deltaK) + L

    print(Ak)
    print(deltaK)
    print(bestK)

    k = [i for i in range(K - L + 1)]
    plt.figure(figsize=(10, 5))
    plt.plot(k, nmi, label='NMI', marker='o')
    plt.plot(k, arc, label='AR', marker='x')

    plt.title('HC clustering performance')
    plt.xlabel('k')
    plt.ylabel('score')
    plt.xticks(k, [f"{i + L}" for i in k])
    plt.legend()

    plt.savefig(f"{file_path}cp.png")


if __name__ == "__main__":

    start = time.time()
    data_sum = load_wine()
    # data_sum = load_breast_cancer()
    x = data_sum.data
    labels = data_sum.target
    x = scale(x, axis=0)

    '''
    eig_sv = gen_eigmap(n_components=3)
    x = eig_sv.gen_eig(x)
    '''

    main(x,labels)

    end = time.time()

    print(end-start)




