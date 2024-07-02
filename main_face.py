# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:46:34 2024

@author: yanbi
"""

import nimfa
from PIL import Image

import numpy as np
import pandas as pd

# import the self defined functions

from framework_nmf import framework_nmf
from framework_svd import framework_svd
from framework_bd import framework_bd
from framework_bmf import framework_bmf
from framework_snmf import framework_snmf
from framework_gnmf import framework_gnmf
from framework_pnmf import framework_pnmf
from framework_lfnmf import framework_lfnmf

#%%
# TEST for the dataset
# nimfa.examples.cbcl_images.run()

#%%

# prepare the dataset

V = nimfa.examples.cbcl_images.read()

V = nimfa.examples.cbcl_images.preprocess(V)

#%%

# check the figure

V_origi = V[:,10].reshape(19, 19)

image = Image.fromarray(V_origi.astype('uint8'), 'L')  # 'L' is for grayscale
# image.show()

#%%

# set parameters for the anomaly detection
n_anomaly = 5
indices_anomal = [3,5,10,110,129]
indices = [i for i in range(2429)]
for indice in indices_anomal:
    V[:,indice] = V[:,indice]+np.abs(np.random.normal(loc=0.05, scale=0.1, size=361))
X = V.T
n = 2429
#%%

# decomposition using different methods

K = 5
steps = 100
N_min = 5
N_max = 9
# the NMF method
pan_nmf, Ns, scores_nmf, H_nmf, W_nmf = framework_nmf(K, steps, X, n, indices_anomal, indices, N_min, N_max)
# the SVD method
pan_svd, Ns, scores_svd = framework_svd(K, steps, X, n, indices_anomal, indices, N_min, N_max)
# the SNMF method
pan_snmf,Ns,scores_snmf = framework_snmf(K,steps,X,n,indices_anomal,indices,N_min,N_max)

# %%
# the BD method
pan_bd, Ns, scores_bd = framework_bd(K, steps, X, n, indices_anomal, indices, N_min, N_max)
# %%
# the BMF method
pan_bmf, Ns, scores_bmf = framework_bmf(K, steps, X, n, indices_anomal, indices, N_min, N_max)
# %%
# the LFNMF method
pan_lfnmf, Ns, scores_lfnmf = framework_lfnmf(K, steps, X, n, indices_anomal, indices, N_min, N_max)
# %%
# the GNMF method
length = 5
lamb = 10
pan_gnmf, Ns, scores_gnmf =  framework_gnmf(K, steps, X, n, indices_anomal, indices, N_min, N_max, H_nmf, W_nmf, length, lamb)

# %%

#  the proposed pnmf 
length = 1
# if lamb = 0, there is no graph assisstance penalty
lamb1 = 10
lamb2 = 10
# determine edge_list_between_featur
dfXcorr = pd.DataFrame(X).corr().to_numpy()
# find the two features with the largest correlation coeffecient
ind = np.unravel_index(np.argmax(dfXcorr-np.diag([1 for i in range(dfXcorr.shape[0])]), axis=None), dfXcorr.shape)
B = []
for i in range(19):
    for j in range(1,9):
        b = (i*19+j,i*19+19-j)
        B.append(b)
# B = [(77,267), (106,257), ind]
edge_list_between_featur = []
for p in range(len(B)):
    edge_list_between_featur.append((B[p][0]-1, B[p][1]-1))
# %%
#  the proposed pnmf 
pan_pnmf, Ns, scores_pnmf = framework_pnmf(K, steps, X, n, indices_anomal, indices, N_min, N_max, H_nmf, W_nmf, length, lamb1, lamb2, edge_list_between_featur)
