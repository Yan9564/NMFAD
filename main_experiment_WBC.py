# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:07:47 2023

@author: yanbi
"""

# import the public packages

import nimfa
import numpy as np
from IPython import get_ipython
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import seaborn as sns
import random

# import the self defined functions

from mask_data import mask_data
from adjent_matrix import adjent_matrix
from GNMF import GNMF
from PGNMF import PGNMF
from framework_nmf import framework_nmf
from framework_svd import framework_svd
from framework_bd import framework_bd
from framework_bmf import framework_bmf
from framework_snmf import framework_snmf
from framework_gnmf import framework_gnmf
from framework_pgnmf import framework_pgnmf
from framework_lfnmf import framework_lfnmf

# %%

# set the figure plot

# if we want to pop up plot
get_ipython().run_line_magic('matplotlib', 'qt')
# if we want to inline plot
# get_ipython().run_line_magic('matplotlib', 'inline')

# %%

"""
(1) load data for experiment: Glass data set
"""

path = r'F:\10-NMF-ind1\experiment\Datasets\WBC'
df = pd.read_excel(path+'.xlsx')

# have a look of the summary of the data
print(df.head())

Y = df.to_numpy()
n = Y.shape[0]
# %%
# random choose n samples from Y
n_anomaly = 10
indices_normal = random.sample([i for i in range(n_anomaly, Y.shape[0])], k=n-n_anomaly)
indices_anomal = random.sample([i for i in range(n_anomaly)], k=n_anomaly)
indices = indices_normal+indices_anomal
indices.sort()
X = Y[indices,0:Y.shape[1]-1]

# %%

# decomposition using different methods

K = 5
steps = 1000
N_min = 10
N_max = 20
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
lamb = 100
pan_gnmf, Ns, scores_gnmf =  framework_gnmf(K, steps, X, n, indices_anomal, indices, N_min, N_max, H_nmf, W_nmf, length, lamb)

# %%

#  the proposed PGNMF 
length = 5
# if lamb = 0, there is no graph assisstance penalty
lamb1 = 100
lamb2 = 100
# determine edge_list_between_featur
df_X = df.copy()
df_X = df_X.drop(columns=['label'], axis=1)
dfXcorr = df_X.corr().to_numpy()
# find the two features with the largest correlation coeffecient
ind = np.unravel_index(np.argmax(dfXcorr-np.diag([1 for i in range(dfXcorr.shape[0])]), axis=None), dfXcorr.shape)
edge_list_between_featur = [ind,(2,5),(1,7),(1,3)]
# %%
#  the proposed PGNMF 
pan_pgnmf, Ns, scores_pgnmf = framework_pgnmf(K, steps, X, n, indices_anomal, indices, N_min, N_max, H_nmf, W_nmf, length, lamb1, lamb2, edge_list_between_featur)


# %%

# anomaly detection

save_flag = 0
save_path = r'F:\10-NMF-ind1\manuscript - itr\figures'
save_name = '\experiment_wbc_anomalies'

plt.rcParams["figure.figsize"] = (4,3)

plt.plot(scores_svd, 'x:', markersize=5, label='SVD')
plt.plot(scores_nmf, 'v-.', markersize=5, markerfacecolor='none', label='NMF')
plt.plot(scores_gnmf, 's--', markersize=5, markerfacecolor='none', label='GNMF')
plt.plot(scores_pgnmf, 'p-', markersize=5, markerfacecolor='none', label='PGNMF')

plt.legend()

plt.xlabel('number N of cut-off value')
plt.ylabel('anomaly score')
plt.tight_layout()

if save_flag == 1:
    plt.savefig(save_path+save_name+'.eps', dpi=300)
    print('==================================')
    print('Figure saved')
elif save_flag == 0:
    print('==================================')    
    print('Figure is not saved')



# %% 

# evaluate the accuracy
 
save_flag = 1
save_path = r'F:\10-NMF-ind1\manuscript - itr\figures'
save_name = '\experiment_wbc_PaN'

plt.rcParams["figure.figsize"] = (5,3)

plt.plot(Ns, pan_svd, 'x:', markersize=6, label='SVD')
plt.plot(Ns, pan_bd, 'x--', markersize=6, label='BD')
plt.plot(Ns, pan_bmf, 'x-.', markersize=6, label='BMF')
plt.plot(Ns, pan_nmf, '*-.', markersize=6, markerfacecolor='none', label='NMF')
plt.plot(Ns, pan_lfnmf, '*-.', markersize=6, markerfacecolor='none', label='LFNMF')
plt.plot(Ns, pan_snmf, 'v-.', markersize=6, markerfacecolor='none', label='SNMF')
plt.plot(Ns, pan_gnmf, 's--', markersize=6, markerfacecolor='none', label='GNMF')
plt.plot(Ns, pan_pgnmf, 'p-', markersize=6, markerfacecolor='none', label='PGNMF')

plt.legend(ncol=2)

plt.ylim([0.06,0.75])

plt.xlabel('number N of cut-off value')
plt.ylabel('P@N')
plt.tight_layout()

if save_flag == 1:
    plt.savefig(save_path+save_name+'.eps', dpi=300)
    print('==================================')
    print('Figure saved')
elif save_flag == 0:
    print('==================================')    
    print('Figure is not saved')

# %%
# save the results

experiment_wbc_results = pd.DataFrame({'Ns':Ns,
                                       'pan_svd':pan_svd,
                                       'pan_bd':pan_bd,
                                       'pan_bmf':pan_bmf,
                                       'pan_nmf':pan_nmf,
                                       'pan_lfnmf':pan_lfnmf,
                                       'pan_snmf':pan_snmf,
                                       'pan_gnmf':pan_gnmf,
                                       'pan_pgnmf':pan_pgnmf})

experiment_wbc_results.to_excel(r'F:\10-NMF-ind1\experiment\experiment_wbc_results.xlsx')