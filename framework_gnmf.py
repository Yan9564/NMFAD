# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:12:44 2023

@author: yanbi
"""


import numpy as np
import pandas as pd
from GNMF import GNMF

def framework_gnmf(K,steps,X,n,indices_anomal,indices,N_min,N_max,H_nmf,W_nmf,length,lamb):
    nRepeat = steps
    vertex = 'row'
    W_gnmf, H_gnmf, X_est_gnmf, rss_gnmf, null, null = GNMF(length, vertex, X, K, nRepeat, lamb, W_nmf, H_nmf)
    
    # anomaly detection
    scores_gnmf = []
    for i in range(n):
        score_gnmf = np.sum(np.power(X_est_gnmf[i]-X[i],2))
                
        scores_gnmf.append(score_gnmf)
        
    # flag top N as anomalies


    data_matrix = pd.DataFrame({'indices':indices,
                                'scores_gnmf':scores_gnmf})
 
    pan_gnmf = []
    Ns = []

    for N in range(N_min,N_max):
        Ns.append(N)
        pan_gnmf_current = 0
        # indices NMF
        topn_gnmf_indx = data_matrix.scores_gnmf.nlargest(n=N).index.values
        topn_gnmf = data_matrix.indices[topn_gnmf_indx].values
        for j in range(len(indices_anomal)):
            # P@N of NMF
            if indices_anomal[j] in topn_gnmf[:]:
                pan_gnmf_current = pan_gnmf_current+1
            else:
                pan_gnmf_current = pan_gnmf_current
    
        pan_gnmf.append(pan_gnmf_current/N)

    return pan_gnmf,Ns,scores_gnmf
        