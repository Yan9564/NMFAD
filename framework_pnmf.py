# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:19:43 2023

@author: yanbi
"""

import numpy as np
import pandas as pd
from PNMF import PNMF

def framework_pnmf(K,steps,X,n,indices_anomal,indices,N_min,N_max,H_nmf,W_nmf,length,lamb1,lamb2,edge_list_between_featur):
    nRepeat = steps
    W_pnmf, H_pnmf, X_est_pnmf, rss_pnmf, rss_pnmfs, tryNos = PNMF(length, X, K, nRepeat, lamb1, lamb2, W_nmf, H_nmf, edge_list_between_featur)

    
    # anomaly detection
    scores_pnmf = []
    for i in range(n):
        score_pnmf = np.sum(np.power(X_est_pnmf[i]-X[i],2))
                
        scores_pnmf.append(score_pnmf)
        
    # flag top N as anomalies


    data_matrix = pd.DataFrame({'indices':indices,
                                'scores_pnmf':scores_pnmf})
 
    pan_pnmf = []
    Ns = []

    for N in range(N_min,N_max):
        Ns.append(N)
        pan_pnmf_current = 0
        # indices NMF
        topn_pnmf_indx = data_matrix.scores_pnmf.nlargest(n=N).index.values
        topn_pnmf = data_matrix.indices[topn_pnmf_indx].values
        for j in range(len(indices_anomal)):
            # P@N of NMF
            if indices_anomal[j] in topn_pnmf[:]:
                pan_pnmf_current = pan_pnmf_current+1
            else:
                pan_pnmf_current = pan_pnmf_current
    
        pan_pnmf.append(pan_pnmf_current/N)

    return pan_pnmf,Ns,scores_pnmf