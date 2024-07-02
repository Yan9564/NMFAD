# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 16:15:10 2023

@author: yanbi
"""


import nimfa
import numpy as np
import pandas as pd

def framework_lfnmf(K,steps,X,n,indices_anomal,indices,N_min,N_max):

    nmf_model = nimfa.Bd(X, rank=K, max_iter=steps)
    W_lfnmf = nmf_model().fit.W
    H_lfnmf = nmf_model().fit.H
    X_est_lfnmf = nmf_model().fit.fitted()
    rss_lfnmf = nmf_model().fit.rss()
    
    # anomaly detection
    scores_lfnmf = []
    for i in range(n):
        score_lfnmf = np.sum(np.power(X_est_lfnmf[i]-X[i],2))
                
        scores_lfnmf.append(score_lfnmf)
        
    # flag top N as anomalies


    data_matrix = pd.DataFrame({'indices':indices,
                                'scores_lfnmf':scores_lfnmf})
 
    pan_lfnmf = []
    Ns = []

    for N in range(N_min,N_max):
        Ns.append(N)
        pan_lfnmf_current = 0
        # indices NMF
        topn_lfnmf_indx = data_matrix.scores_lfnmf.nlargest(n=N).index.values
        topn_lfnmf = data_matrix.indices[topn_lfnmf_indx].values
        for j in range(len(indices_anomal)):
            # P@N of NMF
            if indices_anomal[j] in topn_lfnmf[:]:
                pan_lfnmf_current = pan_lfnmf_current+1
            else:
                pan_lfnmf_current = pan_lfnmf_current
    
        pan_lfnmf.append(pan_lfnmf_current/N)

    return pan_lfnmf,Ns,scores_lfnmf