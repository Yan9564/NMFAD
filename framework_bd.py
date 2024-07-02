# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:52:54 2023

@author: yanbi
"""

import nimfa
import numpy as np
import pandas as pd

def framework_bd(K,steps,X,n,indices_anomal,indices,N_min,N_max):

    nmf_model = nimfa.Bd(X, rank=K, max_iter=steps)
    W_bd = nmf_model().fit.W
    H_bd = nmf_model().fit.H
    X_est_bd = nmf_model().fit.fitted()
    rss_bd = nmf_model().fit.rss()
    
    # anomaly detection
    scores_bd = []
    for i in range(n):
        score_bd = np.sum(np.power(X_est_bd[i]-X[i],2))
                
        scores_bd.append(score_bd)
        
    # flag top N as anomalies


    data_matrix = pd.DataFrame({'indices':indices,
                                'scores_bd':scores_bd})
 
    pan_bd = []
    Ns = []

    for N in range(N_min,N_max):
        Ns.append(N)
        pan_bd_current = 0
        # indices NMF
        topn_bd_indx = data_matrix.scores_bd.nlargest(n=N).index.values
        topn_bd = data_matrix.indices[topn_bd_indx].values
        for j in range(len(indices_anomal)):
            # P@N of NMF
            if indices_anomal[j] in topn_bd[:]:
                pan_bd_current = pan_bd_current+1
            else:
                pan_bd_current = pan_bd_current
    
        pan_bd.append(pan_bd_current/N)

    return pan_bd,Ns,scores_bd