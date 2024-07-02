# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:27:28 2023

@author: yanbi
"""

import nimfa
import numpy as np
import pandas as pd

def framework_snmf(K,steps,X,n,indices_anomal,indices,N_min,N_max):

    nmf_model = nimfa.Snmf(X, rank=K, max_iter=steps)
    W_snmf = nmf_model().fit.W
    H_snmf = nmf_model().fit.H
    X_est_snmf = nmf_model().fit.fitted()
    rss_snmf = nmf_model().fit.rss()
    
    # anomaly detection
    scores_snmf = []
    for i in range(n):
        score_snmf = np.sum(np.power(X_est_snmf[i]-X[i],2))
                
        scores_snmf.append(score_snmf)
        
    # flag top N as anomalies


    data_matrix = pd.DataFrame({'indices':indices,
                                'scores_snmf':scores_snmf})
 
    pan_snmf = []
    Ns = []

    for N in range(N_min,N_max):
        Ns.append(N)
        pan_snmf_current = 0
        # indices NMF
        topn_snmf_indx = data_matrix.scores_snmf.nlargest(n=N).index.values
        topn_snmf = data_matrix.indices[topn_snmf_indx].values
        for j in range(len(indices_anomal)):
            # P@N of NMF
            if indices_anomal[j] in topn_snmf[:]:
                pan_snmf_current = pan_snmf_current+1
            else:
                pan_snmf_current = pan_snmf_current
    
        pan_snmf.append(pan_snmf_current/N)

    return pan_snmf,Ns,scores_snmf