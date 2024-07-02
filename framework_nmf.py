# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 23:55:45 2023

@author: yanbi
"""

import nimfa
import numpy as np
import pandas as pd

def framework_nmf(K,steps,X,n,indices_anomal,indices,N_min,N_max):

    nmf_model = nimfa.Nmf(X, 
                    seed="random_c", 
                    rank=K, 
                    max_iter=steps)
    W_nmf = nmf_model().fit.W
    H_nmf = nmf_model().fit.H
    X_est_nmf = nmf_model().fit.fitted()
    rss_nmf = nmf_model().fit.rss()
    
    # anomaly detection
    scores_nmf = []
    for i in range(n):
        score_nmf = np.sum(np.power(X_est_nmf[i]-X[i],2))
                
        scores_nmf.append(score_nmf)
        
    # flag top N as anomalies


    data_matrix = pd.DataFrame({'indices':indices,
                                'scores_nmf':scores_nmf})
 
    pan_nmf = []
    Ns = []

    for N in range(N_min,N_max):
        Ns.append(N)
        pan_nmf_current = 0
        # indices NMF
        topn_nmf_indx = data_matrix.scores_nmf.nlargest(n=N).index.values
        topn_nmf = data_matrix.indices[topn_nmf_indx].values
        for j in range(len(indices_anomal)):
            # P@N of NMF
            if indices_anomal[j] in topn_nmf[:]:
                pan_nmf_current = pan_nmf_current+1
            else:
                pan_nmf_current = pan_nmf_current
    
        pan_nmf.append(pan_nmf_current/N)

    return pan_nmf,Ns,scores_nmf, H_nmf, W_nmf
        