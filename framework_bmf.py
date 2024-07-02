# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 16:12:10 2023

@author: yanbi
"""

import nimfa
import numpy as np
import pandas as pd

def framework_bmf(K,steps,X,n,indices_anomal,indices,N_min,N_max):

    nmf_model = nimfa.Bd(X, rank=K, max_iter=steps)
    W_bmf = nmf_model().fit.W
    H_bmf = nmf_model().fit.H
    X_est_bmf = nmf_model().fit.fitted()
    rss_bmf = nmf_model().fit.rss()
    
    # anomaly detection
    scores_bmf = []
    for i in range(n):
        score_bmf = np.sum(np.power(X_est_bmf[i]-X[i],2))
                
        scores_bmf.append(score_bmf)
        
    # flag top N as anomalies


    data_matrix = pd.DataFrame({'indices':indices,
                                'scores_bmf':scores_bmf})
 
    pan_bmf = []
    Ns = []

    for N in range(N_min,N_max):
        Ns.append(N)
        pan_bmf_current = 0
        # indices NMF
        topn_bmf_indx = data_matrix.scores_bmf.nlargest(n=N).index.values
        topn_bmf = data_matrix.indices[topn_bmf_indx].values
        for j in range(len(indices_anomal)):
            # P@N of NMF
            if indices_anomal[j] in topn_bmf[:]:
                pan_bmf_current = pan_bmf_current+1
            else:
                pan_bmf_current = pan_bmf_current
    
        pan_bmf.append(pan_bmf_current/N)

    return pan_bmf,Ns,scores_bmf