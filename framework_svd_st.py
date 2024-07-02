# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:34:16 2024

@author: yanbi
"""


import numpy as np
import pandas as pd

# Define the soft-thresholding function
def soft_thresholding(s, lambda_val):
    return np.sign(s) * np.maximum(np.abs(s) - lambda_val, 0) 

def framework_svd_st(K,steps,X,n,indices_anomal,indices,N_min,N_max):

    U, S, V = np.linalg.svd(np.matrix(X,dtype='float'), full_matrices=False)
    
    # Apply soft-thresholding to the singular values with a threshold of lambda_val
    lambda_val = 8 # Example threshold value, can be adjusted based on needs
    s_thresholded = soft_thresholding(S, lambda_val)

    # Reconstruct the matrix with the thresholded singular values
    S = s_thresholded
    
    print(s_thresholded)
  
    W_svd = np.dot(U[:,0:K],np.diag(S[0:K]))
    H_svd = V[0:K,:]
    X_est_svd = np.dot(W_svd,H_svd)
    rss_svd = np.sum(np.power(X-X_est_svd,2))
    
    # anomaly detection
    scores_svd = []
    for i in range(n):
        score_svd = np.sum(np.power(X_est_svd[i]-X[i],2))
                
        scores_svd.append(score_svd)
        
    # flag top N as anomalies


    data_matrix = pd.DataFrame({'indices':indices,
                                'scores_svd':scores_svd})
 
    pan_svd = []
    Ns = []

    for N in range(N_min,N_max):
        Ns.append(N)
        pan_svd_current = 0
        # indices NMF
        topn_svd_indx = data_matrix.scores_svd.nlargest(n=N).index.values
        topn_svd = data_matrix.indices[topn_svd_indx].values
        for j in range(len(indices_anomal)):
            # P@N of NMF
            if indices_anomal[j] in topn_svd[:]:
                pan_svd_current = pan_svd_current+1
            else:
                pan_svd_current = pan_svd_current
    
        pan_svd.append(pan_svd_current/N)

    return pan_svd,Ns,scores_svd