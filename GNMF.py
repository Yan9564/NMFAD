# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:52:29 2023

@author: yanbi
"""
import numpy as np
from adjent_matrix import adjent_matrix

def GNMF(length,vertex,X,K,nRepeat,lamb,W_nmf,H_nmf):
    
    n,P= X.shape
    
    edge_list = []
    for d in range(length):
        for i in range(n-d):
            edge_list.append((i,i+d))

    A = adjent_matrix(vertex,n,P,edge_list)


    DCol = np.sum(A,1)
    D = np.diag(DCol)


    W = W_nmf
    H = H_nmf


    tryNo = 0
    tryNos = []
    rss_gnmfs = []
    for tryNo in range(nRepeat):
        tryNos.append(tryNo)
        # print('========================================')
        # print(tryNo)
        # ===================== update H ========================
        WX = np.dot(W.T,X)
        WWH = np.dot(np.dot(W.T,W),H)
        WWH_min = 0.0000001*np.ones(WWH.shape)
        WWH = np.maximum(WWH,WWH_min)
        H = np.multiply(H,np.multiply(WX,1/WWH))
            
        # ===================== update W ========================
        XH = np.dot(X,H.T)
        lambAW = np.multiply(lamb,np.dot(A,W))
        WHH = np.dot(np.dot(W,H),H.T)
        lambDW = np.multiply(lamb,np.dot(D,W)) 
        WHHDW = WHH+lambDW
        XHAW = XH+lambAW
        WHHDW_min = 0.000001*np.ones(WHHDW.shape)
        WHHDW = np.maximum(WHHDW,WHHDW_min)
        W = np.multiply(W,np.multiply(XHAW,1/WHHDW))
            
        W_gnmf = W
        H_gnmf = H
        
        X_est_gnmf = np.dot(W_gnmf,H_gnmf)
        
        # print('X.shape : ',X.shape)
        # print('X_est_gnmf.shape : ',X_est_gnmf.shape)
        
        rss_gnmf = np.sum(np.multiply(X_est_gnmf-X,X_est_gnmf-X))
        rss_gnmfs.append(rss_gnmf)

    #print('RSS : ',rss_gnmf)
    
    return W_gnmf,H_gnmf,X_est_gnmf,rss_gnmf,rss_gnmfs,tryNos