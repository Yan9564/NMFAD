# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:03:18 2023

@author: yanbi
"""


import numpy as np
from adjent_matrix import adjent_matrix

def PNMF(length,X,K,nRepeat,lamb1,lamb2,W_nmf,H_nmf,edge_list_between_featur):
    
    n,P= X.shape
    
    edge_list_between_measus = []
    for d in range(length):
        for i in range(n-d):
            edge_list_between_measus.append((i,i+d))

    vertex = 'row'
    A = adjent_matrix(vertex,n,P,edge_list_between_measus)
    vertex = 'column'
    B = adjent_matrix(vertex,n,P,edge_list_between_featur)


    DCol = np.sum(A,1)
    UCol = np.sum(B,1)
    D = np.diag(DCol)
    U = np.diag(UCol)

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
        lamb2HB = np.multiply(lamb2,np.dot(H,B))
        WXHB = WX+lamb2HB
        WWH = np.dot(np.dot(W.T,W),H)
        lamb2HU = np.multiply(lamb2,np.dot(H,U))
        WWHHU = WWH+lamb2HU
        WWHHU_min = 0.0000001*np.ones(WWHHU.shape)
        WWHHU = np.maximum(WWHHU,WWHHU_min)
        H = np.multiply(H,np.multiply(WXHB,1/WWHHU))
            
        # ===================== update W ========================
        XH = np.dot(X,H.T)
        lamb1AW = np.multiply(lamb1,np.dot(A,W))
        WHH = np.dot(np.dot(W,H),H.T)
        lamb1DW = np.multiply(lamb1,np.dot(D,W)) 
        WHHDW = WHH+lamb1DW
        XHAW = XH+lamb1AW
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