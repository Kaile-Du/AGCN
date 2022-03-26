"""
functions used to calculate the metrics for multi-label classification
cmap=mAP, emap=MiAP
"""
import numpy as np
import pdb

def cemap_cal_old(y_pred,y_true):
    '''

    y_true: -1 negative; 0 difficult_examples; 1 positive.
    '''
    nTest = y_true.shape[0]
    nLabel = y_true.shape[1]
    ap = np.zeros(nTest)
    for i in range(0,nTest):
        R = np.sum(y_true[i,:]==1)
        for j in range(0,nLabel):            
            if y_true[i,j]==1:
                r = np.sum(y_pred[i,np.nonzero(y_true[i,:]!=0)]>=y_pred[i,j])
                rb = np.sum(y_pred[i,np.nonzero(y_true[i,:]==1)] >= y_pred[i,j])

                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    emap = np.nanmean(ap)


    ap = np.zeros(nLabel)
    for i in range(0,nLabel):
        R = np.sum(y_true[:,i]==1)
        for j in range(0,nTest):
            if y_true[j,i]==1:
                r = np.sum(y_pred[np.nonzero(y_true[:,i]!=0),i] >= y_pred[j,i])
                rb = np.sum(y_pred[np.nonzero(y_true[:,i]==1),i] >= y_pred[j,i])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    cmap = np.nanmean(ap)

    return cmap,emap
def prf_cal(scores_,targets_):
    """
    function to calculate top-k precision/recall/f1-score
    y_true: 0 1
    """
    n = targets_.shape[0]
    num_classes = targets_.shape[1]
    Nc, Np, Ng = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)
    for k in range(num_classes):

        scores = scores_[ : , k]
        targets = targets_[ : , k]
        targets[targets == -1] = 0
        Ng[k] = np.sum(targets == 1)
        Np[k] = np.sum(scores >= 0)
        Nc[k] = np.sum(targets * (scores >= 0))
    Np[Np == 0] = 1
    OP = np.sum(Nc) / np.sum(Np)
    OR = np.sum(Nc) / np.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)
    for i in range(len(Np)):
        if Np[i] == 0:
            Np[i] = 0.0001
    CP = np.sum(Nc / Np) / num_classes

    CR = np.sum(Nc / Ng) / num_classes
    CF1 = (2 * CP * CR) / (CP + CR)
    return OP, OR, OF1, CP, CR, CF1




def cemap_cal(y_pred,y_true):
    """
    function to calculate C-MAP (mAP) and E-MAP
    y_true: 0 1
    """
    nTest = y_true.shape[0]
    nLabel = y_true.shape[1]
    ap = np.zeros(nTest)
    for i in range(0,nTest):
        R = np.sum(y_true[i,:])
        for j in range(0,nLabel):            
            if y_true[i,j]==1:
                r = np.sum(y_pred[i,:]>=y_pred[i,j])
                rb = np.sum(y_pred[i,np.nonzero(y_true[i,:])] >= y_pred[i,j])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    emap = np.nanmean(ap)

    ap = np.zeros(nLabel)
    for i in range(0,nLabel):
        R = np.sum(y_true[:,i])
        for j in range(0,nTest):
            if y_true[j,i]==1:
                r = np.sum(y_pred[:,i] >= y_pred[j,i])
                rb = np.sum(y_pred[np.nonzero(y_true[:,i]),i] >= y_pred[j,i])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    cmap = np.nanmean(ap)

    return cmap,emap
