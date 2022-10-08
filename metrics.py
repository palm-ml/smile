import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, hamming_loss, zero_one_loss, coverage_error, label_ranking_loss
from scipy.io import loadmat
from copy import deepcopy

def HammingLoss(pred_labels, target_labels):
    '''
    Computing Hamming loss

    Parameters
    ----------
    pred_labels : Tensor
        MxQ Tensor storing the predicted labels of the classifier, if the ith 
        instance belongs to the jth class, then pred_labels[i,j] equals to +1, 
        otherwise pred_labels[i,j] equals to 0.
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    hammingloss : float
    ''' 
    return torch.mean((pred_labels != target_labels).float()).item()

def AdjustedHammingLoss(pred_labels, target_labels):
    '''
    Computing Adjusted Hamming loss

    Parameters
    ----------
    pred_labels : Tensor
        MxQ Tensor storing the predicted labels of the classifier, if the ith 
        instance belongs to the jth class, then pred_labels[i,j] equals to +1, 
        otherwise pred_labels[i,j] equals to 0.
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    adjustedhammingloss : float
    '''
    sum_pos_pred = (pred_labels == 1).sum(dim=1)
    sum_pos_target = (target_labels == 1).sum(dim=1)
    sum_pos = sum_pos_pred + sum_pos_target
    wrong_pred = (pred_labels != target_labels).float().sum(dim=1)
    adjustedhammingloss = torch.mean(wrong_pred/(sum_pos + 1e-6)).item()
    return adjustedhammingloss
    
def OneError(pred_scores, target_labels):
    '''
    Computing one error

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    oneerror : float
    '''
    _, index = torch.max(pred_scores, dim=1)
    
    oneerror = 0.0
    num_data = pred_scores.size(0)
    for i in range(num_data):
        if target_labels[i, index[i]] != 1:
            oneerror += 1
            
    return oneerror / num_data

def Coverage(pred_scores, target_labels):
    '''
    Computing coverage

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    coverage : float
    '''
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    
    coverage = 0.0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        if has_label[i,:].sum() > 0:
            r = torch.max(order[i, has_label[i,:]]).item() + 1
            coverage += r
    coverage = coverage / num_data - 1.0
    return coverage / num_classes
    
def RankingLoss(pred_scores, target_labels):
    '''
    Computing ranking loss

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    rankingloss : float
    '''
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    
    rankingloss = 0.0
    count = 0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        m = torch.sum(has_label[i,:]).item()
        n = num_classes - m
        if m != 0 and n != 0:
            rankingloss = rankingloss + (torch.sum(order[i, has_label[i, :]]).item()
                                         - m*(m-1)/2.0) / (m*n)
            count += 1
            
    return rankingloss / count

def AveragePrecision(pred_scores, target_labels):
    '''
    Computing average precision

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    ap : float
    '''
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    
    ap = 0.0
    count = 0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        m = torch.sum(has_label[i,:]).item()
        if m != 0:
            sorts, _ = torch.sort(order[i, has_label[i, :]])
            temp = 0.0
            for j in range(sorts.size(0)):
                temp += (j+1.0) / (sorts[j].item() + 1)
            ap += temp / m
            count += 1
    
    return ap / count

def MacroAUC(pred_scores, target_labels):
    '''
    Computing macro-averaging AUC

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    macroauc : float
    '''
    _, index = torch.sort(pred_scores, 0)
    _, order = torch.sort(index, 0)
    has_label = target_labels == 1
    
    macroauc = 0.0
    count = 0
    num_data, num_classes = pred_scores.size()
    for i in range(num_classes):
        m = torch.sum(has_label[:,i]).item()
        n = num_data - m
        if m != 0 and n != 0:
            macroauc = macroauc + (torch.sum(order[has_label[:,i], i]).item()
                                         - m*(m-1)/2.0) / (m*n)
            count += 1
            
    return macroauc / count

def Multi_Hot(y_scores):
    y_preds = torch.zeros_like(y_scores)
    y_preds[y_scores > 0.5] = 1.0
    return y_preds

def check_inputs(targs, preds):
    
    '''
    Helper function for input validation.
    '''
    
    assert (np.shape(preds) == np.shape(targs))
    assert type(preds) is np.ndarray
    assert type(targs) is np.ndarray
    assert (np.max(preds) <= 1.0) and (np.min(preds) >= 0.0)
    assert (np.max(targs) <= 1.0) and (np.min(targs) >= 0.0)
    assert (len(np.unique(targs)) <= 2)

def compute_avg_precision(targs, preds):
    
    '''
    Compute average precision.
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    '''
    
    check_inputs(targs,preds)
    
    if np.all(targs == 0):
        # If a class has zero true positives, we define average precision to be zero.
        metric_value = 0.0
    else:
        metric_value = average_precision_score(targs, preds)
    
    return metric_value

def compute_precision_at_k(targs, preds, k):
    
    '''
    Compute precision@k. 
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    k: Number of predictions to consider.
    '''
    
    check_inputs(targs, preds)
    
    classes_rel = np.flatnonzero(targs == 1)
    if len(classes_rel) == 0:
        return 0.0
    
    top_k_pred = np.argsort(preds)[::-1][:k]
    
    metric_value = float(len(np.intersect1d(top_k_pred, classes_rel))) / k
    
    return metric_value

def compute_recall_at_k(targs, preds, k):
    
    '''
    Compute recall@k. 
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    k: Number of predictions to consider.
    '''
    
    check_inputs(targs,preds)
    
    classes_rel = np.flatnonzero(targs == 1)
    if len(classes_rel) == 0:
        return 0.0
    
    top_k_pred = np.argsort(preds)[::-1][:k]
    
    metric_value = float(len(np.intersect1d(top_k_pred, classes_rel))) / len(classes_rel)
    
    return metric_value
