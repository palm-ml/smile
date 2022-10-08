import os
import json
import numpy as np
from PIL import Image
import random
from numpy.lib.utils import source
import torch
import copy
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torchvision import transforms
from scipy.io import savemat
from scipy.io.matlab.mio import loadmat


def get_data(P):
    '''
    Given a parameter dictionary P, initialize and return the specified dataset. 
    '''
    
    # select and return the right dataset:
    
    ds = multilabel(P).get_datasets()
    
    
    # Optionally overwrite the observed training labels with clean labels:
    assert P['train_set_variant'] in ['clean', 'observed']
    if P['train_set_variant'] == 'clean':
        print('Using clean labels for training.')
        ds['train'].label_matrix_obs = copy.deepcopy(ds['train'].label_matrix)
    else:
        print('Using single positive labels for training.')
    
    # Optionally overwrite the observed val labels with clean labels:
    assert P['val_set_variant'] in ['clean', 'observed']
    if P['val_set_variant'] == 'clean':
        print('Using clean labels for validation.')
        ds['val'].label_matrix_obs = copy.deepcopy(ds['val'].label_matrix)
    else:
        print('Using single positive labels for validation.')
    
    # We always use a clean test set:
    ds['test'].label_matrix_obs = copy.deepcopy(ds['test'].label_matrix)
            
    return ds

def load_spl_from_mat(ds, source_fold="SPL_Datasets_mat/"):
    print("loading Dataset {}".format(ds))
    path = source_fold + ds + ".mat"
    data = loadmat(path)
    return data['X'], data['Y'], data['Y_obs'], data['tr_idx'], data['va_idx'], data['te_idx']

def normalize(x):
    return (x - x.mean(axis=0, keepdims=True) )/ x.std(axis=0, keepdims=True)

def load_data(P):
    X, Y, Y_obs, tr_idx, va_idx, te_idx = load_spl_from_mat(P['dataset'])
    # X = normalize(X)
    data = {}
    for phase in ['train', 'val', 'test']:
        if phase == 'train':
            idx = tr_idx[0]
        if phase == 'val':
            idx = va_idx[0]
        if phase == 'test':
            idx = te_idx[0]
        data[phase] = {}
        data[phase]['image_idx'] = idx
        data[phase]['labels'] = Y[idx, :]
        data[phase]['labels_obs'] = Y_obs[idx, :]
        data[phase]['images'] = X[idx, :]
        data[phase]['feats'] = X[idx, :]
    return data

class multilabel:

    def __init__(self, P):
        
        # load data:
        source_data = load_data(P)
        
        # define train set:
        self.train = ds_multilabel(
            source_data['train']['labels'],
            source_data['train']['labels_obs'],
            source_data['train']['feats'],
            source_data['train']['image_idx']
        )
            
        # define val set:
        self.val = ds_multilabel(
            source_data['val']['labels'],
            source_data['val']['labels_obs'],
            source_data['val']['feats'],
            source_data['val']['image_idx']
        )
        
        # define test set:
        self.test = ds_multilabel(
            source_data['test']['labels'],
            source_data['test']['labels_obs'],
            source_data['test']['feats'],
            source_data['test']['image_idx']
        )
        
        # define dict of dataset lengths: 
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}

class ds_multilabel(Dataset):

    def __init__(self, label_matrix, label_matrix_obs, feats, image_idx):
    
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs
        self.feats = feats
        self.image_idx = image_idx

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        
        out = {
            'image': torch.FloatTensor(np.copy(self.feats[idx, :])),
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'idx': idx,
        }
        
        return out
