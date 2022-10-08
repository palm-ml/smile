import numpy as np
import copy
import metrics
from metrics import AdjustedHammingLoss, Coverage, RankingLoss, OneError, AveragePrecision, HammingLoss
import os
import torch

class train_logger:
    
    '''
    An instance of this class keeps track of various metrics throughout
    the training process.
    '''
    
    def __init__(self, params):
        
        self.params = params
        
        # epoch-level objects:
        self.best_stop_metric = -np.Inf
        self.best_epoch = -1
        self.running_loss = 0.0
        self.num_examples = 0
        
        # batch-level objects:
        self.temp_preds = []
        self.temp_true = [] # true labels
        self.temp_obs = [] # observed labels
        self.temp_indices = [] # indices for each example
        self.temp_batch_loss = []
        self.temp_batch_reg = []
        
        # output objects: 
        self.logs = {}
        self.logs['metrics'] = {}
        self.logs['best_preds'] = {}
        self.logs['gt'] ={}
        self.logs['obs'] = {}
        self.logs['targ'] = {}
        self.logs['idx'] = {}
        for field in self.logs:
            for phase in ['train', 'val', 'test']:
                self.logs[field][phase] = {}
    
    def compute_phase_metrics(self, phase, epoch, labels_est=None):
        
        '''
        Compute and store end-of-phase metrics. 
        '''
        
        # compute metrics w.r.t. clean ground truth labels:
        results = compute_metrics(self.temp_preds, self.temp_true)
        for k in results:
            self.logs['metrics'][phase].setdefault(k, [])
            self.logs['metrics'][phase][k].append(results[k])
        return results

    def update_phase_data(self, batch):
        
        '''
        Store data from a batch for later use in computing metrics. 
        '''
        
        for i in range(len(batch['idx'])):
            self.temp_preds.append(batch['preds_np'][i, :].tolist())
            self.temp_true.append(batch['label_vec_true'][i, :].tolist())
            self.temp_obs.append(batch['label_vec_obs'][i, :].tolist())
            self.temp_indices.append(int(batch['idx'][i]))
            self.num_examples += 1
        self.temp_batch_loss.append(float(batch['loss_np']))
        self.temp_batch_reg.append(float(batch['reg_loss_np']))
        self.running_loss += float(batch['loss_np'] * batch['image'].size(0))
        
    def reset_phase_data(self):
        
        '''
        Reset for a new phase. 
        '''
        
        self.temp_preds = []
        self.temp_true = []
        self.temp_obs = []
        self.temp_indices = []
        self.temp_batch_reg = []
        self.running_loss = 0.0
        self.num_examples = 0.0
    
    def get_best_metric(self, phase):
        best = {}
        for k in self.logs['metrics'][phase]:
            best[k] = min(self.logs['metrics'][phase][k])
        return best
        
        
        

def compute_metrics(y_pred, y_true):
    
    '''
    Given predictions and labels, compute a few metrics.
    '''
    num_examples, num_classes = np.shape(y_true)
    results = {}
    average_precision_list = []
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        average_precision_list.append(metrics.compute_avg_precision(y_true[:, j], y_pred[:, j]))
        
    results['map'] = - 100.0 * float(np.mean(average_precision_list))

    pred_scores = torch.FloatTensor(y_pred)
    pred_labels = torch.zeros_like(pred_scores)
    pred_labels[pred_scores > 0.5] = 1.0
    target_labels = torch.FloatTensor(y_true)
    results['AdjustedHammingloss'] = AdjustedHammingLoss(pred_labels, target_labels)
    results['Hammingloss'] = HammingLoss(pred_labels, target_labels)
    results['Coverage'] = Coverage(pred_scores, target_labels)
    results['AveragePrecision'] = - AveragePrecision(pred_scores, target_labels)
    results['RankingLoss'] = RankingLoss(pred_scores, target_labels)
    results['OneError'] = OneError(pred_scores, target_labels)
    
    return results
