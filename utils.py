import imp
import numpy as np
import copy
import metrics
from metrics import Coverage, RankingLoss, OneError, AveragePrecision, HammingLoss
import os
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io.matlab.mio import loadmat


def prepare_data(args):
    print("loading Dataset {}".format(args.ds))
    path = os.path.join(args.source_fold, args.ds + ".mat")
    data = loadmat(path)
    X, Y, Y_obs, tr_idx, va_idx, te_idx = data['X'], data['Y'], data['Y_obs'], data['tr_idx'][0], data['va_idx'][0], data['te_idx'][0]
    train_X, train_Y, train_s_Y = X[tr_idx], Y[tr_idx], Y_obs[tr_idx]
    valid_X, valid_Y, valid_s_Y = X[va_idx], Y[va_idx], Y_obs[va_idx]
    test_X, test_Y, test_s_Y = X[te_idx], Y[te_idx], Y_obs[te_idx]
    return train_X, train_Y, train_s_Y, valid_X, valid_Y, valid_s_Y, test_X, test_Y, test_s_Y


def prepare_dataloader(args, train_X, train_Y, train_s_Y, valid_X, valid_Y, valid_s_Y, test_X, test_Y, test_s_Y):
    class ds_multilabel(Dataset):
        def __init__(self, X, Y, Y_obs):
            self.feats = X
            self.label_matrix= Y
            self.label_matrix_obs = Y_obs

        def __len__(self):
            return len(self.feats)

        def __getitem__(self, idx):
            feats = torch.FloatTensor(np.copy(self.feats[idx, :]))
            label_vec_obs = torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :]))
            label_vec_true = torch.FloatTensor(np.copy(self.label_matrix[idx, :]))
    
            return feats, label_vec_obs, label_vec_true, idx

    train_dataset, valid_dataset, test_dataset = ds_multilabel(train_X, train_Y, train_s_Y), ds_multilabel(valid_X, valid_Y, valid_s_Y), ds_multilabel(test_X, test_Y, test_s_Y)
    train_loader, valid_loader, test_loader =  DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8), \
        DataLoader(valid_dataset, batch_size=args.bs, shuffle=True, num_workers=8), \
        DataLoader(test_dataset, batch_size=args.bs, shuffle=True, num_workers=8)
    return train_loader, valid_loader, test_loader

def evaluate(model, X, Y, flag=""):
    results = {}
    if flag != "":
        flag += "_"
    with torch.no_grad():
        X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)
        X, Y, model = map(lambda x: x.cuda(), (X, Y, model))
        model.eval()
        Y_score = torch.sigmoid(model(X))
        Y_pred  = torch.zeros_like(Y_score) 
        Y_pred[Y_score > 0.5]  = 1.0
        results[flag + 'Hammingloss'] = HammingLoss(Y_pred, Y)
        results[flag + 'Coverage'] = Coverage(Y_score, Y)
        results[flag + 'AveragePrecision'] = AveragePrecision(Y_score,Y)
        results[flag + 'RankingLoss'] = RankingLoss(Y_score, Y)
        results[flag + 'OneError'] = OneError(Y_score, Y)
        # average_precision_list = []
        # y_pred = Y_score.cpu().numpy()
        # y_true = Y.cpu().numpy()
        # y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
        # for j in range(y_true.shape[-1]):
        #     average_precision_list.append(metrics.compute_avg_precision(y_true[:, j], y_pred[:, j]))
    return results#, 100.0 * float(np.mean(average_precision_list))

from copy import deepcopy
from math import inf
class Best: 
    def __init__(self, measures, directions):
        self.best_dict = {}
        self.best_measure = ""
        self.measures = measures
        self.directions = directions
        for measure, direction in zip(self.measures, self.directions):
            if direction == "max":
                default_dict = {
                    "best_va_score": -inf,
                    "best_va_results": {},
                    "best_te_results": {},
                    "best_epoch": -1,
                    "best_weight": None
                }
            else:
                default_dict = {
                    "best_va_score": inf,
                    "best_va_results": {},
                    "best_te_results": {},
                    "best_epoch": -1,
                    "best_weight": None
                }
            self.best_dict.setdefault(measure, default_dict)
    
    def update(self, epoch, va_results, te_results, model):
        for measure, direction in zip(self.measures, self.directions):
            if direction == "max":
                if va_results[measure] > self.best_dict[measure]["best_va_score"]:
                    self.best_dict[measure]["best_va_score"] = va_results[measure]
                    self.best_dict[measure]["best_va_results"] = deepcopy(va_results)
                    self.best_dict[measure]["best_te_results"] = deepcopy(te_results)
                    self.best_dict[measure]["best_epoch"] = epoch
                    self.best_dict[measure]["best_weight"] = deepcopy(model.state_dict())
            else:
                if va_results[measure] < self.best_dict[measure]["best_va_score"]:
                    self.best_dict[measure]["best_va_score"] = va_results[measure]
                    self.best_dict[measure]["best_va_results"] = deepcopy(va_results)
                    self.best_dict[measure]["best_te_results"] = deepcopy(te_results)
                    self.best_dict[measure]["best_epoch"] = epoch
                    self.best_dict[measure]["best_weight"] = deepcopy(model.state_dict())
    
    def show(self, flag="va"):
        # print(self.best_dict)
        if flag == "va":
            result_key = "best_va_results"
        if flag == "te":
            result_key = "best_te_results"
        print(result_key)
        for measure, direction in zip(self.measures, self.directions):
            print("| {:<20} | {:<5} |".format(measure, direction), end=" ")
            for measure2 in self.measures:
                print("{:.3f} |".format(self.best_dict[measure][result_key][measure2]), end=" ")
            print()
    
    def select(self, index=None, default="AveragePrecision"):
        if index==None:
            # find the most 'epoch'
            epoch_count = {}
            for measure in self.measures:
                epoch_count.setdefault(self.best_dict[measure]["best_epoch"], {
                    "num": 0,
                    "measures": []
                })
                epoch_count[self.best_dict[measure]["best_epoch"]]["num"] += 1
                epoch_count[self.best_dict[measure]["best_epoch"]]["measures"].append(measure)
            max_count, max_epoch, max_measures = 0, -1, []
            for epoch, epoch_dict in epoch_count.items():
                if epoch_count[epoch]["num"] > max_count:
                    max_count = epoch_count[epoch]["num"]
                    max_epoch = epoch
                    max_measures = deepcopy(epoch_count[epoch]["measures"])
            print("The Most Epoch is {:<5d}".format(max_epoch))
            print("Its num is {:<5d}".format(max_count))
            print("Corresponding Measures: {}\n".format(max_measures))
            
            if max_count == 1:
                print("According to {}, the final best results is: \n".format(default))
                print(self.best_dict[default]["best_te_results"])
                self.best_measure = default
            else:
                print("According to {}, the final best results is: \n".format(max_measures))
                print(self.best_dict[max_measures[0]]["best_te_results"])
                self.best_measure = max_measures[0]
        else:
            print("According to {}, the final best results is: \n".format(index))
            print(self.best_dict[index]["best_te_results"])
            self.best_measure = index

    def save(self, save_path):
        torch.save(self.best_dict[self.best_measure]["best_weight"], save_path)