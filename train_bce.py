from copy import deepcopy
import os
from statistics import mode
import numpy as np
import torch
import torch.nn.functional as F
from spl_models import MLPNet
from spl_losses import weighted_bce_loss, expected_positive_regularizer, vae_loss, loss_an
from gen_adj import gen_adj_matrix_2 as gen_adj_matrix
import random
import nni
# utils
from utils import evaluate, prepare_data, prepare_dataloader
import metrics
from metrics import Coverage, RankingLoss, OneError, AveragePrecision, HammingLoss

import argparse
parser = argparse.ArgumentParser(
        prog='RIES demo file.',
        usage='GCN and base p.',
        description='.',
        epilog='end',
        add_help=True
)
# basic setting
parser.add_argument('--ds', help='specify a dataset', type=str, \
    choices=['bookmarks', 'CAL500', 'corel5k', 'Corel16k001',
    'delicious', 'espgame', 'iaprtc12', 'Image', 'mediamill',
    'mirflickr', 'rcv1subset1', 'scene', 'tmc2007', 'yeast'], default='yeast')
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--bs', help='batch size', type=int, default=8)
parser.add_argument('--mo', help='model str', type=str, default="mlp")
parser.add_argument('--ep', help='training epochs', type=int, default=25)
parser.add_argument('--nw', help='number of workers', type=int, default=8)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--source_fold', help='data folder', type=str, default="SPL_Datasets_mat/")
parser.add_argument('--gpu', help='gpu id', type=str, default="0")
parser.add_argument('--warm_up', help='warm up epochs', type=int, default=3)
parser.add_argument('--p', help='initial distribution', type=float, default=0.5)
parser.add_argument('--T', help='temperature', type=float, default=1)
parser.add_argument('--bias', help='', type=bool, default=True)
# loss coef
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta' , type=float, default=1)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--delta', type=float, default=1)
parser.add_argument('--mu',    type=float, default=1)
parser.add_argument('--base',  type=float, default=1)
parser.add_argument('--an',    type=float, default=1)
parser.add_argument('--wan',    type=float, default=1)
args = parser.parse_args()

look_up = {
        "CAL500":      [502,   68,   174,   26.044],
        "scene":       [2407,  294,  6,     1.074 ],
        "yeast":       [2417,  103,  14,    4,237 ],
        "corel5k":     [5000,  499,  374,   3.522 ],
        "rcv1subset1": [6000,  944,  101,   2.880 ],
        "Corel16k001": [13766, 500,  153,   2.859 ],
        "delicious":   [16091, 500,  983,   19.02 ],
        "tmc2007":     [28596, 981,  22,    2.158 ],
        "mediamill":   [42177, 120,  101,   4.376 ],
        "bookmarks":   [87856, 2150, 208,   2.028 ],
        "Image":       [2000,  294,  5,     1.236 ],
        "iaprtc12":    [19627, 1000, 291,   5.719 ],
        "espgame":     [20768, 1000, 268,   4.686 ],
        "mirflickr":   [24581, 1000, 38,    4.716 ],
    }

def set_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_default_params():
    params = {
        "lr": 0.001,
        "bs": 16,
        "alpha": 1,
        "beta": 1,
        "gamma": 0.001,
        "delta": 1,
        "mu": 0.5,
        "base": 0.5,
        "an": 1,
        "wan": 1
    }
    return params

def params_to_args(args, params):
    args.lr, args.bs = params['lr'], params['bs']
    args.alpha. args.beta, args.gamma, args.delta = params['alpha'], params['beta'], params['gamma'], params['delta']
    args.mu, args.base, args.an = params['mu'], params['base'], params['an']
    return args

def transform_nni_params(args, r_params):
    args.lr    = r_params['lr']
    args.bs    = r_params['bs']
    args.alpha = r_params['alpha']
    args.beta  = r_params['beta']
    args.gamma = r_params['gamma']
    args.delta = r_params['delta']
    args.mu    = r_params['mu']
    args.base  = r_params['base']
    args.an    = r_params['an']
    args.wan   = r_params['wan']
    return args

def neg_log(x):
    LOG_EPSILON = 1e-5
    return - torch.log(x + LOG_EPSILON)

def log_loss(preds, targs):
    return targs * neg_log(preds)

def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos)**2
    else:
        raise NotImplementedError
    return reg

def loss_bce(preds, Y):
    # input validation: 
    assert torch.min(Y) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(Y)
    loss_mtx[Y == 1] = neg_log(preds[Y == 1])
    loss_mtx[Y == 0] = neg_log(1.0 - preds[Y == 0])
    return loss_mtx.mean()


def train(args):
    print(args)
    # random seed 
    set_seed(args.seed)
    # cuda device
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    # load data
    train_X, train_Y, train_s_Y, valid_X, valid_Y, valid_s_Y, test_X, test_Y, test_s_Y = prepare_data(args)
    train_loader, valid_loader, test_loader = prepare_dataloader(args, train_X, train_Y, train_s_Y, valid_X, valid_Y, valid_s_Y, test_X, test_Y, test_s_Y)
    
    # model
    if args.mo == "mlp":
        model = MLPNet(look_up[args.ds][1], look_up[args.ds][2]).cuda()
    
    # optimizer
    mlp_opt_params = [
        {'params': model.parameters(),        'lr': args.lr  }
    ]
    
    mlp_opt = torch.optim.Adam(mlp_opt_params)
    # start training
    best_valid_score, best_test_score, best_test_result_dict, best_epoch, best_weight = -1, -1, -1, -1, None

    for epoch in range(0, args.ep):
        # train
        model.train()
        
        for i, (batch_X, batch_Y_obs, batch_Y, batch_idx) in enumerate(train_loader):
            # classifier
            batch_X, batch_Y_obs, batch_Y = batch_X.cuda(), batch_Y_obs.cuda(), batch_Y.cuda()
            batch_Y_logits = model(batch_X)
            batch_Y_score  = torch.sigmoid(batch_Y_logits / args.T)
        
            # compute loss
            mlp_opt.zero_grad()
            loss_predict = loss_bce(batch_Y_score, batch_Y)
            loss_predict.backward()
            mlp_opt.step()

        # evaluate
        valid_result_dict, valid_score = evaluate(model, valid_X, valid_Y, "valid")
        test_result_dict,  test_score  = evaluate(model, test_X,  test_Y)
        # nni.report_intermediate_result({"default": valid_score}.update({**valid_result_dict, **test_result_dict}))
        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_test_score = test_score
            best_test_result_dict = test_result_dict
            best_epoch = epoch
            best_weight = deepcopy(model.state_dict())
        print("Epoch: {}, valid score: {:.3f}, test score: {:.3f}, loss_predict: {:.3f}.".format(epoch, \
            valid_score, test_score, loss_predict.item()))
        print("Valid Measures: ")
        print(valid_result_dict)
        print("Test Measures: ")
        print(test_result_dict)
    nni.report_final_result(best_test_score)
    print("Best Epoch: {}".format(best_epoch))
    print("Best Measures: ")
    print(best_test_result_dict)
    print("Saving model ...")
    torch.save(best_weight, "middle/weight/bce/{}.pt".format(args.ds))
    print("Success!")
    return


if __name__ == '__main__':
    params = get_default_params()
    r_params = nni.get_next_parameter()
    params.update(r_params)
    args = transform_nni_params(args, params)
    train(args)
    
    
