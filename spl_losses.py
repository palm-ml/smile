from typing import overload
import torch
from torch.autograd import grad
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F

LOG_EPSILON = 1e-5

'''
helper functions
'''

def neg_log(x):
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

def weighted_bce_loss(preds, obs, weights, bias=False):
    loss_mtx = torch.zeros_like(obs)
    weights[obs == 1.0] = 1.0
    loss_mtx[obs == 1.0] = (weights * neg_log(preds) + (1 - weights) * neg_log(1 - preds))[obs == 1.0]
    if bias:
        loss_mtx[obs == 0.0] =  ((1 / (obs.shape[-1]-1)) * (weights * neg_log(preds) + (1 - weights) * neg_log(1 - preds)))[obs == 0.0]
    else:
        loss_mtx[obs == 0.0] =  (weights * neg_log(preds) + (1 - weights) * neg_log(1 - preds))[obs == 0.0]
    return loss_mtx.mean()

def beta_kl_loss(alpha, beta, prior_value, eps=1e-6):
    prior = torch.ones_like(alpha) * prior_value
    alpha = alpha + eps
    beta = beta + eps
    KL = (torch.lgamma(alpha) + torch.lgamma(beta)) + \
        ((alpha - prior)*(torch.digamma(alpha))+(beta - prior)*(torch.digamma(beta)))
    return KL.mean()

def vae_loss(args, batch_X, batch_Y_obs, batch_Y_score, batch_D_score_1, batch_D_score_2, batch_G_A, batch_X_rec, batch_Y_rec, batch_A_rec):
    # for encoder and decoder
    loss_align = weighted_bce_loss(batch_D_score_1, batch_Y_obs.clone().detach(), batch_D_score_2.clone().detach(), True)
    loss_recx  = F.mse_loss(batch_X_rec, batch_X.clone().detach())
    loss_recy  = weighted_bce_loss(batch_Y_rec, batch_Y_obs.clone().detach(), batch_Y_obs.clone().detach()) 
    loss_recA  = F.mse_loss(batch_A_rec, batch_G_A.detach())
    loss_kl    = beta_kl_loss(batch_D_score_1, 1 - batch_D_score_1, 1)
    loss_1 = loss_align + args.gamma * (loss_recx + loss_recy + loss_recA) + args.delta * loss_kl
    # for label estimator
    loss_base  = weighted_bce_loss(batch_D_score_2, batch_Y_obs.clone().detach(), batch_D_score_1.clone().detach(), True)

    return loss_1 + args.base * loss_base
    
def loss_an(preds, Y_obs):
    # input validation: 
    assert torch.min(Y_obs) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(Y_obs)
    loss_mtx[Y_obs == 1] = neg_log(preds[Y_obs == 1])
    loss_mtx[Y_obs == 0] = neg_log(1.0 - preds[Y_obs == 0])
    return loss_mtx.mean()
