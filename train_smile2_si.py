import os
import numpy as np
import torch
import torch.nn.functional as F
from spl_models import MLPNet, Decoder, LabelEstimator, GCN
from spl_losses import weighted_bce_loss, expected_positive_regularizer, vae_loss, loss_an
from gen_adj import gen_adj_matrix_2 as gen_adj_matrix
import random
import nni
# utils
from utils import evaluate, prepare_data, prepare_dataloader, Best


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
parser.add_argument('--bs', help='batch size', type=int, default=16)
parser.add_argument('--mo', help='model str', type=str, default="mlp")
parser.add_argument('--loss', help='loss function', type=str, default="ries")
parser.add_argument('--lr_decay_rate', help='learning rate', type=float, default=1)
parser.add_argument('--lr_decay_step', help='learning rate', type=float, default=25)
parser.add_argument('--ep', help='training epochs', type=int, default=25)
parser.add_argument('--nw', help='number of workers', type=int, default=8)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--source_fold', help='data folder', type=str, default="SPL_Datasets_rad/")
parser.add_argument('--gpu', help='gpu id', type=str, default="0")
parser.add_argument('--warm_up', help='warm up epochs', type=int, default=0)
parser.add_argument('--p', help='initial distribution', type=float, default=0.5)
parser.add_argument('--T', help='temperature', type=float, default=1)
parser.add_argument('--rate', help='positive negative rate', type=float, default=1)
parser.add_argument('--bias', help='', type=int, default=1)
# loss coef
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta' , type=float, default=1)
parser.add_argument('--gamma', type=float, default=0.001)
parser.add_argument('--delta', type=float, default=1)
parser.add_argument('--mu',    type=float, default=0.5)
parser.add_argument('--base',  type=float, default=0.5)
parser.add_argument('--an',    type=float, default=1)
parser.add_argument('--wan',    type=float, default=1)
parser.add_argument('--role',    type=float, default=1)

args = parser.parse_args()

look_up = {
        "CAL500":      [502,   68,   174,   26.044],
        "scene":       [2407,  294,  6,     1.074 ],
        "yeast":       [2417,  103,  14,    4.237 ],
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

class adjust_lr:
    def __init__(self, lr, ep, decayrate, decaystep):
        self.lr_plan = [lr] * ep 
        for i in range(0, ep):
            self.lr_plan[i] = lr * decayrate ** (i // decaystep)

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_plan[epoch]

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

def loss_an(preds, Y_obs):
    # input validation: 
    assert torch.min(Y_obs) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(Y_obs)
    loss_mtx[Y_obs == 1] = neg_log(preds[Y_obs == 1])
    loss_mtx[Y_obs == 0] = neg_log(1.0 - preds[Y_obs == 0])
    return loss_mtx.mean()

def loss_wan(preds, Y_obs):
    # input validation: 
    assert torch.min(Y_obs) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(Y_obs)
    loss_mtx[Y_obs == 1] = neg_log(preds[Y_obs == 1])
    loss_mtx[Y_obs == 0] = neg_log(1.0 - preds[Y_obs == 0]) / (Y_obs.shape[-1]-1)
    return loss_mtx.mean()

class role:
    def __init__(self, Y) -> None:
        self.label_est = LabelEstimator(Y, None).cuda()

    def __call__(self, preds, Y_obs, expected_num_pos, idx):
        # unpack:
        estimated_labels = self.label_est(idx)
        # input validation:
        # assert torch.min(Y_obs) >= 0
        # (image classifier) compute loss w.r.t. observed positives:
        loss_mtx_pos_1 = torch.zeros_like(Y_obs)
        loss_mtx_pos_1[Y_obs == 1] = neg_log(preds[Y_obs == 1])
        # (image classifier) compute loss w.r.t. label estimator outputs:
        estimated_labels_detached = estimated_labels.detach()
        loss_mtx_cross_1 = torch.zeros_like(Y_obs)
        loss_mtx_cross_1[Y_obs == 0] = (estimated_labels_detached * neg_log(preds) + (1.0 - estimated_labels_detached) * neg_log(1.0 - preds))[Y_obs == 0]
        # (image classifier) compute regularizer: 
        reg_1 = expected_positive_regularizer(preds, expected_num_pos, norm='2') / (Y_obs.size(1) ** 2)
        # (label estimator) compute loss w.r.t. observed positives:
        loss_mtx_pos_2 = torch.zeros_like(Y_obs)
        loss_mtx_pos_2[Y_obs == 1] = neg_log(estimated_labels[Y_obs == 1])
        # (label estimator) compute loss w.r.t. image classifier outputs:
        preds_detached = preds.detach()
        loss_mtx_cross_2 = preds_detached * neg_log(estimated_labels) + (1.0 - preds_detached) * neg_log(1.0 - estimated_labels)
        # (label estimator) compute regularizer:
        reg_2 = expected_positive_regularizer(estimated_labels, expected_num_pos, norm='2') / (Y_obs.size(1) ** 2)
        # compute final loss matrix:
        reg_loss = 0.5 * (reg_1 + reg_2)
        # reg_loss = None
        loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
        loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)
        
        return loss_mtx.mean() + reg_loss

def weighted_bce_loss(preds, obs, weights, bias=True, rate=1.0):
    loss_mtx = torch.zeros_like(obs)
    loss_mtx = weights * neg_log(preds) + (1 - weights) * neg_log(1 - preds)
    if bias:
        loss_mtx[obs == 0.0] =  loss_mtx[obs == 0.0] / (obs.shape[-1]-1)
    else:
        loss_mtx[obs == 0.0] =  loss_mtx[obs == 0.0] * rate
    return loss_mtx.mean()

def origin_align_loss(preds, obs):
    return torch.nn.MSELoss()(preds, obs)

def beta_kl_loss(alpha, beta, prior_value, eps=1e-6):
    prior = torch.ones_like(alpha) * prior_value
    alpha = alpha + eps
    beta = beta + eps
    KL = (torch.lgamma(alpha) + torch.lgamma(beta)) + \
        ((alpha - prior)*(torch.digamma(alpha))+(beta - prior)*(torch.digamma(beta)))
    return KL.mean()

def vae_loss(args, batch_X, batch_Y_obs, batch_D_score_1, batch_G_A, batch_X_rec, batch_Y_rec, batch_A_rec, loss_role, batch_idx):
    # for encoder and decoder
    # loss_align_1 = args.an * loss_an(batch_D_score_1, batch_Y_obs) + args.wan * loss_wan(batch_D_score_1, batch_Y_obs) + args.role * loss_role(batch_D_score_1, batch_Y_obs, look_up[args.ds][-1], batch_idx)
    loss_align_1 = args.an * loss_an(batch_D_score_1, batch_Y_obs)
    # loss_align_2 = weighted_bce_loss(batch_D_score_1, batch_Y_obs, batch_Y_score.clone().detach())
    # loss_align = loss_align_1 + args.base * loss_align_2
    loss_recx  = F.mse_loss(batch_X_rec, batch_X.clone().detach())
    loss_recy  = loss_an(batch_Y_rec, batch_Y_obs)
    loss_recA  = F.mse_loss(batch_A_rec, batch_G_A.detach())
    loss_kl    = beta_kl_loss(batch_D_score_1, 1 - batch_D_score_1, 1)
    loss       = loss_align_1 + args.gamma * (loss_recx + loss_recy + loss_recA) + args.delta * loss_kl

    return loss, loss_align_1, loss - loss_align_1


def train(args):
    print(args)
    # random seed 
    # set_seed(args.seed)
    # cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # load data
    train_X, train_Y, train_s_Y, valid_X, valid_Y, valid_s_Y, test_X, test_Y, test_s_Y = prepare_data(args)
    train_loader, valid_loader, test_loader = prepare_dataloader(args, train_X, train_Y, train_s_Y, valid_X, valid_Y, valid_s_Y, test_X, test_Y, test_s_Y)
    # adj matrix
    G, G_A = gen_adj_matrix(X=train_X, k=3, path="middle/random_adj/{}.npy".format(args.ds))
    G_X = torch.tensor(train_X)
    G_X = torch.Tensor(G_A@G_X)
    # model
    hid_dim = look_up[args.ds][2]
    model = MLPNet(look_up[args.ds][1], look_up[args.ds][2]).cuda()
    encoder = GCN(look_up[args.ds][1], look_up[args.ds][2], look_up[args.ds][2]).cuda()
    encoder_z = MLPNet(look_up[args.ds][1] + look_up[args.ds][2], hid_dim).cuda()
    decoder = Decoder(look_up[args.ds][2], hid_dim, look_up[args.ds][1]).cuda()

    lr_adjustor = adjust_lr(args.lr, args.ep, args.lr_decay_rate, args.lr_decay_step)
    # optimizer
    mlp_opt_params = [
        {'params': model.parameters(),        'lr': args.lr  }
    ]
    gcn_opt_params = [
        {'params': encoder.parameters(),      'lr': args.lr  }, 
        {'params': decoder.parameters(),      'lr': args.lr  },
        {'params': encoder_z.parameters(),      'lr': args.lr  }
    ]
    # warm_opt = torch.optim.Adam([opt_params[0], opt_params[2]])
    mlp_opt = torch.optim.Adam(mlp_opt_params)
    gcn_opt = torch.optim.Adam(gcn_opt_params)
    # mlp_opt = torch.optim.SGD(mlp_opt_params, momentum=0.9)
    # gcn_opt = torch.optim.SGD(gcn_opt_params, momentum=0.9)

    # warming
    train_X = torch.FloatTensor(train_X).cuda()
    train_Y = torch.FloatTensor(train_Y).cuda()
    for epoch in range(0, args.warm_up):
        model.train()
        for i, (batch_X, batch_Y_obs, batch_Y, batch_idx) in enumerate(train_loader):
            # # classifier
            batch_X, batch_Y_obs, batch_Y = batch_X.cuda(), batch_Y_obs.cuda(), batch_Y.cuda()
            batch_Y_logits = model(batch_X)
            batch_Y_score  = torch.sigmoid(batch_Y_logits)
            # label distribution
            batch_D_logits_1 = encoder(G, train_X)[batch_idx]
            # batch_D_logits_1 = encoder(batch_X)
            batch_D_score_1  = torch.sigmoid(batch_D_logits_1)
            # compute loss
            mlp_opt.zero_grad()
            loss_predict = loss_an(batch_Y_score, batch_Y_obs)
            loss_predict.backward()
            mlp_opt.step()

            gcn_opt.zero_grad()
            loss_distribution = loss_an(batch_D_score_1, batch_Y_obs)
            loss_distribution.backward()
            gcn_opt.step()
        # evaluate
        valid_result_dict, valid_score = evaluate(model, valid_X, valid_Y, "valid")
        test_result_dict,  test_score  = evaluate(model, test_X,  test_Y)
        print("Epoch: {}, valid score: {:.3f}, loss_warm: {:.3f}.".format(epoch, \
            valid_score, loss_predict.item()))
        print("Valid Measures: ")
        print(valid_result_dict)
        print("Test Measures: ")
        print(test_result_dict)

    # start training
    measures=[
        "Hammingloss", "Coverage", "AveragePrecision", "RankingLoss", "OneError"
    ]
    directions=["min", "min", "max", "min", "min"]
    best_recorder = Best(measures, directions)

    loss_role = role(train_s_Y)

    for epoch in range(0, args.ep):
        # train
        lr_adjustor.adjust_learning_rate(mlp_opt, epoch)
        model.train()
        encoder.train()
        decoder.train()
        for i, (batch_X, batch_Y_obs, batch_Y, batch_idx) in enumerate(train_loader):
            # classifier
            batch_X, batch_Y_obs, batch_Y = batch_X.cuda(), batch_Y_obs.cuda(), batch_Y.cuda()
            batch_Y_logits = model(batch_X)
            batch_Y_score  = torch.sigmoid(batch_Y_logits / args.T)
            # label distribution
            # batch_G_A = G_A[batch_idx, :][:, batch_idx].cuda()
            # batch_D_logits_1 = encoder(G, train_X)[batch_idx]
            # # batch_D_logits_1 = encoder(batch_X)
            # batch_D_score_1  = torch.sigmoid(batch_D_logits_1 / args.T)
            # hid_feat = encoder_z(torch.cat((batch_X, batch_D_score_1), dim=1))
            # batch_X_rec, batch_Y_rec, batch_A_rec = decoder(batch_D_score_1, hid_feat)

            batch_D_score_2  = batch_Y_obs.clone().detach()
            batch_D_score_2[batch_D_score_2 == 0] = args.p

            batch_D_score = args.mu * batch_Y_score.clone().detach() + (1 - args.mu) * batch_D_score_2
            batch_D = batch_D_score.clone().detach()

            # compute loss
            mlp_opt.zero_grad()
            loss_predict = weighted_bce_loss(batch_Y_score, batch_Y_obs, batch_D, args.bias, args.rate) #+ origin_align_loss(batch_Y_score, batch_Y_obs)
            loss_predict.backward()
            mlp_opt.step()

            # gcn_opt.zero_grad()
            # loss_distribution, loss_align_1, loss_else = vae_loss(args, batch_X, batch_Y_obs, batch_D_score_1, batch_G_A, batch_X_rec, batch_Y_rec, batch_A_rec, loss_role, batch_idx)
            # loss_distribution.backward()
            # gcn_opt.step()
        # evaluate
        valid_result_dict = evaluate(model, valid_X, valid_Y)
        test_result_dict  = evaluate(model, test_X,  test_Y)
        # nni.report_intermediate_result({"default": valid_score}.update({**valid_result_dict, **test_result_dict}))
        print("Epoch: {}, loss: {:.3f}.".format(epoch, loss_predict.item()))
        print("Valid Measures: ")
        print(valid_result_dict)
        print("Test Measures: ")
        print(test_result_dict)
        best_recorder.update(epoch, valid_result_dict, test_result_dict, model)
    # nni.report_final_result(best_test_score)
    best_recorder.show("va")
    best_recorder.show("te")
    best_recorder.select()
    # os.makedirs("baseline_weight/rad/{}/{}".format(args.loss, args.ds))
    # best_recorder.save("baseline_weight/rad/{}/{}/{}_{}.pt".format(args.loss, args.ds, args.ds, args.seed))
    return


if __name__ == '__main__':
    # params = get_default_params()
    # r_params = nni.get_next_parameter()
    # params.update(r_params)
    # args = transform_nni_params(args, params)
    train(args)
    
    
