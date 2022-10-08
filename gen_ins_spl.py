from copy import deepcopy
import torch
import torch.nn.functional as F
from spl_models import MLPNet
import os
from scipy.io.matlab.mio import loadmat, savemat
import random
import numpy as np

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

def gen_instance_dependent_spl(source_fold, ds, target_fold):
    print("loading Dataset {}".format(ds))
    path = os.path.join(source_fold, ds + ".mat")
    data = loadmat(path)
    X, Y, _, tr_idx, va_idx, te_idx = data['X'], data['Y'], data['Y_obs'], data['tr_idx'], data['va_idx'], data['te_idx']
    print(len(X))
    print(len(tr_idx[0]), len(te_idx[0]), len(va_idx[0]))
    # load model
    model = MLPNet(look_up[ds][1], look_up[ds][2]).cuda()
    model.load_state_dict(torch.load("middle/weight/bce/{}.pt".format(ds)))
    with torch.no_grad():
        Y_logits = model(torch.FloatTensor(X).cuda())
        Y_score  = torch.sigmoid(Y_logits)
        Y_max    = torch.argmax(Y_score * torch.FloatTensor(Y).cuda(), dim=1)
    Y_obs = torch.zeros_like(torch.FloatTensor(Y).cuda())
    Y_obs[torch.arange(0, len(Y_obs)), Y_max] = 1.0
    Y_obs = deepcopy(Y_obs.cpu().numpy())
    data['Y_obs'] = Y_obs
    # check
    check = (Y_obs * (1 - Y)).sum()
    print(check)
    savemat(r"{}/{}.mat".format(target_fold, ds), data)


if __name__ == "__main__":
    torch.set_printoptions(threshold=1e6)
    ds = "tmc2007"
    gen_instance_dependent_spl("SPL_Datasets_mat/", ds, "SPL_Datasets_rad/")