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

def gen_random_spl_tr_va_te(source_fold, ds, target_fold):
    print("loading Dataset {}".format(ds))
    path = os.path.join(source_fold, ds + ".mat")
    data = loadmat(path)
    X, Y, Y_obs, tr_idx, va_idx, te_idx = data['X'], data['Y'], data['Y_obs'], data['tr_idx'], data['va_idx'], data['te_idx']
    tr_idx_list = tr_idx[0].tolist()
    va_idx_list = va_idx[0].tolist()
    print(len(X))
    print(len(tr_idx[0]), len(va_idx[0]), len(te_idx[0]))
    for idx in va_idx_list:
        tr_idx_list.remove(idx) 
    tr_idx = np.array([tr_idx_list])
    print(len(tr_idx[0]), len(va_idx[0]), len(te_idx[0]))
    # check
    data['tr_idx'] = deepcopy(tr_idx)
    data['va_idx'] = deepcopy(va_idx)
    data['te_idx'] = deepcopy(te_idx)
    print(len(data['tr_idx'][0]), len(data['va_idx'][0]), len(data['te_idx'][0]))
    savemat(r"{}/{}.mat".format(target_fold, ds), data)


if __name__ == "__main__":
    torch.set_printoptions(threshold=1e6)
    ds = "CAL500"
    for key in look_up.keys():
        gen_random_spl_tr_va_te("SPL_Datasets_mat/", key, "SPL_Datasets_rad/")