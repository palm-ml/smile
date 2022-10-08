import os
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import euclidean_distances
import dgl

def adj_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def gen_adj_matrix(X, k=10, path=""):
    if os.path.exists(path):
        print("Found adj matrix file and Load.")
        adj_m = np.load(path)
        print("Adj matrix Finished.")
    else:
        print("Not Found adj matrix file and Compute.")
        dm = euclidean_distances(X, X)
        adj_m = np.zeros_like(dm)
        row = np.arange(0, X.shape[0])
        dm[row, row] = np.inf
        for _ in range(0, k):
            col = np.argmin(dm, axis=1)
            dm[row, col] = np.inf
            adj_m[row, col] = 1.0
        np.save(path, adj_m)
        print("Adj matrix Finished.")
    adj_m = sp.coo_matrix(adj_m)
    adj_m = adj_normalize(adj_m + sp.eye(adj_m.shape[0]))
    adj_m = sparse_mx_to_torch_sparse_tensor(adj_m)
    return adj_m

def gen_adj_matrix_2(X, k=10, path=""):
    if os.path.exists(path):
        print("Found adj matrix file and Load.")
        adj_m = np.load(path)
        print("Adj matrix Finished.")
    else:
        print("Not Found adj matrix file and Compute.")
        dm = euclidean_distances(X, X)
        adj_m = np.zeros_like(dm)
        row = np.arange(0, X.shape[0])
        dm[row, row] = np.inf
        for _ in range(0, k):
            col = np.argmin(dm, axis=1)
            dm[row, col] = np.inf
            adj_m[row, col] = 1.0
        np.save(path, adj_m)
        print("Adj matrix Finished.")
    adj_coo = sp.coo_matrix(adj_m)
    adj_coo = adj_coo.tocoo().astype(np.float32)
    g = dgl.graph((torch.LongTensor(adj_coo.row).cuda(), torch.LongTensor(adj_coo.col).cuda()))
    print(g.device)
    return g, torch.FloatTensor(adj_m)

