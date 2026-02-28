"""Adjacency matrix construction for LUMO and QUGS datasets."""

import numpy as np
import scipy.sparse as sp
import torch


def normalize(mx):
    """Row-normalize a scipy sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


def sparse_to_torch(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def _lumo_adjacency():
    """18x18 adjacency for the LUMO dataset (paper Fig. 4a).

    Sensors are arranged as alternating radial/tangential pairs at 9 locations:
      R1, M1, R2, M2, ..., R9, M9
    Each sensor is connected to its neighbor in the ring topology.
    """
    adj = sp.coo_matrix([
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    ], shape=(18, 18), dtype=np.float32)
    return adj


def _qugs_adjacency():
    """30x30 adjacency for the QUGS dataset (paper Fig. 12).

    Tridiagonal band matrix from joint proximity: each sensor is connected
    to its immediate neighbours in the structural chain.
    """
    n = 30
    data = np.ones(n, dtype=np.float32)
    diag_main = sp.diags(data, 0, shape=(n, n))
    data_off = np.ones(n - 1, dtype=np.float32)
    diag_upper = sp.diags(data_off, 1, shape=(n, n))
    diag_lower = sp.diags(data_off, -1, shape=(n, n))
    adj = (diag_main + diag_upper + diag_lower).tocoo().astype(np.float32)
    return adj


def get_adjacency(dataset='lumo'):
    """Return normalized adjacency matrices for the given dataset.

    Args:
        dataset: 'lumo' (18 sensors) or 'qugs' (30 sensors).

    Returns:
        adj_norm: Normalized adjacency (without self-loops), torch sparse.
        adj_with_self_loops: Normalized adjacency + I, torch sparse.
        num_sensors: Number of sensor nodes.
    """
    if dataset == 'lumo':
        adj = _lumo_adjacency()
    elif dataset == 'qugs':
        adj = _qugs_adjacency()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'lumo' or 'qugs'.")

    num_sensors = adj.shape[0]
    adj_with_self_loops = normalize(adj + sp.eye(num_sensors))
    adj_norm = normalize(adj)

    return (sparse_to_torch(adj_norm),
            sparse_to_torch(adj_with_self_loops),
            num_sensors)
