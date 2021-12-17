import torch
import numpy as np
import random
import scipy.sparse as sp

# Ratio for data split (training/validation/test)
train_ratio = 0.5
val_ratio = 0.1
test_ratio = 0.4

# Split dataset into training/validation/test sets
def split_dataset(labels):
    label_min = labels.min()
    label_max = labels.max() + 1
    idx = {}
    for i in range(label_min, label_max):
        idx[i] = []
    for i in range(len(labels)):
        idx[labels[i]].append(i)
    train_idx = []
    val_idx = []
    test_idx = []
    for i in range(label_min, label_max):
        train_num = int(train_ratio*len(idx[i]))
        for j in range(train_num):
            train_idx.append(idx[i][j])
    for i in range(label_min, label_max):
        train_num = int(train_ratio*len(idx[i]))
        val_num = int(val_ratio*len(idx[i]))
        for j in range(train_num, train_num+val_num):
            val_idx.append(idx[i][j])
    for i in range(label_min, label_max):
        train_val_num = int((train_ratio + val_ratio)*len(idx[i]))
        test_idx.extend(idx[i][train_val_num:-1])
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)
    return train_idx, val_idx, test_idx

# Load npz-format files
def load_npz(device, datadir='./Data', dataset="ms_cs"):
    dataset = datadir + "/" + dataset + ".npz"
    with np.load(dataset, allow_pickle = True) as loader:
        loader = dict(loader)
        graph = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])
        # Undirected graph
        graph = graph + graph.transpose()
        # Remove self-loop
        for i in range(graph.shape[0]):
            graph[i,i] = 1
        # Feature matrix
        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape'])
            features = torch.FloatTensor(sp.csr_matrix.toarray(attr_matrix))
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
            features = torch.FloatTensor(attr_matrix)
        else:
            attr_matrix = None
            features = None
        # Labels
        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']), shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

    features = torch.nn.functional.normalize(features, p=2, dim=1)

    train_idx, val_idx, test_idx = split_dataset(labels)
    label_num = labels.max() - labels.min() + 1
    labels = np.append(labels, [-1])
    labels = torch.LongTensor(labels).to(device)

    idx = {}
    idx["train"] = torch.LongTensor(train_idx).to(device)
    idx["val"] = torch.LongTensor(val_idx).to(device)
    idx["test"] = torch.LongTensor(test_idx).to(device)

    return graph, features, labels, label_num, idx

