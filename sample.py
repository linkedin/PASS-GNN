import numpy as np
import torch
import torch.nn.functional as f


# Decide neighborhood for sampling
def sampleScope(sample_scope, adj, edge, weight):
    # Graph is presented as edge/weight lists of fixed length (sample_scope)
    # Edge list maintains the indices of neighborhood
    # Weight list maintains the weight(attention) of neighborhood

    num_data = adj.shape[0]
    edge.data *= num_data
    for v in range(num_data):
        neighbors = torch.from_numpy(np.nonzero(adj[v, :])[1])
        len_neighbors = len(neighbors)
        len_neighbors = neighbors.shape[0]
        if len_neighbors == 0:
            edge.data[v, 0] = v
            weight.data[v, 0] = 1
        elif len_neighbors > sample_scope:
            perm = torch.randperm(len_neighbors)[:sample_scope]
            neighbors = neighbors[perm]
            edge.data[v, :sample_scope] = neighbors
            weight.data[v, :sample_scope] = 1./sample_scope
        else:
            edge.data[v, :len_neighbors] = neighbors
            weight.data[v, :len_neighbors] = 1./len_neighbors
    return

# Generate sub-adjacency matrix
def computeSubGraph(edge, weight, in_nodes, out_nodes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Collect nodes participated in the sub-adjacency matrix
    adj = torch.gather(edge[in_nodes], dim=1, index=out_nodes).type(torch.int64)
    # Unique indices for nodes to generate a new sub-adjacency matrix
    unique, index = torch.unique(adj.view(-1), return_inverse=True)
    adj_shape = adj.shape
    del adj

    row = (torch.arange(index.shape[0]) // adj_shape[1]).type(torch.int64).to(device)
    col = index
    attention = torch.ones(index.size()).to(device)
    indices = torch.stack([row, col])
    dense_shape = torch.Size([adj_shape[0], unique.shape[0]])
    new_adj = torch.sparse.FloatTensor(indices, attention, dense_shape).to_dense()
    del indices
    del attention

    # Unweighted attention: same as normalized binary adjacency matrix
    indices = torch.nonzero(new_adj, as_tuple=True)
    new_adj[indices] = 1
    del indices
    # Normalize
    row_sum = torch.sum(new_adj, dim=1)
    row_sum = torch.diag(1/row_sum)
    new_adj = torch.spmm(row_sum, new_adj)
    return unique, new_adj

