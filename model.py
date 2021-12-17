import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sample import sampleScope, computeSubGraph
from sklearn import metrics

# GCN template
class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, step_num,
                 nonlinear, dropout, weight_decay):
        super(GCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.step_num = step_num

        self.W =  nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False)])
        for _ in range(step_num-2):
            self.W.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.W.append(nn.Linear(hidden_dim, output_dim, bias=False))

        for w in self.W:
            nn.init.xavier_uniform_(w.weight)

        self.nonlinear = nonlinear
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay

    def initialize(self):
        return NotImplementedError

    def forward(self, ids):
        return NotImplementedError

    def calc_loss(self, y_pred, y_true):
        loss_train = F.cross_entropy(y_pred, y_true)
        return loss_train

    def calc_f1(self, y_pred, y_true):
        y_pred = torch.argmax(y_pred, dim=1)
        return metrics.f1_score(y_true, y_pred, average="micro"),\
            metrics.f1_score(y_true, y_pred, average="macro")

# GCN models generating sub-adjacency matrices for every batch
class sampleGCN(GCN):

    def __init__(self, input_dim, output_dim, hidden_dim, step_num,
                 graph, feature, sample_scope, sample_num,
                 nonlinear, dropout, weight_decay):
        super(sampleGCN, self).__init__(input_dim, output_dim, hidden_dim, step_num,
                                        nonlinear, dropout, weight_decay)

        self.node_num = graph.shape[0]
        self.sample_scope = sample_scope
        self.sample_num = sample_num

        # Indices of neighbors
        self.edge = nn.Parameter(torch.ones(self.node_num+1, sample_scope, dtype=torch.float))
        self.edge.requires_grad = False

        # Weights (attention) of neighbors
        self.weight = torch.zeros(self.node_num+1, sample_scope, dtype=torch.float)
        self.weight.requires_grad = False

        # Decide the neighborhood for sampling
        sampleScope(self.sample_scope, graph, self.edge, self.weight)

        # Trainable parameters of sampling probability function
        self.sample_W = nn.Parameter(torch.zeros(size=(input_dim, hidden_dim)))
        self.sample_W.requires_grad = True
        nn.init.xavier_uniform_(self.sample_W.data, gain=1.414)
        self.sample_W2 = nn.Parameter(torch.zeros(size=(input_dim, hidden_dim)))
        self.sample_W2.requires_grad = True
        nn.init.xavier_uniform_(self.sample_W2.data, gain=1.414)
        self.sample_a = nn.Parameter(torch.FloatTensor(np.array([[10e-3], [10e-3], [10e-1]])))
        self.sample_a.requires_grad = True
        self.softmax_a = nn.Softmax(dim=0)

        # Feature matrix
        self.feature = torch.cat([feature, torch.zeros(1, input_dim)], dim=0)

    # Loss for sampling probability parameters
    def sample_loss(self, loss_up):
        # batch_sampler: nodes from upper layer sampled their neighbors
        # batch_sampled: nodes from lower layer were sampled by their parents
        # log probability for "batch_sampler" to sample "batch_sampled"
        logp = self.get_policy(self.batch_sampler).log_prob(self.batch_sampled.transpose(0,1)).transpose(0,1)
        batch_sampled_node = torch.gather(self.edge[self.batch_sampler], dim=1, index=self.batch_sampled).type(torch.int64)
        X = logp.unsqueeze(2)*self.feature[batch_sampled_node].to(self.device)
        X = X.mean(dim=1)
        # Chain rule
        batch_loss = torch.bmm(loss_up.unsqueeze(1), X.unsqueeze(2))
        return batch_loss.mean()
    
    # A helper function to compute node attention
    def _node_attention(self, source, target, weight):
        # source: embedding of source nodes
        # target: embedding of target nodes
        # weight: weight for the attention transformation.
        ss = torch.mm(source.reshape(-1, self.input_dim), weight)
        tt = torch.mm(target.reshape(-1, self.input_dim), weight)
        att = torch.bmm(ss.unsqueeze(1),  tt.unsqueeze(2)).squeeze(2)
        return att

    # Compute sampling probability given source node.
    def weight_function(self, source_idx):
        # source_idx: parent node we use to compute the sampling weights.
        # 1st head of importance sampling
        neighbor_idx = self.edge[source_idx].type(torch.int64)
        source_ = self.feature[source_idx].unsqueeze(1).expand(-1, self.sample_scope, -1).to(self.device)
        target_ = self.feature[neighbor_idx].to(self.device)
        att1 = self._node_attention(source_, target_, self.sample_W)
        # 2nd head of importance sampling
        att2 =  self._node_attention(source_, target_, self.sample_W2)
        # Random sampling
        att3 = self.weight[source_idx].view(-1).unsqueeze(1).to(self.device)
        # Attention of Attentions
        att = torch.cat([att1, att2, att3], dim=1)
        att = F.relu(torch.mm(att, self.softmax_a(self.sample_a)))
        att = att + 10e-10*torch.ones_like(att)
        att = att.reshape(-1, self.sample_scope)
        return att

    # Get sampling probability distributions (= policy) of each node
    def get_policy(self, target_nodes):
        probs = self.weight_function(target_nodes)
        return torch.distributions.Categorical(probs=probs)

    # Sample neighbors (=action) from sampling probability distributions
    def get_action(self, target_nodes):
        policy = self.get_policy(target_nodes)
        return policy.sample_n(self.sample_num).transpose(0,1)

    # Sample neighbors and generate sub-adjacency matrices for the given batch
    def sampleNodes(self, nodes):
        all_adj = [[]] * self.step_num
        all_feats = [[]] * self.step_num
        # Top-1 layer = batch node
        in_nodes = nodes
        # Top-down sampling from top-1 layer to the input layer
        for i in range(self.step_num):
            layer = self.step_num - i - 1
            # Neighbors are sampled dynamically
            out_nodes = self.get_action(in_nodes).type(torch.int64)
            if layer == 0:
                self.batch_sampler = in_nodes
                self.batch_sampled = out_nodes
            # Generate sub-adjacency matrix
            out_nodes, adj = computeSubGraph(self.edge, self.weight, in_nodes, out_nodes)
            if layer == 0:
                start_feat = self.feature[out_nodes].to(self.device)
            all_feats[layer] = self.feature[in_nodes].to(self.device)
            all_adj[layer] = adj
            in_nodes = out_nodes
        return start_feat, all_feats, all_adj

    def forward(self, ids):
        # Sample neighbors and generate sub-adjacency matrices for the given batch
        X, feat_list, adj_list = self.sampleNodes(ids)
        for idx, (adj, feat, w) in enumerate(zip(adj_list, feat_list, self.W)):
            X = torch.sparse.mm(adj, X)
            # For loss computation: consider only 1st layer nodes
            if idx == 0:
                self.X1 = nn.Parameter(X)
                X = self.X1
            X = w(X)
            # Softmax of the final layer will be taken in loss function later
            if idx < self.step_num - 1:
                X = self.dropout(X)
                if self.nonlinear:
                    X = F.relu(X)
        return X
