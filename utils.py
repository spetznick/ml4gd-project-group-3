import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import pickle # to access dataframe faster than csv
import glob, re
import os
import csv
from pathlib import Path
import scipy as sp
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm # adjacency matrix normalization
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros # initialize weights and biases for nn
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm # for sparse matrix multiplication

class TemporalGCNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, P: int): # P from window_idx = np.arange(P)
        super(TemporalGCNLayer, self).__init__(aggr = 'add') # 'Add' aggregation
        self.K = K
        self.P = P
        # self.normalize = normalize
        # self.linear = nn.Linear(in_channels, out_channels)
        self.h = nn.Parameter(torch.Tensor(K+1, P)) # depends on the order of filter and P
        self.reset_parameters() # initialize parameters
        self.m = nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset_parameters(self):
        #nn.init.xavier_uniform_(self.linear.weight) # initialize the weight of the linear layer
        #nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.h)

    def forward(self, dataset) -> Tensor:
        # keys of the given dataset as tensor
        indices = torch.Tensor(list(dataset.keys())).int()
        batch_size = len(indices) # 128
        first_idx = indices[0].item() # since it is a tensor

        # Intialize output tensor # (428, 104)
        out = torch.zeros(dataset[first_idx].x.shape[0], batch_size - self.P).to(self.device)

        # Cache adjacency matrices and feature matrices
        adj_matrices = []
        x_t_matrices = []

        for i in range(first_idx, first_idx + batch_size):
            edge_index = dataset[i].edge_index.long().to(self.device)
            edge_weight = dataset[i].edge_weight.to(self.device)
            adj_sparse_tensor = SparseTensor(row = edge_index[0], col = edge_index[1], value = edge_weight)
            adj_matrices.append(adj_sparse_tensor)
            x_t_matrices.append(dataset[i].x.to(self.device))

        # Compute the output using cached matrices
        for i_p in range(batch_size - self.P):
            out_kp = torch.zeros_like(x_t_matrices[0]).to(self.device)

            for p in range(self.P):
                idx = i_p + p
                x_t_minus_p = x_t_matrices[idx]
                adj_sparse_tensor = adj_matrices[idx]

                for k in range(self.K + 1):
                    h_kp = self.h[k, p]
                    out_kp += h_kp * self.propagate(adj_sparse_tensor, x = x_t_minus_p, k = k)

            out[:, i_p] = out_kp.view(-1)

        return self.m(out)

    def propagate(self, adj_sparse_tensor, x = None, k = 1):
        x_out = x
        for _ in range(k):
            x_out = adj_sparse_tensor.matmul(x_out)
        return x_out

class TemporalGCN(nn.Module):
    def __init__(self, in_channels, out_channels, K, P):
        super(TemporalGCN, self).__init__()
        self.gcn_layer = TemporalGCNLayer(in_channels, out_channels, K, P)

    def forward(self, dataset):
        return self.gcn_layer(dataset)


def get_target(window_idx, all_nodes, P, df_agg):
    target_timestamps = window_idx[P:]
    # get target values for all the nodes active for training
    # target = torch.tensor(df_agg.loc[df_agg.index[target_timestamp],list(all_nodes)].values).view(-1,1).float().to(device)
    targets = torch.Tensor([])
    for tt in target_timestamps:
        if targets.numel() == 0:  # Check if targets is empty
            targets = torch.tensor(df_agg.loc[df_agg.index[tt], list(all_nodes)].values).view(-1, 1).float()
        else:
            new_tensor = torch.tensor(df_agg.loc[df_agg.index[tt], list(all_nodes)].values).view(-1, 1).float()
            targets = torch.cat((targets, new_tensor), dim=1)
    targets = torch.nan_to_num(targets)
    return targets


def get_subadj_book(window_idx, adjacency_matrix, df_agg, windows):
    """
    Input:
        - window_idx:       window index (for eg., range(0,23) for first 24 hours)
        - adjacency_matrix: full adjacency matrix
        - df_agg:           aggregated dataframe
        - windows:          window dataframe with LCLids and their timestamps
    Returns
        - idx_book:       dict of indices (of dataframe) of all active nodes throughout the window for all time instances
        - actnod_book:    dict of LCLids of active nodes throughout the window for all time instances
        - subadj_book:    dict of subadjacency matrices throughout the window for all time instances (can be of different sizes)
        - all_nodes:      set of LCLids of all active unique nodes inside the window_idx
        - node_feat_book: dict of node feature matrix throughout the window for all instances
        - edge_idx_book:  dict of edge index throughout the window for all instances
    """
    node_feat_book = {}
    edge_idx_book = {}
    subadj_book = {}
    idx_book = {}
    actnod_book = {}
    all_nodes = set()

    def process_idx(i, idx):
        idx_i, actnod_i, subadj_i, node_feat_i, edge_idx_i = get_snapshot_adjacency(idx, adjacency_matrix, df_agg, windows)
        return i, idx_i, actnod_i, subadj_i, node_feat_i, edge_idx_i

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_idx, i, idx) for i, idx in enumerate(window_idx)]
        #print(futures)
        for future in as_completed(futures):
            i, idx_i, actnod_i, subadj_i, node_feat_i, edge_idx_i = future.result()
            idx_book[i] = idx_i
            actnod_book[i] = actnod_i
            subadj_book[i] = subadj_i.toarray()
            node_feat_book[i] = node_feat_i
            edge_idx_book[i] = edge_idx_i
            all_nodes.update(actnod_i)

    return idx_book, actnod_book, subadj_book, all_nodes, node_feat_book, edge_idx_book


def get_snapshot_adjacency(timestampidx, full_adjacency_matrix, df_agg, windows):
    """
    timestampidx: Time index; For example [0]: '2012-01-01 00:00:00'
    full_adjacency_matrix (np.array()): adjacency matrix for all the LCLids
    df_agg: aggregated dataframe with timeseries for all LCLids
    windows: window dataframe with LCLids and their timestamps

    Returns:
    indices_active_nodes: dataframe indices of active nodes
    active nodes: LCLids of active nodes
    active_sparse_submat: adjacency matrix obtained for that particular timestamp (using timestampidx)
    """
    full_adjacency_matrix = full_adjacency_matrix.toarray()

    # get all the active nodes for that particular time-stamp
    active_nodes = df_agg.columns[df_agg.loc[df_agg.index[timestampidx], :].notna()]

    # indices of active nodes
    indices_active_nodes = windows[windows['LCLid'].isin(active_nodes.values)].index

    # active nodes sub-adjacency matrix
    active_adj_submat = full_adjacency_matrix[np.ix_(indices_active_nodes, indices_active_nodes)]

    if active_adj_submat.shape[0] != active_nodes.shape[0]:
        print(f'# active nodes = {active_nodes.shape}, while \
        Adjacency Matrix Shape = {active_adj_submat.shape}')
        raise RuntimeError()

    # create graph from the adjacency submatrix to check if it is connected
    active_sparse_submat = sp.sparse.bsr_array(active_adj_submat)

    ################ too slow ############################
    #G = networkx.from_scipy_sparse_array(active_sparse_submat)
    # check if the graph is fully connected
    # assert networkx.is_connected(G)

    # FOR FURTHER ANALYSIS
    #sparsity_submat = 1 - sp.sparse.bsr_matrix.count_nonzero(active_sparse_submat) \
    #/ np.prod(active_sparse_submat.shape)
    #print(f'Sparsity = {sparsity_submat}')


    # get edge indices from the adjacency submatrix COO Format
    # edge_index = torch.tensor(np.array(G.edges).T)
    #######################################################

    active_sparse_submat_coo = active_sparse_submat.tocoo()

    row = active_sparse_submat_coo.row
    col = active_sparse_submat_coo.col

    edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)

    # node feature matrix
    node_feat = torch.tensor(df_agg.loc[df_agg.index[timestampidx],active_nodes].values).view(-1,1)
    return indices_active_nodes, active_nodes, active_sparse_submat, node_feat, edge_index


def align_adjacency_matrix(active_nodes, subadj, all_nodes, node_index_map):
    """
    Input:
        - active_nodes: list of active nodes
        - subadj: sub-adjacency matrix for these active nodes
        - all_nodes: all the nodes for the entire window not just the instant where subadj and active_nodes are found
        - node_index_map: sorted all_nodes with {index: node...}


    """
    if not isinstance(subadj, np.ndarray):
        subadj = subadj.toarray()

    # Create a mapping from active node to index
    active_node_indices = [node_index_map[node] for node in active_nodes]

    # Create aligned subadjacency matrix
    aligned_subadj = np.zeros((len(all_nodes), len(all_nodes)))

    # Use numpy advanced indexing to place the subadj matrix in the correct positions
    aligned_subadj[np.ix_(active_node_indices, active_node_indices)] = subadj

    return aligned_subadj


def get_aligned_adj_book(subadj_book, node_feat_book, actnod_book, all_nodes):

    # create node index mapping
    # for ex., index_mapping = {0,1,2,3,4,5} for sorted nodes {1,2,3,4,5,6}
    node_index_map = {node: i for i, node in enumerate(sorted(all_nodes))} # node: idx dictionary

    # total nodes
    num_nodes = len(node_index_map)

    aligned_adj_book = {}
    aligned_node_feat_book = {}
    aligned_edge_index_book = {}

    for i, ((actkey, actnod), (subadjkey, subadj)) in enumerate(zip(actnod_book.items(), subadj_book.items())):

        aligned_adj_book[i] = align_adjacency_matrix(actnod, subadj, all_nodes, node_index_map)


        # Extract the indices of the non-zero elements (edges)
        row_indices, col_indices = np.nonzero(aligned_adj_book[i])

        # Combine the row and column indices to form the edge_index
        aligned_edge_index_book[i] = torch.tensor(np.vstack((row_indices, col_indices)))

        for j, key in enumerate(actnod.values):
            temp = np.zeros((len(all_nodes), 1))
            temp[node_index_map[key]] = node_feat_book[actkey][j]
        aligned_node_feat_book[i] = temp
    return aligned_adj_book, node_index_map, aligned_node_feat_book, aligned_edge_index_book


class TemporalGraphDataset:
    def __init__(self, adjacency_matrix, df_agg):
        # very dirty to import this here
        print('Warning: This function uses uk-smart-meter-aggregated/windows_agg_ids.pkl file.')
        file = open('uk-smart-meter-aggregated/windows_agg_ids.pkl','rb')
        df_windows = pickle.load(file)
        self.graph_data = {}
        self.adjacency_matrix = adjacency_matrix
        self.df_agg = df_agg
        self.df_window = df_windows
        self.all_nodes_dict = {}

    def add_batch_instance(self, window_idx, b):
        # extract the subadjacency matrix for all those time-stamps
        idx_book, actnod_book, subadj_book, all_nodes, node_feat_book, edge_idx_book = get_subadj_book(window_idx, self.adjacency_matrix, self.df_agg, self.df_window)
        aligned_adj_book, node_index_map, aligned_node_feat_book, aligned_edge_index_book = get_aligned_adj_book(subadj_book, node_feat_book, actnod_book, all_nodes)

        if b not in self.all_nodes_dict:
            self.all_nodes_dict[b] = {}
        self.all_nodes_dict[b] = all_nodes

        # Initialize graph data for batch if not present
        if b not in self.graph_data:
            self.graph_data[b] = {}


        for i, idx in enumerate(window_idx):

            aligned_node_feat_book[i], aligned_edge_index_book[i], idx

            edge_index, edge_weight = gcn_norm(edge_index=aligned_edge_index_book[i].long(),
                                           edge_weight=None,
                                           num_nodes=aligned_node_feat_book[i].shape[0],
                                           add_self_loops=False)

            # Create a Data object with node features, edge index, and edge weights
            data = Data(x=torch.tensor(aligned_node_feat_book[i]).float(), edge_index=edge_index.float(), edge_weight = edge_weight)

            self.graph_data[b][idx] = data

    def get_batch_instance(self, b):
        return self.graph_data.get(b, "Batch not found.")

    def get_batch_time_instance(self, b, idx):
        return self.graph_data.get(b, {}).get(idx, "Time instance not found.")

    def get_all_nodes(self, b):
        return self.all_nodes_dict[b]


def create_adjacency_matrix(lclids, k):
    number_of_nodes = sum([len(l) for l in lclids])
    adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))
    # Create the graph by iterating over the list of lists of LCLids
    # and connecting all nodes in the list with each other
    # and with the k-nearest lists
    for i in range(len(lclids)): # range 2156
        for j in range(len(lclids)): # range 2156
            if i == j:
                for lclid in lclids[i]:
                    for lclid2 in lclids[j]:
                        adjacency_matrix[lclid, lclid2] = 1
            elif abs(i-j) <= k:
                for lclid in lclids[i]:
                    for lclid2 in lclids[j]:
                        adjacency_matrix[lclid, lclid2] = 1
    adjacency_matrix = adjacency_matrix - np.eye(number_of_nodes)
    return sp.sparse.bsr_array(adjacency_matrix)


def blocked_cross_validation_idx(df,train_months=3,test_months=1,overlap_months=0):
    """
    return: list of tuples containing (train_start_idx, train_end_idx, test_start_idx, test_end_idx) for year 2012-2013
    """
    blocks=[]
    start_date=df['DateTime'].iloc[0]
    end_date=df['DateTime'].iloc[-1]
    current_train_start=start_date
    while current_train_start+pd.DateOffset(months=train_months+test_months)<=end_date+pd.Timedelta(hours=1):
        train_end=current_train_start+pd.DateOffset(months=train_months)-pd.Timedelta(hours=1)
        test_start=train_end+pd.Timedelta(hours=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(hours=1)
        train_start_idx=df.index[df['DateTime'] == current_train_start][0]
        train_end_idx=df.index[df['DateTime'] == train_end][0]
        test_end_idx=df.index[df['DateTime'] == test_end][0]
        test_start_idx=df.index[df['DateTime'] == test_start][0]
        blocks.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))
        current_train_start = current_train_start + pd.DateOffset(months=train_months + test_months - overlap_months)
    return blocks


def get_windows_idx(start,end,in_window,out_window):
    #timestamps=[]
    in_idx=[]
    out_idx=[]
    current=start
    while current+in_window+out_window<=end+1:
        in_idx.append(np.arange(start=current,stop=current+in_window))
        out_idx.append(np.arange(start=current+in_window,stop=current+in_window+out_window))
        #out_idx.append(pd.date_range(start=current+pd.Timedelta(hours=in_window),periods=out_window,freq='h'))
        current+=1
    return in_idx,out_idx


def get_train_batch_blocks(start,end,batch_size=64):
    blocks=[]
    curr_start=start
    while curr_start<end:
        curr_end=min(curr_start+batch_size,end)
        blocks.append(np.arange(curr_start,curr_end))
        curr_start+=batch_size
    return blocks