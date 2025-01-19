from itertools import permutations, product
import math
import torch
import copy
import torch.nn as nn
from torch.nn import Parameter


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class HeterGConv_Edge(torch.nn.Module):

    def __init__(self, feature_size, encoder_layer, num_layers, dropout,
                 no_cuda):
        super(HeterGConv_Edge, self).__init__()
        self.num_layers = num_layers
        self.no_cuda = no_cuda

        self.edge_weight = nn.Parameter(torch.ones(500000))

        self.hetergcn_layers = _get_clones(encoder_layer, num_layers)
        self.fc_layer = nn.Sequential(nn.Linear(feature_size, feature_size),
                                      nn.LeakyReLU(), nn.Dropout(dropout))
        self.fc_layers = _get_clones(self.fc_layer, num_layers)

    def forward(self, feature_tuple, dia_lens, win_p, win_f, edge_index=None):

        num_modal = len(feature_tuple)
        feature = torch.cat(feature_tuple, dim=0)

        if edge_index is None:
            edge_index = self._heter_no_weight_edge(feature, num_modal,
                                                    dia_lens, win_p, win_f)
        edge_weight = self.edge_weight[0:edge_index.size(1)]

        adj_weight = self._edge_index_to_adjacency_matrix(
            edge_index,
            edge_weight,
            num_nodes=feature.size(0),
            no_cuda=self.no_cuda)
        feature_sum = feature
        for i in range(self.num_layers):
            feature = self.hetergcn_layers[i](feature, num_modal, adj_weight)
            feature_sum = feature_sum + self.fc_layers[i](feature)
        feat_tuple = torch.chunk(feature_sum, num_modal, dim=0)

        return feat_tuple, edge_index

    def _edge_index_to_adjacency_matrix(self,
                                        edge_index,
                                        edge_weight=None,
                                        num_nodes=100,
                                        no_cuda=False):

        if edge_weight is not None:
            edge_weight = edge_weight.squeeze()
        else:
            edge_weight = torch.ones(
                edge_index.size(1)).cuda() if not no_cuda else torch.ones(
                    edge_index.size(1))
        adj_sparse = torch.sparse_coo_tensor(edge_index,
                                             edge_weight,
                                             size=(num_nodes, num_nodes))
        adj = adj_sparse.to_dense()
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[d_inv_sqrt == float("inf")] = 0
        d_inv_sqrt_mat = torch.diag_embed(d_inv_sqrt)
        gcn_fact = torch.matmul(d_inv_sqrt_mat,
                                torch.matmul(adj, d_inv_sqrt_mat))

        if not no_cuda and torch.cuda.is_available():
            gcn_fact = gcn_fact.cuda()

        return gcn_fact

    def _heter_no_weight_edge(self, feature, num_modal, dia_lens, win_p,
                              win_f):
        index_inter = []
        all_dia_len = sum(dia_lens)
        all_nodes = list(range(all_dia_len * num_modal))
        nodes_uni = [None] * num_modal

        for m in range(num_modal):
            nodes_uni[m] = all_nodes[m * all_dia_len:(m + 1) * all_dia_len]

        start = 0
        for dia_len in dia_lens:
            for m, n in permutations(range(num_modal), 2):

                for j, node_m in enumerate(nodes_uni[m][start:start +
                                                        dia_len]):
                    if win_p == -1 and win_f == -1:
                        nodes_n = nodes_uni[n][start:start + dia_len]
                    elif win_p == -1:
                        nodes_n = nodes_uni[n][
                            start:min(start + dia_len, start + j + win_f + 1)]
                    elif win_f == -1:
                        nodes_n = nodes_uni[n][max(start, start + j -
                                                   win_p):start + dia_len]
                    else:
                        nodes_n = nodes_uni[n][
                            max(start, start + j -
                                win_p):min(start + dia_len, start + j + win_f +
                                           1)]
                    index_inter.extend(list(product([node_m], nodes_n)))
            start += dia_len
        edge_index = (torch.tensor(index_inter).permute(1, 0).cuda() if
                      not self.no_cuda else torch.tensor(index_inter).permute(
                          1, 0))

        return edge_index


class HeterGConvLayer(torch.nn.Module):

    def __init__(self, feature_size, dropout=0.3, no_cuda=False):
        super(HeterGConvLayer, self).__init__()
        self.no_cuda = no_cuda
        self.hetergconv = SGConv_Our(feature_size, feature_size)

    def forward(self, feature, num_modal, adj_weight):

        if num_modal > 1:
            feature_heter = self.hetergconv(feature, adj_weight)
        else:
            print("Unable to construct heterogeneous graph!")
            feature_heter = feature

        return feature_heter


class SGConv_Our(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SGConv_Our, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SenShift_Feat(nn.Module):

    def __init__(self, hidden_dim, dropout, shift_win):
        super().__init__()

        self.shift_win = shift_win

        hidden_dim_shift = 2 * hidden_dim

        self.shift_output_layer = nn.Sequential(nn.Linear(hidden_dim_shift,
                                                          2), )

    def forward(self, embeds, embeds_temp=None, dia_lens=[]):

        if embeds_temp == None:
            embeds_temp = embeds
        embeds_shift = self._build_match_sample(embeds, embeds_temp, dia_lens,
                                                self.shift_win)
        logits = self.shift_output_layer(embeds_shift)

        return logits

    def _build_match_sample(self, embeds, embeds_temp, dia_lens, shift_win):

        start = 0
        embeds_shifts = []
        if shift_win == -1:
            for dia_len in dia_lens:
                embeds_shifts.append(
                    torch.cat(
                        [
                            embeds[start:start + dia_len, None, :].repeat(
                                1, dia_len, 1),
                            embeds_temp[None, start:start + dia_len, :].repeat(
                                dia_len, 1, 1),
                        ],
                        dim=-1,
                    ).view(-1, 2 * embeds.size(-1)))
                start += dia_len
            embeds_shift = torch.cat(embeds_shifts, dim=0)

        elif shift_win > 0:
            for dia_len in dia_lens:
                win_start = 0
                for i in range(math.ceil(dia_len / shift_win)):
                    if (i == math.ceil(dia_len / shift_win) - 1
                            and dia_len % shift_win != 0):
                        win = dia_len % shift_win
                    else:
                        win = shift_win
                    embeds_shifts.append(
                        torch.cat(
                            [
                                embeds[
                                    start + win_start : start + win_start + win, None, :
                                ].repeat(1, win, 1),
                                embeds_temp[
                                    None, start + win_start : start + win_start + win, :
                                ].repeat(win, 1, 1),
                            ],
                            dim=-1,
                        ).view(-1, 2 * embeds.size(-1))
                    )
                    win_start += shift_win
                start += dia_len
            embeds_shift = torch.cat(embeds_shifts, dim=0)
        else:
            print("Window must be greater than 0 or equal to -1")
            raise NotImplementedError

        return embeds_shift


def build_match_sen_shift_label(shift_win, dia_lengths, label_sen):
    start = 0
    label_shifts = []
    if shift_win == -1:
        for dia_len in dia_lengths:
            dia_label_shift = ((label_sen[start:start + dia_len, None]
                                != label_sen[None, start:start +
                                             dia_len]).long().view(-1))
            label_shifts.append(dia_label_shift)
            start += dia_len
        label_shift = torch.cat(label_shifts, dim=0)
    elif shift_win > 0:
        for dia_len in dia_lengths:
            win_start = 0
            for i in range(math.ceil(dia_len / shift_win)):
                if i == math.ceil(
                        dia_len / shift_win) - 1 and dia_len % shift_win != 0:
                    win = dia_len % shift_win
                else:
                    win = shift_win
                dia_label_shift = (
                    (
                        label_sen[start + win_start : start + win_start + win, None]
                        != label_sen[None, start + win_start : start + win_start + win]
                    )
                    .long()
                    .view(-1)
                )
                label_shifts.append(dia_label_shift)
                win_start += shift_win
            start += dia_len
        label_shift = torch.cat(label_shifts, dim=0)
    else:
        print("Window must be greater than 0 or equal to -1")
        raise NotImplementedError

    return label_shift
