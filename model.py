from module import HeterGConv_Edge, HeterGConvLayer, SenShift_Feat
import torch.nn as nn
import torch
from utils import batch_to_all_tva


class GraphSmile(nn.Module):

    def __init__(self, args, embedding_dims, n_classes_emo):
        super(GraphSmile, self).__init__()
        self.textf_mode = args.textf_mode
        self.no_cuda = args.no_cuda
        self.win_p = args.win[0]
        self.win_f = args.win[1]
        self.modals = args.modals
        self.shift_win = args.shift_win

        self.batchnorms_t = nn.ModuleList(
            nn.BatchNorm1d(embedding_dims[0]) for _ in range(4))

        in_dims_t = (4 * embedding_dims[0] if args.textf_mode == "concat4" else
                     (2 * embedding_dims[0]
                      if args.textf_mode == "concat2" else embedding_dims[0]))
        self.dim_layer_t = nn.Sequential(nn.Linear(in_dims_t, args.hidden_dim),
                                         nn.LeakyReLU(), nn.Dropout(args.drop))
        self.dim_layer_v = nn.Sequential(
            nn.Linear(embedding_dims[1], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )
        self.dim_layer_a = nn.Sequential(
            nn.Linear(embedding_dims[2], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )

        # Heter
        hetergconvLayer_tv = HeterGConvLayer(args.hidden_dim, args.drop,
                                             args.no_cuda)
        self.hetergconv_tv = HeterGConv_Edge(
            args.hidden_dim,
            hetergconvLayer_tv,
            args.heter_n_layers[0],
            args.drop,
            args.no_cuda,
        )
        hetergconvLayer_ta = HeterGConvLayer(args.hidden_dim, args.drop,
                                             args.no_cuda)
        self.hetergconv_ta = HeterGConv_Edge(
            args.hidden_dim,
            hetergconvLayer_ta,
            args.heter_n_layers[1],
            args.drop,
            args.no_cuda,
        )
        hetergconvLayer_va = HeterGConvLayer(args.hidden_dim, args.drop,
                                             args.no_cuda)
        self.hetergconv_va = HeterGConv_Edge(
            args.hidden_dim,
            hetergconvLayer_va,
            args.heter_n_layers[2],
            args.drop,
            args.no_cuda,
        )

        self.modal_fusion = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LeakyReLU(),
        )

        self.emo_output = nn.Linear(args.hidden_dim, n_classes_emo)
        self.sen_output = nn.Linear(args.hidden_dim, 3)
        self.senshift = SenShift_Feat(args.hidden_dim, args.drop,
                                      args.shift_win)

    def forward(self, feature_t0, feature_t1, feature_t2, feature_t3,
                feature_v, feature_a, umask, qmask, dia_lengths):

        (
            (seq_len_t, batch_size_t, feature_dim_t),
            (seq_len_v, batch_size_v, feature_dim_v),
            (seq_len_a, batch_size_a, feature_dim_a),
        ) = [feature_t0.shape, feature_v.shape, feature_a.shape]
        features_t = [
            batchnorm_t(feature_t.transpose(0, 1).reshape(
                -1, feature_dim_t)).reshape(-1, seq_len_t,
                                            feature_dim_t).transpose(1, 0)
            for batchnorm_t, feature_t in
            zip(self.batchnorms_t,
                [feature_t0, feature_t1, feature_t2, feature_t3])
        ]
        feature_t0, feature_t1, feature_t2, feature_t3 = features_t

        dim_layer_dict_t = {
            "concat4":
            lambda: self.dim_layer_t(
                torch.cat([feature_t0, feature_t1, feature_t2, feature_t3],
                          dim=-1)),
            "sum4":
            lambda:
            (self.dim_layer_t(feature_t0) + self.dim_layer_t(feature_t1) + self
             .dim_layer_t(feature_t2) + self.dim_layer_t(feature_t3)) / 4,
            "concat2":
            lambda: self.dim_layer_t(
                torch.cat([feature_t0, feature_t1], dim=-1)),
            "sum2":
            lambda:
            (self.dim_layer_t(feature_t0) + self.dim_layer_t(feature_t1)) / 2,
            "textf0":
            lambda: self.dim_layer_t(feature_t0),
            "textf1":
            lambda: self.dim_layer_t(feature_t1),
            "textf2":
            lambda: self.dim_layer_t(feature_t2),
            "textf3":
            lambda: self.dim_layer_t(feature_t3),
        }
        featdim_t = dim_layer_dict_t[self.textf_mode]()
        featdim_v, featdim_a = self.dim_layer_v(feature_v), self.dim_layer_a(
            feature_a)

        emo_t, emo_v, emo_a = featdim_t, featdim_v, featdim_a

        emo_t, emo_v, emo_a = batch_to_all_tva(emo_t, emo_v, emo_a,
                                               dia_lengths, self.no_cuda)

        featheter_tv, heter_edge_index = self.hetergconv_tv(
            (emo_t, emo_v), dia_lengths, self.win_p, self.win_f)
        featheter_ta, heter_edge_index = self.hetergconv_ta(
            (emo_t, emo_a), dia_lengths, self.win_p, self.win_f,
            heter_edge_index)
        featheter_va, heter_edge_index = self.hetergconv_va(
            (emo_v, emo_a), dia_lengths, self.win_p, self.win_f,
            heter_edge_index)

        feat_fusion = (self.modal_fusion(featheter_tv[0]) + self.modal_fusion(
            featheter_ta[0]) + self.modal_fusion(featheter_tv[1]) +
                       self.modal_fusion(featheter_va[0]) +
                       self.modal_fusion(featheter_ta[1]) +
                       self.modal_fusion(featheter_va[1])) / 6

        logit_emo = self.emo_output(feat_fusion)
        logit_sen = self.sen_output(feat_fusion)

        logit_shift = self.senshift(feat_fusion, feat_fusion, dia_lengths)

        return logit_emo, logit_sen, logit_shift, feat_fusion
