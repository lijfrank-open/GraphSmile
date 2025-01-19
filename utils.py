import torch
import torch.nn as nn


def batch_to_all_tva(feature_t, feature_v, feature_a, lengths, no_cuda):

    node_feature_t, node_feature_v, node_feature_a = [], [], []
    batch_size = feature_t.size(1)

    for j in range(batch_size):
        node_feature_t.append(feature_t[:lengths[j], j, :])
        node_feature_v.append(feature_v[:lengths[j], j, :])
        node_feature_a.append(feature_a[:lengths[j], j, :])

    node_feature_t = torch.cat(node_feature_t, dim=0)
    node_feature_v = torch.cat(node_feature_v, dim=0)
    node_feature_a = torch.cat(node_feature_a, dim=0)

    if not no_cuda:
        node_feature_t = node_feature_t.cuda()
        node_feature_v = node_feature_v.cuda()
        node_feature_a = node_feature_a.cuda()

    return node_feature_t, node_feature_v, node_feature_a


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i]**
                               2) * loss + torch.log(1 + self.params[i]**2)
        return loss_sum
