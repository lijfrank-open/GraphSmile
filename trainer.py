import numpy as np, random
import torch
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from module import build_match_sen_shift_label
from utils import AutomaticWeightedLoss

seed = 2024


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_or_eval_model(
    model,
    loss_function_emo,
    loss_function_sen,
    loss_function_shift,
    dataloader,
    epoch,
    cuda,
    modals,
    optimizer=None,
    train=False,
    dataset="IEMOCAP",
    loss_type="",
    lambd=[1.0, 1.0, 1.0],
    epochs=100,
    classify="",
    shift_win=5,
):
    losses, preds_emo, labels_emo = [], [], []
    preds_sft, labels_sft = [], []
    preds_sen, labels_sen = [], []
    vids = []
    initial_feats, extracted_feats = [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for iter, data in enumerate(dataloader):

        if train:
            optimizer.zero_grad()

        textf0, textf1, textf2, textf3, visuf, acouf, qmask, umask, label_emotion, label_sentiment = (
            [d.cuda() for d in data[:-1]] if cuda else data[:-1])

        dia_lengths, label_emotions, label_sentiments = [], [], []
        for j in range(umask.size(1)):
            dia_lengths.append((umask[:, j] == 1).nonzero().tolist()[-1][0] +
                               1)
            label_emotions.append(label_emotion[:dia_lengths[j], j])
            label_sentiments.append(label_sentiment[:dia_lengths[j], j])
        label_emo = torch.cat(label_emotions)
        label_sen = torch.cat(label_sentiments)

        logit_emo, logit_sen, logit_sft, extracted_feature = model(
            textf0, textf1, textf2, textf3, visuf, acouf, umask, qmask,
            dia_lengths)

        prob_emo = F.log_softmax(logit_emo, -1)
        loss_emo = loss_function_emo(prob_emo, label_emo)
        prob_sen = F.log_softmax(logit_sen, -1)
        loss_sen = loss_function_sen(prob_sen, label_sen)
        prob_sft = F.log_softmax(logit_sft, -1)
        label_sft = build_match_sen_shift_label(shift_win, dia_lengths,
                                                label_sen)
        loss_sft = loss_function_shift(prob_sft, label_sft)

        if loss_type == "auto":
            awl = AutomaticWeightedLoss(3)
            loss = awl(loss_emo, loss_sen, loss_sft)
        elif loss_type == "epoch":
            loss = (epoch / epochs) * (lambd[0] * loss_emo) + (
                1 - epoch / epochs) * (lambd[1] * loss_sen +
                                       lambd[2] * loss_sft)
        elif loss_type == "emo_sen_sft":
            loss = lambd[0] * loss_emo + lambd[1] * loss_sen + lambd[
                2] * loss_sft
        elif loss_type == "emo_sen":
            loss = lambd[0] * loss_emo + lambd[1] * loss_sen
        elif loss_type == "emo_sft":
            loss = lambd[0] * loss_emo + lambd[2] * loss_sft
        elif loss_type == "emo":
            loss = loss_emo
        elif loss_type == "sen_sft":
            loss = lambd[1] * loss_sen + lambd[2] * loss_sft
        elif loss_type == "sen":
            loss = loss_sen
        else:
            NotImplementedError

        preds_emo.append(torch.argmax(prob_emo, 1).cpu().numpy())
        labels_emo.append(label_emo.cpu().numpy())
        preds_sen.append(torch.argmax(prob_sen, 1).cpu().numpy())
        labels_sen.append(label_sen.cpu().numpy())
        preds_sft.append(torch.argmax(prob_sft, 1).cpu().numpy())
        labels_sft.append(label_sft.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()

        extracted_feats.append(extracted_feature.cpu().detach().numpy())

    if preds_emo != []:
        preds_emo = np.concatenate(preds_emo)
        labels_emo = np.concatenate(labels_emo)
        preds_sen = np.concatenate(preds_sen)
        labels_sen = np.concatenate(labels_sen)
        preds_sft = np.concatenate(preds_sft)
        labels_sft = np.concatenate(labels_sft)

        extracted_feats = np.concatenate(extracted_feats)

    vids += data[-1]
    labels_emo = np.array(labels_emo)
    preds_emo = np.array(preds_emo)
    labels_sen = np.array(labels_sen)
    preds_sen = np.array(preds_sen)
    labels_sft = np.array(labels_sft)
    preds_sft = np.array(preds_sft)
    vids = np.array(vids)

    extracted_feats = np.array(extracted_feats)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_acc_emo = round(accuracy_score(labels_emo, preds_emo) * 100, 2)
    avg_f1_emo = round(
        f1_score(labels_emo, preds_emo, average="weighted") * 100, 2)
    avg_acc_sen = round(accuracy_score(labels_sen, preds_sen) * 100, 2)
    avg_f1_sen = round(
        f1_score(labels_sen, preds_sen, average="weighted") * 100, 2)
    avg_acc_sft = round(accuracy_score(labels_sft, preds_sft) * 100, 2)
    avg_f1_sft = round(
        f1_score(labels_sft, preds_sft, average="weighted") * 100, 2)

    return avg_loss, labels_emo, preds_emo, avg_acc_emo, avg_f1_emo, labels_sen, preds_sen, avg_acc_sen, avg_f1_sen, avg_acc_sft, avg_f1_sft, vids, initial_feats, extracted_feats
