import logging
import os
import numpy as np
import pickle as pk
import datetime
import torch.nn as nn
import torch.optim as optim
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import time
from utils import AutomaticWeightedLoss
from model import GraphSmile
from sklearn.metrics import confusion_matrix, classification_report
from trainer import train_or_eval_model, seed_everything
from dataloader import (
    IEMOCAPDataset_BERT,
    IEMOCAPDataset_BERT4,
    MELDDataset_BERT,
    CMUMOSEIDataset7,
)
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--no_cuda",
                    action="store_true",
                    default=False,
                    help="does not use GPU")
parser.add_argument("--gpu", default="2", type=str, help="GPU ids")
parser.add_argument("--port", default="15301", help="MASTER_PORT")
parser.add_argument("--classify", default="emotion", help="sentiment, emotion")
parser.add_argument("--lr",
                    type=float,
                    default=0.00001,
                    metavar="LR",
                    help="learning rate")
parser.add_argument("--l2",
                    type=float,
                    default=0.0001,
                    metavar="L2",
                    help="L2 regularization weight")
parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    metavar="BS",
                    help="batch size")
parser.add_argument("--epochs",
                    type=int,
                    default=100,
                    metavar="E",
                    help="number of epochs")
parser.add_argument("--tensorboard",
                    action="store_true",
                    default=False,
                    help="Enables tensorboard log")
parser.add_argument("--modals", default="avl", help="modals")
parser.add_argument(
    "--dataset",
    default="IEMOCAP",
    help="dataset to train and test.MELD/IEMOCAP/IEMOCAP4/CMUMOSEI7",
)
parser.add_argument(
    "--textf_mode",
    default="textf0",
    help="concat4/concat2/textf0/textf1/textf2/textf3/sum2/sum4",
)

parser.add_argument(
    "--conv_fpo",
    nargs="+",
    type=int,
    default=[3, 1, 1],
    help="n_filter,n_padding; n_out = (n_in + 2*n_padding -n_filter)/stride +1",
)

parser.add_argument("--hidden_dim", type=int, default=256, help="hidden_dim")
parser.add_argument(
    "--win",
    nargs="+",
    type=int,
    default=[17, 17],
    help="[win_p, win_f], -1 denotes all nodes",
)
parser.add_argument("--heter_n_layers",
                    nargs="+",
                    type=int,
                    default=[6, 6, 6],
                    help="heter_n_layers")

parser.add_argument("--drop",
                    type=float,
                    default=0.3,
                    metavar="dropout",
                    help="dropout rate")

parser.add_argument("--shift_win",
                    type=int,
                    default=12,
                    help="windows of sentiment shift")

parser.add_argument(
    "--loss_type",
    default="emo_sen_sft",
    help="auto/epoch/emo_sen_sft/emo_sen/emo_sft/emo/sen_sft/sen",
)
parser.add_argument(
    "--lambd",
    nargs="+",
    type=float,
    default=[1.0, 1.0, 1.0],
    help="[loss_emotion, loss_sentiment, loss_shift]",
)

args = parser.parse_args()

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = args.port
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
world_size = torch.cuda.device_count()
os.environ["WORLD_SIZE"] = str(world_size)

MELD_path = ""
IEMOCAP_path = ""
IEMOCAP4_path = ""
CMUMOSEI7_path = ""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_ddp(local_rank):
    try:
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            os.environ["RANK"] = str(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            logger.info("Distributed process group already initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize distributed process group: {e}")
        raise


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()

    return rt


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)

    return g


def get_train_valid_sampler(trainset, valid_ratio):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid_ratio * size)

    return DistributedSampler(idx[split:]), DistributedSampler(idx[:split])


def get_data_loaders(path, dataset_class, batch_size, valid_ratio, num_workers,
                     pin_memory):

    trainset = dataset_class(path)
    train_sampler, valid_sampler = get_train_valid_sampler(
        trainset, valid_ratio)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testset = dataset_class(path, train=False)
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


def setup_samplers(trainset, valid_ratio, epoch):
    train_sampler, valid_sampler = get_train_valid_sampler(
        trainset, valid_ratio=valid_ratio)
    train_sampler.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)


def main(local_rank):
    print(f"Running main(**args) on rank {local_rank}.")
    init_ddp(local_rank)  # 初始化

    today = datetime.datetime.now()
    name_ = args.modals + "_" + args.dataset

    cuda = torch.cuda.is_available() and not args.no_cuda
    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals

    if args.dataset == "IEMOCAP":
        embedding_dims = [1024, 342, 1582]
    elif args.dataset == "IEMOCAP4":
        embedding_dims = [1024, 512, 100]
    elif args.dataset == "MELD":
        embedding_dims = [1024, 342, 300]
    elif args.dataset == "CMUMOSEI7":
        embedding_dims = [1024, 35, 384]

    if args.dataset == "MELD" or args.dataset == "CMUMOSEI7":
        n_classes_emo = 7
    elif args.dataset == "IEMOCAP":
        n_classes_emo = 6
    elif args.dataset == "IEMOCAP4":
        n_classes_emo = 4

    seed_everything()
    model = GraphSmile(args, embedding_dims, n_classes_emo)

    model = model.to(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    loss_function_emo = nn.NLLLoss()
    loss_function_sen = nn.NLLLoss()
    loss_function_shift = nn.NLLLoss()

    if args.loss_type == "auto_loss":
        awl = AutomaticWeightedLoss(3)
        optimizer = optim.AdamW(
            [
                {
                    "params": model.parameters()
                },
                {
                    "params": awl.parameters(),
                    "weight_decay": 0
                },
            ],
            lr=args.lr,
            weight_decay=args.l2,
            amsgrad=True,
        )
    else:
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.l2,
                                amsgrad=True)

    if args.dataset == "MELD":
        train_loader, valid_loader, test_loader = get_data_loaders(
            path=MELD_path,
            dataset_class=MELDDataset_BERT,
            valid_ratio=0.1,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )
    elif args.dataset == "IEMOCAP":
        train_loader, valid_loader, test_loader = get_data_loaders(
            path=IEMOCAP_path,
            dataset_class=IEMOCAPDataset_BERT,
            valid_ratio=0.1,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )
    elif args.dataset == "IEMOCAP4":
        train_loader, valid_loader, test_loader = get_data_loaders(
            path=IEMOCAP4_path,
            dataset_class=IEMOCAPDataset_BERT4,
            valid_ratio=0.1,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )
    elif args.dataset == "CMUMOSEI7":
        train_loader, valid_loader, test_loader = get_data_loaders(
            path=CMUMOSEI7_path,
            dataset_class=CMUMOSEIDataset7,
            valid_ratio=0.1,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )
    else:
        print("There is no such dataset")

    best_f1_emo, best_f1_sen, best_loss = None, None, None
    best_label_emo, best_pred_emo = None, None
    best_label_sen, best_pred_sen = None, None
    best_extracted_feats = None
    all_f1_emo, all_acc_emo, all_loss = [], [], []
    all_f1_sen, all_acc_sen = [], []
    all_f1_sft, all_acc_sft = [], []

    for epoch in range(n_epochs):
        if args.dataset == "MELD":
            trainset = MELDDataset_BERT(MELD_path)
        elif args.dataset == "IEMOCAP":
            trainset = IEMOCAPDataset_BERT(IEMOCAP_path)
        elif args.dataset == "IEMOCAP4":
            trainset = IEMOCAPDataset_BERT4(IEMOCAP4_path)
        elif args.dataset == "CMUMOSEI7":
            trainset = CMUMOSEIDataset7(CMUMOSEI7_path)

        setup_samplers(trainset, valid_ratio=0.1, epoch=epoch)

        start_time = time.time()

        train_loss, _, _, train_acc_emo, train_f1_emo, _, _, train_acc_sen, train_f1_sen, train_acc_sft, train_f1_sft, _, _, _ = train_or_eval_model(
            model,
            loss_function_emo,
            loss_function_sen,
            loss_function_shift,
            train_loader,
            epoch,
            cuda,
            args.modals,
            optimizer,
            True,
            args.dataset,
            args.loss_type,
            args.lambd,
            args.epochs,
            args.classify,
            args.shift_win,
        )

        valid_loss, _, _, valid_acc_emo, valid_f1_emo, _, _, valid_acc_sen, valid_f1_sen, valid_acc_sft, valid_f1_sft, _, _, _ = train_or_eval_model(
            model,
            loss_function_emo,
            loss_function_sen,
            loss_function_shift,
            valid_loader,
            epoch,
            cuda,
            args.modals,
            None,
            False,
            args.dataset,
            args.loss_type,
            args.lambd,
            args.epochs,
            args.classify,
            args.shift_win,
        )

        print(
            "epoch: {}, train_loss: {}, train_acc_emo: {}, train_f1_emo: {}, valid_loss: {}, valid_acc_emo: {}, valid_f1_emo: {}"
            .format(
                epoch + 1,
                train_loss,
                train_acc_emo,
                train_f1_emo,
                valid_loss,
                valid_acc_emo,
                valid_f1_emo,
            ))

        if local_rank == 0:
            test_loss, test_label_emo, test_pred_emo, test_acc_emo, test_f1_emo, test_label_sen, test_pred_sen, test_acc_sen, test_f1_sen, test_acc_sft, test_f1_sft, _, test_initial_feats, test_extracted_feats = train_or_eval_model(
                model,
                loss_function_emo,
                loss_function_sen,
                loss_function_shift,
                test_loader,
                epoch,
                cuda,
                args.modals,
                None,
                False,
                args.dataset,
                args.loss_type,
                args.lambd,
                args.epochs,
                args.classify,
                args.shift_win,
            )

            all_f1_emo.append(test_f1_emo)
            all_acc_emo.append(test_acc_emo)
            all_f1_sft.append(test_f1_sft)
            all_acc_sft.append(test_acc_sft)

            print(
                "test_loss: {}, test_acc_emo: {}, test_f1_emo: {}, test_acc_sen: {}, test_f1_sen: {}, test_acc_sft: {}, test_f1_sft: {}, total time: {} sec, {}"
                .format(
                    test_loss,
                    test_acc_emo,
                    test_f1_emo,
                    test_acc_sen,
                    test_f1_sen,
                    test_acc_sft,
                    test_f1_sft,
                    round(time.time() - start_time, 2),
                    time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(time.time())),
                ))
            print("-" * 100)

            if args.classify == "emotion":
                if best_f1_emo == None or best_f1_emo < test_f1_emo:
                    best_f1_emo = test_f1_emo
                    best_f1_sen = test_f1_sen
                    best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                    best_label_sen, best_pred_sen = test_label_sen, test_pred_sen

            elif args.classify == "sentiment":
                if best_f1_sen == None or best_f1_sen < test_f1_sen:
                    best_f1_emo = test_f1_emo
                    best_f1_sen = test_f1_sen
                    best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                    best_label_sen, best_pred_sen = test_label_sen, test_pred_sen

            if (epoch + 1) % 10 == 0:
                np.set_printoptions(suppress=True)
                print(
                    classification_report(best_label_emo,
                                          best_pred_emo,
                                          digits=4,
                                          zero_division=0))
                print(confusion_matrix(best_label_emo, best_pred_emo))
                print(
                    classification_report(best_label_sen,
                                          best_pred_sen,
                                          digits=4,
                                          zero_division=0))
                print(confusion_matrix(best_label_sen, best_pred_sen))
                print("-" * 100)

        dist.barrier()

        if args.tensorboard:
            writer.add_scalar("test: accuracy", test_acc_emo, epoch)
            writer.add_scalar("test: fscore", test_f1_emo, epoch)
            writer.add_scalar("train: accuracy", train_acc_emo, epoch)
            writer.add_scalar("train: fscore", train_f1_emo, epoch)

        if epoch == 1:
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            print(f"Allocated Memory: {allocated_memory / 1024**2:.2f} MB")
            print(f"Reserved Memory: {reserved_memory / 1024**2:.2f} MB")
            print(
                f"All Memory: {(allocated_memory + reserved_memory) / 1024**2:.2f} MB"
            )

    if args.tensorboard:
        writer.close()
    if local_rank == 0:
        print("Test performance..")
        print("Acc: {}, F-Score: {}".format(max(all_acc_emo), max(all_f1_emo)))
        if not os.path.exists("results/record_{}_{}_{}.pk".format(
                today.year, today.month, today.day)):
            with open(
                    "results/record_{}_{}_{}.pk".format(
                        today.year, today.month, today.day),
                    "wb",
            ) as f:
                pk.dump({}, f)
        with open(
                "results/record_{}_{}_{}.pk".format(today.year, today.month,
                                                    today.day),
                "rb",
        ) as f:
            record = pk.load(f)
        key_ = name_
        if record.get(key_, False):
            record[key_].append(max(all_f1_emo))
        else:
            record[key_] = [max(all_f1_emo)]
        if record.get(key_ + "record", False):
            record[key_ + "record"].append(
                classification_report(best_label_emo,
                                      best_pred_emo,
                                      digits=4,
                                      zero_division=0))
        else:
            record[key_ + "record"] = [
                classification_report(best_label_emo,
                                      best_pred_emo,
                                      digits=4,
                                      zero_division=0)
            ]
        with open(
                "results/record_{}_{}_{}.pk".format(today.year, today.month,
                                                    today.day),
                "wb",
        ) as f:
            pk.dump(record, f)

        print(
            classification_report(best_label_emo,
                                  best_pred_emo,
                                  digits=4,
                                  zero_division=0))
        print(confusion_matrix(best_label_emo, best_pred_emo))

    dist.destroy_process_group()


if __name__ == "__main__":
    print(args)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("not args.no_cuda:", not args.no_cuda)
    n_gpus = torch.cuda.device_count()
    print(f"Use {n_gpus} GPUs")
    mp.spawn(fn=main, args=(), nprocs=n_gpus)
