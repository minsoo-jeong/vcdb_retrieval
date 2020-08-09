import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as trn
from collections import defaultdict

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from tensorboardX import SummaryWriter
from model.nets import MobileNet_RMAC, TripletNet
from model.loss import TripletLoss
from dataset.loader import TripletDataset, PairDataset, ListDataset
from dataset.autoaugment import ImageNetPolicy
from utils.Measure import AverageMeter
import warnings
from tqdm import tqdm
import faiss
import logging
import pprint

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)
kst = timezone(timedelta(hours=9))


def read_triplets(csv_path):
    df = pd.read_csv(csv_path)[['anc', 'anc_frame', 'pos', 'pos_frame', 'neg', 'neg_frame']].to_numpy()
    return df


def read_positive_csv(csv_path):
    df = pd.read_csv(csv_path)[['a', 'a_frame', 'b', 'b_frame']].to_numpy()
    return df


def init_logger(comment=''):
    current = datetime.now(kst)
    current_date = current.strftime('%m%d')
    current_time = current.strftime('%H%M%S')
    basename = comment if comment is not None and comment != '' else current_time

    def timetz(*args):
        return datetime.now(kst).timetuple()

    logging.Formatter.converter = timetz
    log_dir = f'/hdd/ms/vcdb_retrieval_ckpt/{current_date}/{basename}'
    if os.path.exists(log_dir):
        log_dir = f'/hdd/ms/vcdb_retrieval_ckpt/{current_date}/{basename}_{current_time}'

    ckpt_dir = f'{log_dir}/saved_model'
    os.makedirs(ckpt_dir)
    global writer
    writer = SummaryWriter(logdir=log_dir)
    global logger
    logger = logging.getLogger(current_time)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)  # console +file

    file_handler = logging.FileHandler(filename=f"{log_dir}/log.txt", mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # file

    logger.addHandler(console)
    logger.addHandler(file_handler)

    import socket
    logger.info("=========================================================")
    logger.info(f'Start - {socket.gethostname()}')
    logger.info(f'Log directory ... {log_dir}')
    logger.info("=========================================================")


def scan_vcdb_annotation(root):
    def parse(ann):
        a, b, *times = ann.strip().split(',')
        times = [sum([60 ** (2 - n) * int(u) for n, u in enumerate(t.split(':'))]) for t in times]
        return [a, b, *times]

    groups = os.listdir(root)
    annotations = []
    frame_annotations = []
    for g in groups:
        f = open(os.path.join(root, g), 'r')
        annotations += [[os.path.splitext(g)[0], *parse(l)] for l in f.readlines()]

    for ann in annotations:
        g, a, b, sa, ea, sb, eb = ann
        if a != b and sa != sb and ea != eb:
            cnt = min(ea - sa, eb - sb)
            af = np.linspace(sa, ea, cnt, endpoint=False, dtype=np.int)
            bf = np.linspace(sb, eb, cnt, endpoint=False, dtype=np.int)
            frame_annotations += [[g, a, f[0], b, f[1]] for f in zip(af, bf)]
            frame_annotations += [[g, b, f[1], a, f[0]] for f in zip(af, bf)]

    return annotations, frame_annotations


def train(net, loader, optimizer, criterion, l2_dist, epoch):
    losses = AverageMeter()
    pos_distance = AverageMeter()
    neg_distance = AverageMeter()
    distance_gap = AverageMeter()
    net.train()
    bar = tqdm(loader, ncols=200)
    for i, (path, frames) in enumerate(loader, 1):
        optimizer.zero_grad()
        out = net(*frames)
        loss = criterion(*out)
        pos_dist = l2_dist(*out[:2])
        neg_dist = l2_dist(out[0], out[2])
        losses.update(loss)
        pos_distance.update(torch.mean(pos_dist), len(path[0]))
        neg_distance.update(torch.mean(neg_dist), len(path[0]))
        distance_gap.update(torch.mean(neg_dist - pos_dist), len(path[0]))
        loss.backward()
        optimizer.step()
        bar.set_description(f'[Epoch {epoch}] '  # [iter {epoch * len(loader) + i}]
                            f'train_loss: {losses.val:.4f}({losses.avg:.4f}), '
                            f'pos_dist: {torch.mean(pos_dist):.4f}({pos_distance.avg:.4f}), '
                            f'neg_dist: {torch.mean(neg_dist):.4f}({neg_distance.avg:.4f}), '
                            f'distance_gap: {torch.mean(neg_dist - pos_dist):.4f}({distance_gap.avg:.4f})')

        bar.update()
    bar.close()
    logger.info(f'[EPOCH {epoch}] '
                f'train_loss: {losses.avg}, '
                f'pos_dist: {pos_distance.avg:.4f}, '
                f'neg_dist: {neg_distance.avg:.4f}, '
                f'gap: {distance_gap.avg:.4f}')
    writer.add_scalar('loss/train_loss', losses.avg, epoch)
    writer.add_scalar('distance/train_pos_distance', pos_distance.avg, epoch)
    writer.add_scalar('distance/train_neg_distance', neg_distance.avg, epoch)
    writer.add_scalar('distance/train_distance_gap', distance_gap.avg, epoch)


@torch.no_grad()
def valid(net, loader, criterion, l2_dist, epoch):
    losses = AverageMeter()
    pos_distance = AverageMeter()
    neg_distance = AverageMeter()
    distance_gap = AverageMeter()
    net.eval()
    bar = tqdm(loader, ncols=200)

    for i, (path, frames) in enumerate(loader, 1):
        out = net(*frames)
        loss = criterion(*out)
        losses.update(loss)
        pos_dist = l2_dist(*out[:2])
        neg_dist = l2_dist(out[0], out[2])
        pos_distance.update(torch.mean(pos_dist), len(path[0]))
        neg_distance.update(torch.mean(neg_dist), len(path[0]))
        distance_gap.update(torch.mean(neg_dist - pos_dist), len(path[0]))

        bar.set_description(f'[Epoch {epoch}] '  # [iter {epoch * len(loader) + i}]
                            f'valid_loss: {losses.val:.4f}({losses.avg:.4f}), '
                            f'pos_dist: {torch.mean(pos_dist):.4f}({pos_distance.avg:.4f}), '
                            f'neg_dist: {torch.mean(neg_dist):.4f}({neg_distance.avg:.4f}), '
                            f'distance_gap: {torch.mean(neg_dist - pos_dist):.4f}({distance_gap.avg:.4f})')

        bar.update()
    bar.close()
    logger.info(f'[EPOCH {epoch}] '
                f'valid_loss: {losses.avg}, '
                f'pos_dist: {pos_distance.avg:.4f}, '
                f'neg_dist: {neg_distance.avg:.4f}, '
                f'gap: {distance_gap.avg:.4f}')
    writer.add_scalar('loss/valid_loss', losses.avg, epoch)
    writer.add_scalar('distance/valid_pos_distance', pos_distance.avg, epoch)
    writer.add_scalar('distance/valid_neg_distance', neg_distance.avg, epoch)
    writer.add_scalar('distance/valid_distance_gap', distance_gap.avg, epoch)


@torch.no_grad()
def positive_ranking2(net, vcdb_loader, vcdb_frame_annotation, epoch, idx_margin=2, query=1000):
    net.eval()
    features = []
    bar = tqdm(vcdb_loader, ncols=200)
    frame_idx = defaultdict(list)
    c = 0
    for i, (path, frame) in enumerate(vcdb_loader):
        out = net(frame, single=True)
        features.append(out)
        bar.update()
        for p in path:
            vid = os.path.basename(os.path.dirname(p))
            frame_idx[vid].append((os.path.basename(p), c))
            c += 1
    features = torch.cat(features).cpu().numpy()

    frame_idx = {k: np.array(sorted(frame_idx[k], key=lambda x: x[0]))[:, 1].astype(np.int) for k, v in
                 frame_idx.items()}
    bar.close()
    vcdb_index = faiss.IndexFlatL2(features.shape[1])
    vcdb_index.add(features)

    idx = defaultdict(list)
    for ann in vcdb_frame_annotation:
        idx[frame_idx[ann[1]][ann[2]]].append(frame_idx[ann[3]][ann[4]])

    # for ann in vcdb_frame_annotation:
    #     g, a, ai, b, bi = ann
    #     if frame_idx.get(a, [-1])[0] != -1 and frame_idx.get(b, [-1])[0] != -1 and len(frame_idx[a]) > ai and len(
    #             frame_idx[b]) > bi:
    #         idx[frame_idx[a][ai]].append(frame_idx[b][bi])

    anchor = np.array(list(idx.keys()))
    bar = tqdm(anchor, ncols=200)
    dist = []
    rank = []
    for i in range(0, len(anchor), query):
        anchor_sub = anchor[i:i + query]
        dist_sub, index = vcdb_index.search(features[anchor_sub], features.shape[0])
        pos = np.array(
            [[n, np.where((correct - idx_margin <= index[n]) & (index[n] <= correct + idx_margin))[0][0]] for n, a in
             enumerate(anchor_sub) for correct in idx[a]])
        pos = (pos[:, 0], pos[:, 1])
        rank_sub = pos[1]
        dist_sub = dist_sub[pos]
        dist.extend(list(dist_sub))
        rank.extend(list(rank_sub))

        bar.set_description(f'[Epoch {epoch}] '  # [iter {epoch * len(loader) + i}]
                            f'dist: {np.mean(dist_sub):.4f}({np.mean(dist):.4f}), '
                            f'rank: {np.mean(rank_sub):>6.1f}({np.mean(rank):.1f})')

        bar.update(anchor_sub.shape[0])
    bar.close()

    bot_30 = int(len(dist) * 0.3)
    logger.info(f'[EPOCH {epoch}] '
                f'dist: {np.mean(dist):.4f}({np.mean(np.sort(dist)[::-1][:bot_30]):.4f})({np.max(dist):.4f}), '
                f'rank: {np.mean(rank):>6.1f}({np.mean(np.sort(rank)[::-1][:bot_30]):.1f})({np.max(rank):.1f})')

    writer.add_scalar('rank/avg_dist', np.mean(dist), epoch)
    writer.add_scalar('rank/avg_rank', np.mean(rank), epoch)
    writer.add_scalar('rank/bottom_30_avg_dist', np.mean(np.sort(dist)[::-1][:bot_30]), epoch)
    writer.add_scalar('rank/bottom_30_rank', np.mean(np.sort(rank)[::-1][:bot_30]), epoch)


@torch.no_grad()
def positive_ranking(net, vcdb_loader, vcdb_positives, epoch):
    net.eval()
    features = []
    paths = []
    bar = tqdm(vcdb_loader, ncols=200)
    for i, (path, frame) in enumerate(vcdb_loader):
        out = net(frame, single=True)
        features.append(out)
        paths.extend(path)
        bar.update()

    features = torch.cat(features).cpu().numpy()
    paths = {(os.path.basename(os.path.dirname(p)), os.path.basename(p)): n for n, p in enumerate(paths)}
    bar.close()

    # vcdb_index=faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(features.shape[1]))
    vcdb_index = faiss.IndexFlatL2(features.shape[1])
    vcdb_index.add(features)
    idx = np.array([[paths[(pos[0], pos[1])], paths[(pos[2], pos[3])]] for pos in vcdb_positives])

    anchor = idx[:, 0]
    dist, index = vcdb_index.search(features[anchor], features.shape[0])
    pos = np.where(index == idx[:, 1:])
    rank = pos[1]
    bot_30 = int(index.shape[0] * 0.3)
    dist = dist[pos]

    logger.info(f'[EPOCH {epoch}] '
                f'dist: {np.mean(dist):.4f}({np.mean(np.sort(dist)[::-1][:bot_30]):.4f})({np.max(dist):.4f}), '
                f'rank: {np.mean(rank):.2f}({np.mean(np.sort(rank)[::-1][:bot_30]):.2f})({np.max(rank):.2f})')
    writer.add_scalar('rank/avg_dist', np.mean(dist), epoch)
    writer.add_scalar('rank/avg_rank', np.mean(rank), epoch)
    writer.add_scalar('rank/bottom_30_avg_dist', np.mean(np.sort(dist)[::-1][:bot_30]), epoch)
    writer.add_scalar('rank/bottom_30_rank', np.mean(np.sort(rank)[::-1][:bot_30]), epoch)


def main():
    parser = argparse.ArgumentParser(description="Train for VCDB Retrieval.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-m', '--margin', type=float, default=0.3)
    parser.add_argument('-c', '--comment', type=str, default='')
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-o', '--optim', type=str, default='sgd')
    args = parser.parse_args()

    margin = args.margin
    learning_rate = args.learning_rate
    weight_decay = 0  # 5e-5
    ckpt = None

    vcdb_positives_path = 'sampling/data/vcdb_positive.csv'
    train_triplets_path = 'sampling/data/fivr_triplet_0809.csv'  # 'sampling/fivr_triplet.csv'
    valid_triplets_path = 'sampling/data/vcdb_triplet_0806.csv'

    init_logger(args.comment)
    logger.info(args)
    logger.info(f'lr: {learning_rate}, margin: {margin}')
    logger.info(f'train_triplets_path: {train_triplets_path}, valid_triplets_path: {valid_triplets_path}')

    # Model
    embed_net = MobileNet_RMAC()
    net = TripletNet(embed_net).cuda()
    writer.add_graph(net, [torch.rand((2, 3, 224, 224)).cuda(),
                           torch.rand((2, 3, 224, 224)).cuda(),
                           torch.rand((2, 3, 224, 224)).cuda()])
    logger.info(net)
    # logger.info(net.summary((3, 3, 224, 224)))
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    # for n,p in net.named_parameters():
    #     print(n, p.requires_grad)

    # Optimizer
    criterion = nn.TripletMarginLoss(margin)
    l2_dist = nn.PairwiseDistance()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    if args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    # Data
    transform = {
        'train': trn.Compose([
            # trn.RandomResizedCrop(224),
            # trn.RandomRotation(30),
            # trn.RandomHorizontalFlip(p=0.3),
            # trn.RandomVerticalFlip(p=0.1),
            trn.Resize((224, 224)),
            ImageNetPolicy(),

            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'valid': trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    logger.info(transform)
    train_triplets = read_triplets(train_triplets_path)
    train_triplets_loader = DataLoader(
        TripletDataset(train_triplets, '/MLVD/FIVR/frames', transform=transform['train']),
        batch_size=64, shuffle=True, num_workers=4)

    valid_triplets = read_triplets(valid_triplets_path)
    valid_triplets_loader = DataLoader(
        TripletDataset(valid_triplets, '/MLVD/VCDB/frames', transform=transform['valid']),
        batch_size=64, shuffle=False, num_workers=4)

    vcdb_core = np.load('/MLVD/VCDB/meta/vcdb_core.pkl', allow_pickle=True)
    vcdb_positives = read_positive_csv(vcdb_positives_path)
    vcdb_annotation, vcdb_frame_annotation = scan_vcdb_annotation('/MLVD/VCDB/annotation')
    vcdb_all_frames = np.array(
        [os.path.join('/MLVD/VCDB/frames', k, f) for k, frames in vcdb_core.items() for f in frames])

    vcdb_all_frames_loader = DataLoader(ListDataset(vcdb_all_frames, transform=transform['valid']),
                                        batch_size=128,
                                        shuffle=False, num_workers=4)

    # valid(net, valid_triplets_loader, criterion, l2_dist, 0)
    positive_ranking2(net, vcdb_all_frames_loader, vcdb_frame_annotation, 0, 2, 10000)
    positive_ranking2(net, vcdb_all_frames_loader, vcdb_frame_annotation, 0, 2, 1000)
    positive_ranking2(net, vcdb_all_frames_loader, vcdb_frame_annotation, 0, 2, 100)
    positive_ranking2(net, vcdb_all_frames_loader, vcdb_frame_annotation, 0, 2, 10)

    for e in range(1, args.epoch, 1):
        train(net, train_triplets_loader, optimizer, criterion, l2_dist, e)
        # valid(net, valid_triplets_loader, criterion, l2_dist, e)
        # positive_ranking(net, vcdb_all_frames_loader, vcdb_positives, e)
        positive_ranking2(net, vcdb_all_frames_loader, vcdb_frame_annotation, 0, 2, 1000)
        scheduler.step()

        # print(f'[EPOCH {e}] {d}')
        # torch.save({'epoch': e,
        #             'model_state_dict': net.module.embedding_net.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             }, f'{ckpt_dir}/epoch_{e}_ckpt.pth')


if __name__ == '__main__':
    main()
