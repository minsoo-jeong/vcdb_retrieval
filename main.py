import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as trn

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
kst = timezone(timedelta(hours=9))


def read_triplets(csv_path):
    df = pd.read_csv(csv_path)[['anc', 'anc_frame', 'pos', 'pos_frame', 'neg', 'neg_frame']].to_numpy()
    return df


# to csv
def read_positive_pair(filename):
    with open(filename, 'r') as f:
        l = [i.strip().split(',') for i in f.readlines()]
        df = pd.DataFrame(l, columns=['idx', 'ann_idx', 'group', 'a', 'a_frame_idx', 'a_frame', 'b', 'b_frame_idx',
                                      'b_frame', 'dist'])[['a', 'a_frame', 'b', 'b_frame']].to_numpy()

    return df


def init_logger(comment=''):
    current = datetime.now(kst)
    current_date = current.strftime('%m%d')
    current_time = current.strftime('%H%M%S')
    basename = comment if comment is not None or comment != '' else current_time

    log_dir = f'/hdd/vcdb_retrieval_ckpt/{current_date}/{basename}'
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

    return log_dir, ckpt_dir


def train(net, loader, optimizer, criterion, l2_dist, epoch, tag='loss/train_loss'):
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
        # if i % 10 == 0:
        #     bar.write(f'[Epoch {epoch}] ' # iter : {epoch * len(loader) + i}
        #               f'train_loss : {losses.val:.4f}({losses.avg:.4f})')
        bar.update()
    bar.close()
    logger.info(f'[EPOCH {epoch}] '
                f'train_loss: {losses.avg}, '
                f'pos_dist: {pos_distance.avg:.4f}, '
                f'neg_dist: {neg_distance.avg:.4f}, '
                f'gap: {distance_gap.avg:.4f}')
    writer.add_scalar(tag, losses.avg, epoch)
    writer.add_scalar('distance/train_pos_distance', pos_distance.avg, epoch)
    writer.add_scalar('distance/train_neg_distance', neg_distance.avg, epoch)
    writer.add_scalar('distance/train_distance_gap', distance_gap.avg, epoch)


def valid(net, loader, criterion, l2_dist, epoch, tag='loss/valid_loss'):
    losses = AverageMeter()
    pos_distance = AverageMeter()
    neg_distance = AverageMeter()
    distance_gap = AverageMeter()
    net.eval()
    bar = tqdm(loader, ncols=200)
    with torch.no_grad():
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
            # if i % 10 == 0:
            #     bar.write(f'[Epoch {epoch}] ' # iter : {epoch * len(loader) + i}
            #               f'valid_loss : {losses.val:.4f}({losses.avg:.4f})')
            bar.update()
    bar.close()
    logger.info(f'[EPOCH {epoch}] '
                f'valid_loss: {losses.avg}, '
                f'pos_dist: {pos_distance.avg:.4f}, '
                f'neg_dist: {neg_distance.avg:.4f}, '
                f'gap: {distance_gap.avg:.4f}')
    writer.add_scalar(tag, losses.avg, epoch)
    writer.add_scalar('distance/valid_pos_distance', pos_distance.avg, epoch)
    writer.add_scalar('distance/valid_neg_distance', neg_distance.avg, epoch)
    writer.add_scalar('distance/valid_distance_gap', distance_gap.avg, epoch)


def eval(net, loader, l2_dist, epoch, tag='eval/positive_distance'):
    distance = AverageMeter()
    net.eval()
    bar = tqdm(loader, ncols=200)
    with torch.no_grad():
        for i, (path, frames) in enumerate(loader, 1):
            out = net(*frames, frames[0])
            dist = l2_dist(*out[:2])
            distance.update(torch.mean(dist), len(path[0]))
            bar.set_description(f'[Epoch {epoch}][{distance.count}/{len(loader.dataset)}] '
                                f'distance : {distance.sum:.4f}')
            # print(torch.pow(loss,2), dist[i])
            # bar.rate({'abc':'123'})
            bar.update()
        bar.close()
        logger.info(f'[EPOCH {epoch}] distance : {distance.sum}')
        writer.add_scalar(tag, distance.sum, epoch)


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
    dist = dist[pos]

    logger.info(f'[EPOCH {epoch}] '
                f'dist: {np.mean(dist):.4f}({np.mean(np.sort(dist)[::-1][:100]):.4f})({np.max(dist):.4f}), '
                f'rank: {np.mean(rank):.2f}({np.mean(np.sort(rank)[::-1][:100]):.4f})({np.max(rank):.2f})')
    writer.add_scalar('rank/avg_dist', np.mean(dist), epoch)
    writer.add_scalar('rank/avg_rank', np.mean(rank), epoch)
    writer.add_scalar('rank/top100_avg_dist', np.mean(np.sort(dist)[::-1][:100]), epoch)
    writer.add_scalar('rank/top100_avg_rank', np.mean(np.sort(rank)[::-1][:100]), epoch)


def main():
    parser = argparse.ArgumentParser(description="Train for VCDB Retrieval.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-m', '--margin', type=float, default=0.3)
    parser.add_argument('-c', '--comment', tyep=str, default='')
    args = parser.parse_args()

    margin = args.margin
    learning_rate = args.learning_rate
    weight_decay = 0  # 5e-5
    ckpt = None

    vcdb_positives_path = 'sampling/vcdb_positive.txt'
    train_triplets_path = 'sampling/fivr_triplet_0805_margin.csv'  # 'sampling/fivr_triplet.csv'
    valid_triplets_path = 'sampling/vcdb_triplet_0805_margin.csv'

    log_dir, ckpt_dir = init_logger(args.comment)
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
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
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
    vcdb_positives = read_positive_pair(vcdb_positives_path)
    vcdb_all_frames = np.array(
        [os.path.join('/MLVD/VCDB/frames', k, f) for k, frames in vcdb_core.items() for f in frames])

    vcdb_all_frames_loader = DataLoader(ListDataset(vcdb_all_frames, transform=transform['valid']), batch_size=128,
                                        shuffle=False, num_workers=4)

    valid(net, valid_triplets_loader, criterion, l2_dist, 0)
    positive_ranking(net, vcdb_all_frames_loader, vcdb_positives, 0)

    for e in range(1, 50, 1):
        train(net, train_triplets_loader, optimizer, criterion, l2_dist, e)
        valid(net, valid_triplets_loader, criterion, l2_dist, e)
        positive_ranking(net, vcdb_all_frames_loader, vcdb_positives, e)
        scheduler.step()

        # print(f'[EPOCH {e}] {d}')
        # torch.save({'epoch': e,
        #             'model_state_dict': net.module.embedding_net.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             }, f'{ckpt_dir}/epoch_{e}_ckpt.pth')


if __name__ == '__main__':
    main()
