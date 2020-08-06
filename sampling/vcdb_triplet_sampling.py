import numpy as np
import os
import faiss
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import torch

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))


def load(p):
    f = torch.load(p)
    # print(n,f.shape,p)
    return f, f.shape[0]


def load_feature(paths):
    # load = lambda p: np.load(p)
    bar = tqdm(paths, ncols=150)
    pool = Pool()
    results = [pool.apply_async(load, args=[p], callback=lambda *a: bar.update()) for p in paths]
    # results = [load(n,p) for n,p in enumerate(paths)]
    pool.close()
    pool.join()
    bar.close()
    results = [(f.get()) for f in results]
    features = np.concatenate([r[0] for r in results])
    length = [r[1] for r in results]

    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    return features, length, index


def read_text(filename):
    with open(filename, 'r') as f:
        l = [i.strip().split(',') for i in f.readlines()]
    df = pd.DataFrame(l, columns=['idx', 'ann_idx', 'group', 'a', 'a_frame_idx', 'a_frame', 'b', 'b_frame_idx',
                                  'b_frame', 'dist'])
    df = df.astype({'idx': int, 'ann_idx': int, 'a_frame_idx': int, 'b_frame_idx': int, 'dist': float})

    return df


def distance(a, b):
    idx = faiss.IndexFlatL2(a.shape[1])
    idx.add(a)
    dist, _ = idx.search(b, 1)

    return dist[0][0]


if __name__ == '__main__':
    fivr_bg = np.load('/MLVD/VCDB/meta/vcdb_bg.pkl', allow_pickle=True)
    # positives = read_text('dataset/positive_beautiful_mind_game_theory.txt')
    positives = read_text('vcdb_positive.txt')
    save_to = 'vcdb_triplet_0805_margin.csv'
    print(positives)

    videos = {v: n for n, v in enumerate(sorted(list(set(list(positives.a) + list(positives.b)))))}
    bg_videos = {v: n for n, v in enumerate(np.load('/MLVD/VCDB/meta/vcdb_videos_bg.npy')[:1000])}

    # feature_base = '/hdd/FIVR_core/mobilenet/rmac'
    feature_base = '/MLVD/VCDB/mobilenet/rmac'
    features, length, index = load_feature([f'{feature_base}/{v}.pth' for v in videos])
    bg_features, bg_length, bg_index = load_feature([f'{feature_base}/{v}.pth' for v in bg_videos])

    bg_cpu_index = faiss.IndexFlatL2(bg_features.shape[1])
    bg_index = faiss.index_cpu_to_all_gpus(bg_cpu_index)
    bg_index.add(bg_features)

    bg_table = np.array([[v, i] for n, v in enumerate(bg_videos) for i in range(bg_length[n])])
    print(bg_table)

    triplets = []
    for pos in tqdm(positives.values):
        idx, ann_idx, group, a, ai, a_frame, b, bi, b_frame, p_dist = pos

        af = features[index[videos[a]][0] + ai].reshape(1, -1)
        bf = features[index[videos[b]][0] + bi].reshape(1, -1)
        pos_dist = distance(af, bf)

        if pos_dist != 0:
            neg_dist, neg_idx = bg_index.search(np.concatenate([af, bf]), 10)
            triplets += [[idx, ann_idx, group, a, ai, a_frame, b, bi, b_frame, *bg_table[neg_idx[0][n]],
                          fivr_bg[bg_table[neg_idx[0][n]][0]][int(bg_table[neg_idx[0][n]][1])],
                          p_dist, pos_dist, i] for n, i in enumerate(neg_dist[0]) if pos_dist-0.3< i < pos_dist]

    print(len(triplets))
    df = pd.DataFrame(triplets,
                      columns=['idx', 'ann_idx', 'group', 'anc', 'anc_frame_idx', 'anc_frame', 'pos',
                               'pos_frame_idx', 'pos_frame', 'neg', 'neg_frame_idx', 'neg_frame', 'p_dist_0',
                               'p_dist_1', 'n_dist'])

    df.to_csv(save_to, index=False)
