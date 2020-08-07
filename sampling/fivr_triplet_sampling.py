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


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.astype({'scene_start': int, 'scene_end': int, 'a_frame_idx': int, 'b_frame_idx': int, 'dist': float})

    return df


def read_text(filename):
    with open(filename, 'r') as f:
        l = [i.strip().split(',') for i in f.readlines()]
    df = pd.DataFrame(l, columns=['idx', 'a', 'scene_start', 'scene_end', 'a_frame_idx', 'a_frame', 'b', 'b_frame_idx',
                                  'b_frame', 'dist'])
    df = df.astype({'scene_start': int, 'scene_end': int, 'a_frame_idx': int, 'b_frame_idx': int, 'dist': float})

    return df


def distance(a, b):
    idx = faiss.IndexFlatL2(a.shape[1])
    idx.add(a)
    dist, _ = idx.search(b, 1)

    return dist[0][0]


# def fivr_triplet_sampling(model,positive_path):
#     fivr_bg = np.load('/hdd/FIVR_core/fivr_bg.pkl', allow_pickle=True)
#     positives = read_text(positive_path)
#     videos = {v: n for n, v in enumerate(sorted(list(set(list(positives.a) + list(positives.b)))))}
#     bg_videos = {v: n for n, v in enumerate(np.load('/hdd/FIVR_core/fivr_videos_bg.npy')[:1000])}

if __name__ == '__main__':
    # fivr_bg = np.load('/hdd/FIVR_core/fivr_bg.pkl', allow_pickle=True)
    fivr_bg = np.load('/MLVD/FIVR/meta/fivr_bg.pkl', allow_pickle=True)
    # positives = read_text('dataset/positive_beautiful_mind_game_theory.txt')
    positives = read_csv('fivr_positive.csv')
    print(positives)
    save_to = 'fivr_triplet_0806.csv'
    videos = {v: n for n, v in enumerate(sorted(list(set(list(positives.a) + list(positives.b)))))}
    # bg_videos = {v: n for n, v in enumerate(np.load('/hdd/FIVR_core/fivr_videos_bg.npy')[:1000])}
    bg_videos = {v: n for n, v in enumerate(np.load('/MLVD/FIVR/meta/fivr_videos_bg.npy')[:10000])}

    feature_base = '/MLVD/FIVR/mobilenet/rmac'
    # feature_base = '/MLVD/VCDB/mobilenet/center224_rmac'
    features, length, index = load_feature([f'{feature_base}/{v}.pth' for v in videos])
    bg_features, bg_length, _ = load_feature([f'{feature_base}/{v}.pth' for v in bg_videos])
    print(features.shape)
    print(bg_features.shape)
    bg_index = faiss.IndexFlatL2(bg_features.shape[1])
    bg_index = faiss.index_cpu_to_all_gpus(bg_index)
    bg_index.add(bg_features)

    bg_table = np.array([[v, i] for n, v in enumerate(bg_videos) for i in range(bg_length[n])])
    # print(bg_table)

    triplets = []
    for pos in tqdm(positives.values):
        idx, a, ss, se, ai, a_frame, b, bi, b_frame, p_dist = pos

        af = features[index[videos[a]][0] + ai].reshape(1, -1)
        bf = features[index[videos[b]][0] + bi].reshape(1, -1)
        pos_dist = distance(af, bf)

        if pos_dist != 0:
            neg_dist, neg_idx = bg_index.search(np.concatenate([af, bf]), 10)
            triplets += [[idx, ss, se, a, ai, a_frame, b, bi, b_frame, *bg_table[neg_idx[0][n]],
                          fivr_bg[bg_table[neg_idx[0][n]][0]][int(bg_table[neg_idx[0][n]][1])],
                          p_dist, pos_dist, i] for n, i in
                         enumerate(neg_dist[0]) if pos_dist - 0.3 < i < pos_dist]

            triplets += [[idx, ss, se, b, bi, b_frame, a, ai, a_frame, *bg_table[neg_idx[1][n]],
                          fivr_bg[bg_table[neg_idx[1][n]][0]][int(bg_table[neg_idx[1][n]][1])],
                          p_dist, pos_dist, i] for n, i in
                         enumerate(neg_dist[1]) if pos_dist - 0.3 < i < pos_dist]

    print(len(triplets))
    df = pd.DataFrame(triplets,
                      columns=['ann_idx', 'scene_start', 'scene_end', 'anc', 'anc_frame_idx', 'anc_frame', 'pos',
                               'pos_frame_idx',
                               'pos_frame', 'neg', 'neg_frame_idx', 'neg_frame', 'p_dist_0', 'p_dist_1', 'n_dist'])
    # df.to_csv('dataset/triplet_beautiful_mind_game_theory2.csv',index=False)
    # df.to_csv('dataset/triplet_baggio_penalty_19942.csv', index=False)
    df.to_csv(save_to, index=False)
