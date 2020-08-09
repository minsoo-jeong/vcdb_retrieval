import numpy as np
import torch
import tqdm
import matplotlib.pylab as plt
import cv2
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
from multiprocessing import Pool
import faiss
import warnings
import os
import pickle as pk


def vcdb_pairs():
    def time2sec(t):
        return sum([int(i) * 60 ** (2 - n) for n, i in enumerate(t.split(':'))])

    videos = np.load('/MLVD/VCDB/meta/vcdb_videos.npy')[:528]
    # query = videos[:528]

    groups = dict()
    pairs = dict()
    core_duration = {meta['filename']: meta['duration'] for meta in
                     np.load(f'/MLVD/VCDB/vcdb_metadata.npy', allow_pickle=True)[:528]}
    video2idx = {name: n for n, name in enumerate(videos)}
    idx2video = {n: name for n, name in enumerate(videos)}

    ann = [dict() for q in videos]
    # redundant pair
    for g in os.listdir(f'/MLVD/VCDB/annotation'):
        group = g.split('.')[0]
        for l in open(f'/MLVD/VCDB/annotation/{g}').readlines():
            a, b, sa, ea, sb, eb = l.strip().split(',')
            groups[a] = groups[b] = group

            sa, ea, sb, eb = list(map(lambda x: time2sec(x), [sa, ea, sb, eb]))
            key, value = (f'{a}/{b}', (a, b, sa, ea, sb, eb)) if a > b else (f'{b}/{a}', (b, a, sb, eb, sa, ea))
            if pairs.get(key):
                pairs[key].append(value)
            else:
                pairs[key] = [value]

    return pairs, groups, video2idx, idx2video


def load(path):
    feat = torch.load(path)
    return feat


def load_features(paths):
    pool = Pool()
    bar = tqdm.tqdm(range(len(paths)), mininterval=1, ncols=150)
    features = [pool.apply_async(load, args=[path], callback=lambda *a: bar.update()) for path in paths]
    pool.close()
    pool.join()
    bar.close()
    features = [f.get() for f in features]
    length = np.array([f.shape[0] for f in features])
    index = np.array([(length[:n].sum(), length[:n].sum() + l) for n, l in enumerate(length)])

    return np.concatenate(features), length, index


def parse_pairs(pairs, video2idx, length, frame_idx):
    # print(video2idx)
    # print(pairs.values())
    frame_pair = dict()
    for key, pair in pairs.items():
        a, b = key.split('/')
        a_v_idx, b_v_idx = video2idx[a], video2idx[b]
        a_v_len, b_v_len = length[a_v_idx], length[b_v_idx]
        a_f_idx, b_f_idx = frame_idx[a_v_idx], frame_idx[b_v_idx]
        # print(a, a_v_idx, a_v_len, a_f_idx, b, b_v_idx, b_v_len, b_f_idx)
        if frame_pair.get(a_v_idx) is None:
            frame_pair[a_v_idx] = {'db': set(), 'video': dict()}
        if frame_pair.get(b_v_idx) is None:
            frame_pair[b_v_idx] = {'db': set(), 'video': dict()}
        for p in pair:
            (_, _, sa, ea, sb, eb) = p
            a_seg = np.arange(sa, min(ea + 1, a_v_len))
            b_seg = np.arange(sb, min(eb + 1, b_v_len))

            a_from_b = np.linspace(a_seg[0], a_seg[-1], num=b_seg.shape[0], dtype=np.int)
            b_from_a = np.linspace(b_seg[0], b_seg[-1], num=a_seg.shape[0], dtype=np.int)

            if frame_pair[a_v_idx]['video'].get(b_v_idx) is None:
                frame_pair[a_v_idx]['video'][b_v_idx] = set()
            # frame_pair[a_v_idx]['video'][b_v_idx].update({i for i in zip(a_f_idx[0]+a_seg, b_from_a)})
            frame_pair[a_v_idx]['video'][b_v_idx].update({i for i in zip(a_seg, b_from_a)})
            frame_pair[a_v_idx]['db'].update({i for i in zip(a_f_idx[0] + a_seg, b_f_idx[0] + b_from_a)})

            if frame_pair[b_v_idx]['video'].get(a_v_idx) is None:
                frame_pair[b_v_idx]['video'][a_v_idx] = set()
            # frame_pair[b_v_idx]['video'][a_v_idx].update({i for i in zip(b_f_idx[0]+b_seg, a_from_b)})
            frame_pair[b_v_idx]['video'][a_v_idx].update({i for i in zip(b_seg, a_from_b)})
            frame_pair[b_v_idx]['db'].update({i for i in zip(b_f_idx[0] + b_seg, a_f_idx[0] + a_from_b)})
    return frame_pair


def search_all_db(query_feature, frame_pair, db_size, margin=0):
    dist, index = all_video_db.search(query_feature, db_size)
    query_idx = list(set([i[0] for i in frame_pair]))
    positive_idx = [frame_pair[frame_pair[:, 0] == qi][:, 1] for qi in query_idx]

    # time margin
    rank = [[np.where(abs(index[n, :] - v) <= margin)[0][0] for v in p] for n, p in enumerate(positive_idx)]
    # rank = [[np.where(index[n, :] == v)[0][0] for v in p] for n, p in enumerate(positive_idx)]
    rank_dist = [dist[n, row] for n, row in enumerate(rank)]
    ap = [sum([k / (r + 1) for k, r in enumerate(sorted(row), start=1)]) / len(row) for n, row in enumerate(rank)]

    return rank, rank_dist, ap


def search_each_video(vcdb_features, frame_pair):
    rank_video, rank_per_video, dist_video, ap_video = [], [], [], []

    video_idx = sorted(frame_pair.keys())
    for vi in video_idx:
        video_frame_pair = np.array(sorted(frame_pair[vi]))
        query_feature_idx = list(set([i[0] for i in video_frame_pair]))
        query_feature = vcdb_features[query_feature_idx, :]

        ref_feature_idx = vcdb_frame_index[vi]
        ref_feature = vcdb_features[ref_feature_idx[0]:ref_feature_idx[1], :]
        reference = faiss.IndexFlatIP(ref_feature.shape[1])
        reference.add(ref_feature)

        rank, rpv, dist, ap = search_video(query_feature, reference, video_frame_pair, ref_feature.shape[0])
        rank_video.append(rank)
        rank_video.append(rpv)
        dist_video.append(dist)
        ap_video.append(ap)
    return rank_video, rank_video, dist_video, ap_video


def search_video(query_feature, reference, frame_pair, dbsize, margin=0):
    dist, index = reference.search(query_feature, dbsize)
    query_idx = list(set([i[0] for i in frame_pair]))
    positive_idx = [frame_pair[frame_pair[:, 0] == qi][:, 1] for qi in query_idx]

    # time margin
    rank = [[np.where(abs(index[n, :] - v) <= margin)[0][0] for v in p] for n, p in enumerate(positive_idx)]
    # rank = [[np.where(index[n, :] == v)[0][0] for v in p] for n, p in enumerate(positive_idx)]
    rank_dist = [dist[n, row] for n, row in enumerate(rank)]
    rank_per_video = [[r / dbsize for r in row] for row in rank]

    ap = np.array(
        [sum([k / (r + 1) for k, r in enumerate(sorted(row), start=1)]) / len(row) for n, row in enumerate(rank)])

    return rank, rank_per_video, rank_dist, ap


def search(query_frame_pair_all_db, query_frame_pair_video):
    query_feature_idx = list(set([i[0] for i in query_frame_pair_all_db]))
    query_feature = vcdb_features[query_feature_idx, :]
    rank_db, dist_db, ap_db = search_all_db(query_feature, query_frame_pair_all_db, vcdb_features.shape[0])

    rank_video, rank_per_video, dist_video, ap_video = search_each_video(vcdb_features, query_frame_pair_video)

    return (rank_db, dist_db, ap_db), (rank_video, rank_per_video, dist_video, ap_video)


def avg(arr):
    return [np.mean(a) for a in arr]


def cat(arr):
    return np.concatenate(arr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('-f', '--feat_dir', type=str,
                        default='/MLVD/VCDB/mobilenet/rmac/')
    parser.add_argument('-q', '--query', type=int, default=528)
    parser.add_argument('-v', '--video', type=int, default=528)
    parser.add_argument('-m', '--time_margin', type=int, default=2)

    args = parser.parse_args()
    print(args)
    assert (args.feat_dir[-1] == '/')
    assert (args.time_margin >= 0)
    pairs, groups, video2idx, idx2video = vcdb_pairs()

    vcdb_videos = np.load('/MLVD/VCDB/meta/vcdb_videos.npy')[:args.video]
    vcdb_feature_path = np.char.add(np.char.add(f'{args.feat_dir}', vcdb_videos), '.pth')
    global vcdb_features
    vcdb_features, length, vcdb_frame_index = load_features(vcdb_feature_path)
    # vcdb_features = vcdb_features / (np.linalg.norm(vcdb_features, ord=2, axis=1, keepdims=True) + 1e-15)
    vcdb_frame_pairs = parse_pairs(pairs, video2idx, length, vcdb_frame_index)

    query_videos = np.load('/MLVD/VCDB/meta/vcdb_videos.npy')[:args.query]

    global all_video_db
    # all_video_db = faiss.IndexFlatL2(vcdb_features.shape[1])
    all_video_db = faiss.IndexFlatIP(vcdb_features.shape[1])
    all_video_db.add(vcdb_features)
    print(vcdb_features.shape)

    # ver1 - non mp 1~2 s/it
    rank_db, dist_db, ap_db, rank_video, rank_ratio_video, dist_video, ap_video = [], [], [], [], [], [], []
    avg_rank_per_query_db, avg_dist_per_query_db, avg_rank_per_query_video, avg_rank_ratio_per_query_video, avg_dist_per_query_video = [], [], [], [], []

    query_progress = tqdm.tqdm(query_videos, ncols=150, mininterval=.1)
    for qv in query_videos:
        query_video_idx = video2idx[qv]
        query_frame_pair = vcdb_frame_pairs[query_video_idx]
        query_frame_pair_all_db = np.array(sorted(query_frame_pair['db']))
        query_frame_pair_video = query_frame_pair['video']

        # all DB
        # query_feature_idx = list(set([i[0] for i in query_frame_pair_all_db]))
        query_feature_idx = list(set([i[0] for i in query_frame_pair_all_db]))
        query_feature = vcdb_features[query_feature_idx, :]
        rank, dist, ap = search_all_db(query_feature, query_frame_pair_all_db, vcdb_features.shape[0],
                                       margin=args.time_margin)

        avg_rank_per_q, avg_dist_per_q = avg(rank), avg(dist)
        rank, dist = cat(rank), cat(dist)

        rank_db.extend(rank), dist_db.extend(dist), ap_db.extend(ap)
        avg_rank_per_query_db.extend(avg_rank_per_q)
        avg_dist_per_query_db.extend(avg_dist_per_q)

        query_progress.write(f'DB - '
              f'Rank: {rank.mean():>6.1f}({np.array(rank_db).mean():.1f}), '
              f'Dist: {dist.mean():>6.4f}({np.array(dist_db).mean():.4f}), '
              f'Rank/Q: {np.array(avg_rank_per_q).mean():>6.1f}({np.array(avg_rank_per_query_db).mean():.1f}), '
              f'Dist/Q: {np.array(avg_dist_per_q).mean():>6.4f}({np.array(avg_dist_per_query_db).mean():.4f}), '
              f'mAP: {np.array(ap).mean():>4.2f}({np.array(ap_db).mean():.2f}), '
              f'{groups[qv][:10]}, {qv}',
              )

        # rank_video, rank_per_video, dist_video, ap_video = search_each_video(vcdb_features, query_frame_pair_video)

        # query_frame_idx = vcdb_frame_index[query_video_idx]
        # query_video_feature = vcdb_features[query_frame_idx[0]:query_frame_idx[1], :]
        # video_idx = sorted(query_frame_pair_video.keys())
        # for vi in video_idx:
        #     ref_vid = idx2video[vi]
        #     video_frame_pair = np.array(sorted(query_frame_pair_video[vi]))
        #     query_feature_idx = list(set([i[0] for i in video_frame_pair]))
        #     query_feature = query_video_feature[query_feature_idx, :]
        #
        #     ref_feature_idx = vcdb_frame_index[vi]
        #     ref_feature = vcdb_features[ref_feature_idx[0]:ref_feature_idx[1], :]
        #     reference = faiss.IndexFlatIP(ref_feature.shape[1])
        #     reference.add(ref_feature)
        #
        #     print(video_frame_pair.shape, query_feature.shape,)
        #     rank, rank_ratio, dist, ap = search_video(query_feature, reference, video_frame_pair,
        #                                               ref_feature.shape[0], margin=args.time_margin)
        #
        #     rank, rank_ratio, dist = cat(rank), cat(rank_ratio), cat(dist)
        #     avg_rank_per_q, avg_rank_ratio_per_q, avg_dist_per_q = avg(rank), avg(rank_ratio), avg(dist)
        #
        #     rank_video.extend(rank), rank_ratio_video.extend(rank_ratio), dist_video.extend(dist), ap_video.extend(ap)
        #     avg_rank_per_query_video.extend(avg_rank_per_q)
        #     avg_rank_ratio_per_query_video.extend(avg_rank_ratio_per_q)
        #     avg_dist_per_query_video.extend(avg_dist_per_q)
        #
        #     print(f'Video - '
        #           f'Rank: {rank.mean():>4.1f}({np.array(rank_video).mean():.1f}), '
        #           f'RankRatio: {rank_ratio.mean():>6.4f}({np.array(rank_ratio_video).mean():.4f}), '
        #           f'Dist: {dist.mean():>6.4f}({np.array(dist_video).mean():.4f}), '
        #           f'Rank/Q: {np.array(avg_rank_per_q).mean():>4.1f}({np.array(avg_rank_per_query_video).mean():.1f}), '
        #           f'RankRatio/Q: {np.array(avg_rank_ratio_per_q).mean():>6.4f}({np.array(avg_rank_ratio_per_query_video).mean():.4f}), '
        #           f'Dist/Q: {np.array(avg_dist_per_q).mean():>6.4f}({np.array(avg_dist_per_query_video).mean():.4f}), '
        #           f'mAP: {np.array(ap).mean():>4.2f}({np.array(ap_video).mean():.2f}), '
        #           f'{groups[qv][:10]}, {groups[ref_vid][:10]}, {qv}, {ref_vid}',
        #           )

        query_progress.update()

    query_progress.close()
