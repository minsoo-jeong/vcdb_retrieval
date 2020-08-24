from multiprocessing import Pool
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import numpy as np
import faiss
import torch
import os

import numpy as np
import sys


class TN(object):
    def __init__(self, D, I, TEMP_WND=3, MIN_MATCH=3):
        sys.setrecursionlimit(5000)

        self.TEMP_WND = TEMP_WND
        self.MIN_MATCH = MIN_MATCH

        self.index = I
        self.dist = D

        self.query_length = D.shape[0]

        # isdetect, next_time,next_rank, scores, count
        self.paths = np.zeros((*D.shape, 5), dtype=object)

    def find_linkable_node(self, t, r):
        v_id, f_id = self.index[t, r]
        time, rank = np.where((self.index[t + 1:t + 1 + self.TEMP_WND, :, 0] == v_id) &
                              (f_id < self.index[t + 1:t + 1 + self.TEMP_WND, :, 1]) &
                              (self.index[t + 1:t + 1 + self.TEMP_WND, :, 1] <= f_id + self.TEMP_WND))

        return np.dstack((time + 1 + t, rank)).squeeze(0).tolist()

    def find_max_score_path(self, t, r):

        if self.paths[t, r, 0] != 1:
            nodes = self.find_linkable_node(t, r)
            paths = [[time, rank, *self.find_max_score_path(time, rank)] for time, rank in nodes]
            if len(paths) != 0:
                path = sorted(paths, key=lambda x: x[-2] / x[-1])[0]
                # print(t,r,sorted(paths, key=lambda x: x[-2]/x[-1], reverse=True))
                next_time, next_rank = path[0], path[1]
                score = path[5] + self.dist[t, r]
                count = path[6] + 1
            else:
                next_time, next_rank, score, count = -1, -1, self.dist[t, r], 1
            # print('find', t,r,[1, next_time, next_rank, score])
            self.paths[t, r] = [1, next_time, next_rank, score, count]
        else:
            pass  # print('find-already', t, r, self.paths[t, r])
        return self.paths[t, r]

    def fit(self):
        candidate = []
        for t in range(self.query_length):
            for rank, (v_idx, f_idx) in enumerate(self.index[t]):
                q = [t, t]
                r = [self.index[t, rank, 1], self.index[t, rank, 1]]

                _, next_time, next_rank, score, count = self.find_max_score_path(t, rank)
                while next_time != -1:
                    q[1] = next_time
                    r[1] = self.index[next_time, next_rank, 1]
                    _, next_time, next_rank, _, _ = self.paths[next_time, next_rank]

                if count >= self.MIN_MATCH:
                    candidate.append((v_idx, q, r, score, count))

        candidate = sorted(candidate, key=lambda x: x[-2] / x[-1])
        nms_candidate = []
        for c in candidate:
            flag = True
            for nc in nms_candidate:
                if nc[0] == c[0] and (not (nc[1][1] < c[1][0] or c[1][1] < nc[1][0]) and not (nc[2][1] < c[2][0] or c[2][1] < nc[2][0])):
                    flag = False
                    break
            if flag:
                nms_candidate.append(c)

        return nms_candidate

        # # for time, tlist in enumerate(self.dist.shape[0]):
        # #     for rank, s in enumerate(tlist):
        # #         if self.score[time, rank] > self.SCORE_THR and self.dy_table[time, rank][0] == -1:
        # #             _, _, max_q, max_r, cnt, path_score = self.get_maximum_path(time, rank)
        # #
        # #             # half-closed form [ )
        # #             if cnt >= self.MIN_MATCH:
        # #                 detect = {'query': Period(time, max_q + 1),
        # #                           'ref': Period(self.idx[time, rank], max_r + 1),
        # #                           'match': cnt,
        # #                           'score': path_score}
        # #                 candidate.append(detect)
        #
        # # candidate.sort(key=lambda x: x['score'], reverse=True)
        # # [print(c) for c in candidate]
        # # 1. overlap -> NMS
        # '''
        # nms = [True for i in range(len(candidate))]
        # for i in range(0, len(candidate)):
        #     key = candidate[i]
        #     for j in range(i + 1, len(candidate)):
        #         if nms[j] and key['query'].is_overlap(candidate[j]['query']) and key['ref'].is_overlap(
        #                 candidate[j]['ref']):
        #             nms[j] = False
        # sink = [candidate[i] for i in range(0, len(candidate)) if nms[i]]
        # '''
        # # 2. overlap -> overlay
        #
        # # 3. return maximum score
        # sink = [] if not len(candidate) else [max(candidate, key=lambda x: x['score'])]
        #
        # # 4. return all candidate
        # # sink = sorted(candidate, key=lambda x: x['score'])
        #
        # return sink


@torch.no_grad()
def extract_videos(model, loader):
    model.eval()
    videos = OrderedDict()
    length = OrderedDict()
    features = []
    bar = tqdm(loader, ncols=200, unit='batch')
    for i, (path, frame) in enumerate(loader):
        out = model(frame)
        features.append(out.cpu().numpy())
        bar.update()
        for p in path:
            vid = os.path.basename(os.path.dirname(p))
            length.setdefault(vid, 0)
            length[vid] += 1
            videos[vid] = vid
    bar.close()
    length = list(length.values())
    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)
    videos = {v: n for n, v in enumerate(videos)}

    return np.concatenate(features), videos, index


def load(path):
    feat = torch.load(path)
    return feat


def load_features(videos, feature_root):
    pool = Pool()
    bar = tqdm(videos, mininterval=1, ncols=150)
    features = [pool.apply_async(load, args=[os.path.join(feature_root, f'{v}.pth')], callback=lambda *a: bar.update())
                for v in videos]
    pool.close()
    pool.join()
    bar.close()
    features = [f.get() for f in features]
    length = [f.shape[0] for f in features]

    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)
    videos = {v: n for n, v in enumerate(videos)}
    return np.concatenate(features), videos, index


def scan_vcdb_annotation(root):
    def parse(ann):
        a, b, *times = ann.strip().split(',')
        times = [sum([60 ** (2 - n) * int(u) for n, u in enumerate(t.split(':'))]) for t in times]
        return [a, b, *times]

    groups = os.listdir(root)
    annotations = defaultdict(list)

    for g in groups:
        f = open(os.path.join(root, g), 'r')
        group = os.path.splitext(g)[0]
        for l in f.readlines():
            a, b, sa, ea, sb, eb = parse(l)
            annotations[a] += [[group, a, b, sa, ea, sb, eb]]
            if a!=b:
                annotations[b] += [[group, b, a, sb, eb, sa, ea]]

    return annotations


if __name__ == '__main__':
    vcdb = np.load('/MLVD/VCDB/meta/vcdb.pkl', allow_pickle=True)
    vcdb_core_video = np.load('/MLVD/VCDB/meta/vcdb_videos.npy')[:528]
    annotation = scan_vcdb_annotation('/MLVD/VCDB/annotation')

    topk = 250
    feature, videos, loc = load_features(vcdb_core_video, '/MLVD/VCDB/mobilenet_rmac/ep23')
    table = {loc[vid][0] + fid: (vid, fid) for v, vid in videos.items() for fid, f in enumerate(vcdb[v])}
    mapping = np.vectorize(lambda x, table: table[x])
    # print(table)
    # print(mapping)

    index = faiss.IndexFlatL2(feature.shape[1])
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(feature)
    for query, gt in annotation.items():
        q_id = videos[query]

        start, end = loc[q_id]
        v_feat = feature[start:end]
        # print(q_id, start, end)
        print(v_feat.shape)
        D, I = index.search(v_feat, topk)

        idx = mapping(I, table)
        tn = TN(D, np.dstack(idx), 5, 5)
        candidate = tn.fit()

        print(candidate)
        print('c',len(candidate))
        print(gt[0][0],[(videos[g[2]], [g[3], g[4]], [g[5], g[6]]) for g in gt])
        print(gt)
        print('g',len(gt))


    # table={(vid,fid):loc[vid][0]+fid for v,vid in videos.items() for fid,f in enumerate(vcdb[v])}
