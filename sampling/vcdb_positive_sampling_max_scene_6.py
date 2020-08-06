import os
import numpy as np
import pickle as pkl
import torch
from tqdm import tqdm
from multiprocessing import Pool
import csv
import faiss
import pandas as pd

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager

from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


def save_vcdb_videos():
    vcdb = np.load('/MLVD/VCDB/vcdb.pkl', allow_pickle=True)
    vcdb_videos = (list(vcdb.keys()))

    vcdb_videos_core = sorted([i for i in vcdb_videos if len(i) > 40])
    np.save('/MLVD/VCDB/vcdb_videos_core.npy', np.array(vcdb_videos_core))
    vcdb_core = {v: sorted(vcdb[v]) for v in vcdb_videos_core}
    pkl.dump(vcdb_core, open('/MLVD/VCDB/vcdb_core.pkl', 'wb'))
    print(vcdb_videos_core)
    print(len(vcdb_videos_core))

    vcdb_videos_bg = sorted([i for i in vcdb_videos if i not in vcdb_videos_core])
    np.save('/MLVD/VCDB/vcdb_videos_bg.npy', np.array(vcdb_videos_bg))
    vcdb_bg = {v: sorted(vcdb[v]) for v in vcdb_videos_bg}
    pkl.dump(vcdb_bg, open('/MLVD/VCDB/vcdb_bg.pkl', 'wb'))
    print(len(vcdb_videos_bg))

    vcdb_videos = vcdb_videos_core + vcdb_videos_bg
    np.save('/MLVD/VCDB/vcdb_videos.npy', np.array(vcdb_videos))
    vcdb = {v: sorted(vcdb[v]) for v in vcdb_videos}
    pkl.dump(vcdb, open('/MLVD/VCDB/vcdb.pkl', 'wb'))


def scan_vcdb_annotation(root):
    def parse(ann):
        a, b, *times = ann.strip().split(',')
        times = [sum([60 ** (2 - n) * int(u) for n, u in enumerate(t.split(':'))]) for t in times]
        return [a, b, *times]

    groups = os.listdir(root)
    annotations = []
    for g in groups:
        f = open(os.path.join(root, g), 'r')
        annotations += [[os.path.splitext(g)[0], *parse(l)] for l in f.readlines()]
    return annotations


def load(p):
    # f = np.load(p)
    f = torch.load(p).numpy()
    return f, f.shape[0]


def load_feature(paths):
    # load = lambda p: np.load(p)
    bar = tqdm(paths, ncols=150, desc='Load Features')
    pool = Pool()
    results = [pool.apply_async(load, args=[p], callback=lambda *a: bar.update()) for p in paths]
    pool.close()
    pool.join()
    results = [(f.get()) for f in results]
    features = np.concatenate([r[0] for r in results])
    length = [r[1] for r in results]

    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    bar.close()
    return features, length, index


def detect_scene(path, fps, min_sec, thresh=30, start=None, end=None):
    video_manager = VideoManager([path], framerate=fps)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=thresh, min_scene_len=fps * min_sec))
    base_timecode = video_manager.get_base_timecode()
    if start != None and end != None:
        start_time = base_timecode + float(start)
        end_time = base_timecode + float(end)
        video_manager.set_duration(start_time=start_time, end_time=end_time)

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=False)
    scene_list = scene_manager.get_scene_list(base_timecode)

    scene_list = [(round(s[0].get_seconds()), round(s[1].get_seconds())) for s in scene_list]
    video_manager.release()
    return scene_list


if __name__ == '__main__':

    video_root = '/workspace/VCDB_core/videos'
    frame_root = '/MLVD/VCDB/frames'
    vcdb = np.load('/MLVD/VCDB/meta/vcdb.pkl', allow_pickle=True)
    vcdb_core_videos = np.load('/MLVD/VCDB/meta/vcdb_videos_core.npy')
    annotation = scan_vcdb_annotation('/MLVD/VCDB/annotation')
    meta = {m['file_name']: dict(m) for m in csv.DictReader(open('/MLVD/VCDB/meta/vcdb_meta.csv', "r"))}

    features_path = [f'/MLVD/VCDB/mobilenet/rmac/{v}.pth' for v in vcdb_core_videos]
    features, length, pos = load_feature(features_path)
    feature_pos = {v: pos[n] for n, v in enumerate(vcdb_core_videos)}
    print(features.shape)

    ret = []
    bar = tqdm(annotation)
    for ann_n, ann in enumerate(bar):
        group, a, b, a_start, a_end, b_start, b_end = ann

        if a != b:
            a_path, a_meta = os.path.join(video_root, a), meta[a]
            b_path, b_meta = os.path.join(video_root, b), meta[b]

            a_feat = features[feature_pos[a][0] + a_start:feature_pos[a][0] + a_end]
            b_feat = features[feature_pos[b][0] + b_start:feature_pos[b][0] + b_end]

            b_feat_index = faiss.IndexFlatL2(b_feat.shape[1])
            b_feat_index.add(b_feat)
            d, i = b_feat_index.search(a_feat, 5)
            thresh = 30
            a_scene = detect_scene(a_path, float(a_meta['frame_rate']), 5, thresh=thresh, start=a_start, end=a_end + 1)
            if len(a_scene) > 6:
                a_scene = sorted(a_scene, key=lambda x: x[1] - x[0], reverse=True)[:6]

            for scene in a_scene:
                scene_start, scene_end = scene[0] - a_start, scene[1] - a_start

                dist = d[scene_start:scene_end]
                index = i[scene_start:scene_end]

                min_idx = index[:, 0]
                min_dist = dist[:, 0]
                select = np.sort(np.argsort(min_dist)[:1])
                candidate_dist = dist[select,]
                # candidate = index[select] + b_start
                candidate = [[s + b_start if s != -1 else s for s in row] for row in index[select]]

                a_select = select + scene[0]

                a_frame = np.array(vcdb[a])[a_select]
                b_frame = np.array([np.array(vcdb[b])[c] for c in candidate])
                ret += [[ann_n, group, a, a_start, a_end, b, b_start, b_end, *scene, a_select[n], a_frame[n],
                         *candidate[n], *b_frame[n],
                         *candidate_dist[n]] for n, af in enumerate(a_frame)]

            a_feat_index = faiss.IndexFlatL2(a_feat.shape[1])
            a_feat_index.add(a_feat)
            d, i = a_feat_index.search(b_feat, 5)
            thresh = 30
            b_scene = detect_scene(b_path, float(b_meta['frame_rate']), 5, thresh=thresh, start=b_start, end=b_end + 1)
            if len(b_scene) > 6:
                b_scene = sorted(b_scene, key=lambda x: x[1] - x[0], reverse=True)[:6]

            for scene in b_scene:
                scene_start, scene_end = scene[0] - b_start, scene[1] - b_start
                dist = d[scene_start:scene_end]
                index = i[scene_start:scene_end]

                min_idx = index[:, 0]
                min_dist = dist[:, 0]
                select = np.sort(np.argsort(min_dist)[:1])
                candidate_dist = dist[select,]
                # candidate = index[select] + b_start
                candidate = [[s + a_start if s != -1 else s for s in row] for row in index[select]]
                b_select = select + scene[0]

                b_frame = np.array(vcdb[b])[b_select]
                a_frame = np.array([np.array(vcdb[a])[c] for c in candidate])
                ret += [
                    [ann_n, group, b, b_start, b_end, a, a_start, a_end, *scene, b_select[n], b_frame[n], *candidate[n],
                     *a_frame[n], *candidate_dist[n]] for n, bf in enumerate(b_frame)]
        bar.set_description(f'{len(ret)}, {group}, {a}, {b}')
    df = pd.DataFrame(ret,
                      columns=['ann_idx', 'group', 'a', 'a_start', 'a_end', 'b', 'b_start', 'b_end', 'a_scene_start',
                               'a_scene_end',
                               'a_frame_idx', 'a_frame',
                               'b_frame_idx_1', 'b_frame_idx_2', 'b_frame_idx_3', 'b_frame_idx_4', 'b_frame_idx_5',
                               'b_frame_1', 'b_frame_2', 'b_frame_3', 'b_frame_4', 'b_frame_5',
                               'dist_1', 'dist_2', 'dist_3', 'dist_4', 'dist_5'])

    writer = pd.ExcelWriter('vcdb_positive_candidate_4.xlsx', engine='xlsxwriter')
    df.to_excel(writer, "xlsxwriter", index=False)
    writer.save()
