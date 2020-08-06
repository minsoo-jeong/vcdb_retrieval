import os
import faiss
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager

from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.detectors import ThresholdDetector
import csv
import pandas as pd
import torch
import json


def detect_scene(path, fps, min_sec):
    video_manager = VideoManager([path], framerate=fps)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=30, min_scene_len=fps*min_sec))
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=False)
    scene_list = scene_manager.get_scene_list(base_timecode)

    # Like FrameTimecodes, each scene in the scene_list can be sorted if the
    # list of scenes becomes unsorted.

    # print('List of scenes obtained:')
    # for i, scene in enumerate(scene_list):
    #     print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
    #         i + 1,
    #         scene[0].get_timecode(), scene[0].get_frames(),
    #         scene[1].get_timecode(), scene[1].get_frames(),))
    scene_list = [(round(s[0].get_seconds()), round(s[1].get_seconds())) for s in scene_list]
    video_manager.release()
    return scene_list


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


if __name__ == '__main__':
    root = '/MLVD/FIVR'
    feature_root=f'{root}/mobilenet/rmac'
    frame_root=f'{root}/frames'
    video_root=f'{root}/videos'
    metafile='/MLVD/FIVR/meta/fivr_meta.csv'
    annotation = json.load(open('/MLVD/FIVR/annotation/annotation.json', 'r'))

    fivr=np.load('/MLVD/FIVR/meta/fivr.pkl',allow_pickle=True)
    annotation = {k: [r for kk, rr in ref.items() if kk != 'IS' for r in rr] for k, ref in annotation.items()}
    print(annotation)

    videos = np.load('/MLVD/FIVR/meta/fivr_videos_core.npy')
    fivr_meta_core = {r['file_name']: dict(r) for r in csv.DictReader(open(metafile, "r")) if r['file_name'] in videos}
    features_path = [f'{feature_root}/{v}.pth' for v in videos]
    features, length, index = load_feature(features_path)

    frame_index = {v: index[n] for n, v in enumerate(videos)}
    frame_count = {v: length[n] for n, v in enumerate(videos)}
    print(videos)

    fivr_frames = {v: fivr[v] for v in videos}

    ret = []
    p_c = 0
    bar=tqdm(annotation.keys())
    for cnt, v in enumerate(bar):
        # print('cnt : ', cnt)

        path = f'{root}/videos/{v}'
        fps = float(fivr_meta_core[v]['frame_rate'])

        scene_idx = detect_scene(path, fps, 5)
        # print('a : ', path, fps, v, scene_idx)

        a_idx = frame_index[v]
        a_feat = features[a_idx[0]:a_idx[1], ]
        for r in annotation[v]:
            b_idx = frame_index[r]
            b_feat = features[b_idx[0]:b_idx[1], ]
            # print('b : ', r, b_idx, b_feat.shape)
            fi = faiss.IndexFlatL2(b_feat.shape[1])
            # fi = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(b_feat.shape[1]))
            fi.add(b_feat)
            d, i = fi.search(a_feat, 5)

            for scene in scene_idx:
                start, end = scene
                dist = d[start:end]
                index = i[start:end]

                min_idx = index[:, 0]
                min_dist = dist[:, 0]

                select = np.sort(np.argsort(min_dist)[:1])

                candidate_dist = dist[select,]
                candidate = index[select]
                a_select = select + start

                a_frame = fivr_frames[v][a_select]
                b_frame = np.array([fivr_frames[r][c] for c in candidate])

                ret += [[v, scene[0], scene[1], a_select[n], a, r, *candidate[n], *b_frame[n], *candidate_dist[n]] for
                        n, a in enumerate(a_frame)]
        bar.set_description(f'{len(ret)}, {v}')
    ret = [[n, *r] for n, r in enumerate(ret)]
    df = pd.DataFrame(ret,
                      columns=['idx', 'a', 'scene_start', 'scene_end', 'a_frame_idx', 'a_frame', 'b',
                               'b_frame_idx_1', 'b_frame_idx_2', 'b_frame_idx_3', 'b_frame_idx_4', 'b_frame_idx_5',
                               'b_frame_1', 'b_frame_2', 'b_frame_3', 'b_frame_4', 'b_frame_5',
                               'dist_1', 'dist_2', 'dist_3', 'dist_4', 'dist_5'])

    writer = pd.ExcelWriter('fivr_positive_candidate.xlsx', engine='xlsxwriter')
    df.to_excel(writer, "xlsxwriter", index=False)
    writer.save()
