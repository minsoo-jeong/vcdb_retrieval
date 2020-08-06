import os
import json
import csv
import pandas as pd
from pymediainfo import MediaInfo
import numpy as np
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    # videos = np.load('/MLVD/VCDB/meta/vcdb_videos.npy')
    # video_root = '/MLVD/VCDB/videos'
    # save = '/MLVD/VCDB/meta/vcdb_meta.csv'
    videos = np.load('/hdd/FIVR_core/fivr_videos.npy')
    video_root = '/MLVD/FIVR/videos'
    save = 'fivr_meta.csv'
    print(len(videos))
    data = []
    for v in tqdm(videos):
        media_info = MediaInfo.parse(os.path.join(video_root, v))
        meta = dict()
        for track in media_info.tracks:
            # print(track.to_data())
            if track.track_type == 'General':
                meta['file_name'] = track.file_name + '.' + track.file_extension
                meta['file_extension'] = track.file_extension
                meta['format'] = track.format
                meta['duration'] = track.duration
                meta['frame_count'] = track.frame_count
                meta['frame_rate'] = track.frame_rate
            elif track.track_type == 'Video':
                meta['width'] = int(track.width)
                meta['height'] = int(track.height)
                meta['rotation'] = float(track.rotation) if track.rotation is not None else 0.
                meta['codec'] = track.codec

        data.append(meta)

    df = pd.DataFrame(data, columns=['file_name', 'file_extension', 'format', 'codec', 'width', 'height', 'rotation',
                                     'frame_rate', 'frame_count', 'duration'])
    df.to_csv(save, index=False)
