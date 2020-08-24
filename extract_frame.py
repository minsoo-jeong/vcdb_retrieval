import os, subprocess
import tqdm
from multiprocessing import Pool
import time
import shutil
from datetime import datetime, timedelta, timezone
import logging
import numpy as np


def extract(video, frame_dir, cnt):
    cmd = 'ffmpeg -i {} -map 0:v:0 -q:v 0 -vsync 2 -vf fps=2 -f image2 {}/%6d.jpg'.format(video, frame_dir).split(' ')

    p = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    out, err = p.communicate()

    return (True if (p.returncode != 1 and len(os.listdir(frame_dir)) != 0) else False, video, cnt)


def update(bar, ret, logger):
    code, video, no = ret
    bar.set_description_str(f'{video}')
    bar.update()
    if not code:
        msg = f'[{datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M:%S")}] Fail {no}: {video}'
        bar.write(msg)
        logger.info(msg)

    if no % 1000 == 0:
        msg = f'[{datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M:%S")}] Process {no} videos'
        bar.write(msg)
        logger.info(msg)


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler('extract_frames.log'))
    logger.setLevel(level=logging.INFO)
    kst = timezone(timedelta(hours=9))

    vr = '/MLVD/VCDB/videos'
    fr = '/MLVD/VCDB/frames2'
    videos = np.load('/MLVD/VCDB/meta/vcdb_videos.npy')
    print(videos)

    bar = tqdm.tqdm(videos, mininterval=.1, ncols=150)
    pool = Pool()
    results = []

    for c, v in enumerate(videos, start=1):
        video = os.path.join(vr, v)
        frame_dir = os.path.join(fr, v)
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        else:
            pass # remove
        pool.apply_async(extract, args=[video, frame_dir, c], callback=lambda ret: update(bar, ret, logger))


    pool.close()
    pool.join()
    bar.close()
    print('finish')
