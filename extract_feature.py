import numpy as np
import torch.nn as nn
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from torchvision.transforms import transforms as trn

from model.nets import MobileNet, MobileNet_RMAC
from dataset.loader import ListDataset

@torch.no_grad()
def extract_list(model, loader, save_to):
    bar = tqdm(loader, ncols=150)
    for idx, (paths, frames) in enumerate(bar):
        features = model(frames.cuda()).cpu()
        for i, p in enumerate(paths):
            vid = p.split('/')[-2]
            no = os.path.basename(p)
            to = os.path.join(save_to, vid)
            if not os.path.exists(to):
                os.makedirs(to)
            np.save(os.path.join(to, f'{no}.npy'), features[i:i + 1])
    bar.close()
    videos = os.listdir(save_to)
    bar2 = tqdm(videos, ncols=150)

    for vi, v in enumerate(bar2):
        f_feat = [np.load(os.path.join(save_to, v, f)) for f in sorted(os.listdir(os.path.join(save_to, v)))]
        v_feat = np.concatenate(f_feat)
        if len(v_feat.shape) == 4:
            v_feat = v_feat.squeeze(-1).squeeze(-1)
        np.save(os.path.join(save_to, f'{v}.npy'), v_feat)
        shutil.rmtree(os.path.join(save_to, v))

    bar2.close()

@torch.no_grad()
def extract_videos(model, loader, frames, save_to, root='/MLVD/VCDB/frames/'):
    model.eval()
    bar = tqdm(frames.keys())
    for v, fl in frames.items():
        bar.set_description_str(v)
        loader.dataset.l = [os.path.join(root, v, f) for f in fl]
        vf = []
        for idx, (p, f) in enumerate(loader):
            f = model(f.cuda())
            vf.append(f.cpu())
        vf = torch.cat(vf)
        torch.save(vf, os.path.join(save_to, v + '.pth'))

        bar.update()


if __name__ == '__main__':
    import os
    net = MobileNet_RMAC().cuda()
    # save_to = '/MLVD/VCDB/mobilenet/rmac'
    # vcdb = np.load('/MLVD/VCDB/meta/vcdb.pkl', allow_pickle=True)
    # videos = np.load('/MLVD/VCDB/meta/vcdb_videos.npy')[:10529]
    # frame_root = '/MLVD/VCDB/frames'
    # frames = {v: vcdb[v] for v in videos}

    save_to = '/hdd/FIVR_core/mobilenet/rmac'
    fivr = np.load('/MLVD/FIVR/meta/fivr.pkl', allow_pickle=True)
    videos = np.load('/MLVD/FIVR/meta/fivr_videos.npy')[:19120]
    frame_root = '/MLVD/FIVR/frames'
    frames = {v: fivr[v] for v in videos}

    net = nn.DataParallel(net)
    batch = 256
    net.eval()

    transform = None

    loader = DataLoader(ListDataset([], transform=transform), batch_size=batch, shuffle=False, num_workers=4)
    extract_videos(net, loader, frames, save_to, root=frame_root)
