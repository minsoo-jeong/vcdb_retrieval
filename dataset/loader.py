import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as trn

try:
    import accimage
except ImportError:
    accimage = None


class ListDataset(Dataset):
    def __init__(self, l, transform=None):
        self.l = l
        self.loader = default_loader  # self.feature_loader
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

    def __getitem__(self, idx):
        path = self.l[idx]
        frame = self.transform(self.loader(path))

        return path, frame

    def __len__(self):
        return len(self.l)


class TripletDataset(Dataset):
    def __init__(self, triplets, base, transform=None):
        self.base = base
        self.triplets = triplets
        self.loader = default_loader  # self.feature_loader
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

    def __getitem__(self, idx):
        a, af, p, pf, n, nf = self.triplets[idx]
        anc_path = os.path.join(self.base, a, af)
        pos_path = os.path.join(self.base, p, pf)
        neg_path = os.path.join(self.base, n, nf)
        anc = self.transform(self.loader(anc_path))
        pos = self.transform(self.loader(pos_path))
        neg = self.transform(self.loader(neg_path))

        return (anc_path, pos_path, neg_path), (anc, pos, neg)

    def __len__(self):
        return len(self.triplets)

class PairDataset(Dataset):
    def __init__(self, pairs, base, transform=None):
        self.base = base
        self.pairs = pairs
        self.loader = default_loader  # self.feature_loader
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

    def __getitem__(self, idx):
        a, af, p, pf = self.pairs[idx]
        anc_path = os.path.join(self.base, a, af)
        pos_path = os.path.join(self.base, p, pf)
        anc = self.transform(self.loader(anc_path))
        pos = self.transform(self.loader(pos_path))

        return (anc_path, pos_path), (anc, pos,)

    def __len__(self):
        return len(self.pairs)

if __name__ == '__main__':
    print(ListDataset(['aaa']))
