import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.summary import summary
from model.pooling import L2N, GeM, RMAC
import numpy as np
from collections import OrderedDict


class BaseModel(nn.Module):
    def __str__(self):
        return self.__class__.__name__

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return summary(self, input_size, batch_size, device)
        except:
            return self.__repr__()


class MobileNet(BaseModel):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.base = models.mobilenet_v2(pretrained=True)

    def forward(self, x):
        return self.base(x)


class MobileNet_RMAC(BaseModel):
    def __init__(self):
        super(MobileNet_RMAC, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x
class MobileNet_RMAC2(BaseModel):
    def __init__(self):
        super(MobileNet_RMAC2, self).__init__()
        self.base = models.mobilenet_v2(pretrained=True).features
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x



class MobileNet_GeM(BaseModel):
    def __init__(self):
        super(MobileNet_GeM, self).__init__()
        self.base = models.mobilenet_v2(pretrained=True).features
        self.pool = GeM()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class DenseNet(BaseModel):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.base = models.densenet121(pretrained=True)

    def forward(self, x):
        return self.base(x)

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return super().summary(input_size, batch_size, device)
        except:
            return nn.Module.__repr__()


class DenseNet_RMAC(BaseModel):
    def __init__(self):
        super(DenseNet_RMAC, self).__init__()

        self.base = nn.Sequential(*list(models.densenet121(pretrained=True).features.children()),
                                  nn.ReLU(inplace=True))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class DenseNet_GeM(BaseModel):
    def __init__(self):
        super(DenseNet_GeM, self).__init__()
        self.base = nn.Sequential(*list(models.densenet121(pretrained=True).features.children()),
                                  nn.ReLU(inplace=True))
        self.pool = GeM()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class TripletNet(BaseModel):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, *x, single=False):
        if single:
            return self.forward_single(x[0])
        else:
            return self.forward_triple(x[0],x[1],x[2])


    def forward_single(self, x):
        output = self.embedding_net(x)
        return output

    def forward_triple(self, x1,x2,x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

    def __str__(self):
        return f'[{super(TripletNet, self).__str__()}]{self.embedding_net.__str__()}'


if __name__ == '__main__':
    # if architecture.startswith('alexnet'):
    #     features = list(net_in.features.children())[:-1]
    # elif architecture.startswith('vgg'):
    #     features = list(net_in.features.children())[:-1]
    # elif architecture.startswith('resnet'):
    #     features = list(net_in.children())[:-2]
    # elif architecture.startswith('densenet'):
    #     features = list(net_in.features.children())
    #     features.append(nn.ReLU(inplace=True))
    # elif architecture.startswith('squeezenet'):
    #     features = list(net_in.features.children())
    # else:
    #     raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # net = DenseNet_RMAC()
    # print(net)
    # print(net.summary((3, 224, 224), device='cpu'))
    emb = MobileNet()
    net = TripletNet(emb).cuda()
    net = nn.DataParallel(net)
    print(net.module.summary((3, 3, 224, 224), device='cpu'))
    print("====")
    print(emb)
    print(net)
