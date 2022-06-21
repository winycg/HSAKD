import sys
import os

sys.path.append('.')
sys.path.append(os.getcwd())
import torch 
import torch.nn as nn


from .resnet import *
from .resnetv2 import *
from .wrn import *
from .vgg import *
from .ShuffleNetv1 import *
from .ShuffleNetv2 import *
from .mobilenetv2 import *
from .resnet_imagenet import *


__all__ = ['mbv2_mbv2', 'WRN_40_2_WRN_40_2', 'WRN_40_1_WRN_40_1', 'resnet56_resnet56',
           'shufflev1_shufflev1', 'shufflev2_shufflev2', 
           'resnet18_resnet18',  'resnet32x4_resnet32x4',
           'vgg13_vgg13']


class mbv2_mbv2(nn.Module):
    def __init__(self,num_classes):
        super(mbv2_mbv2, self).__init__()
        self.net1 = mobilenetV2_aux(num_classes)
        self.net2 = mobilenetV2_aux(num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]


class WRN_40_2_WRN_40_2(nn.Module):
    def __init__(self,num_classes):
        super(WRN_40_2_WRN_40_2, self).__init__()
        self.net1 = wrn_40_2_aux(num_classes=num_classes)
        self.net2 = wrn_40_2_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]


class WRN_16_2_WRN_16_2(nn.Module):
    def __init__(self,num_classes):
        super(WRN_16_2_WRN_16_2, self).__init__()
        self.net1 = wrn_16_2_aux(num_classes=num_classes)
        self.net2 = wrn_16_2_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]


class resnet20_resnet20(nn.Module):
    def __init__(self,num_classes):
        super(resnet20_resnet20, self).__init__()
        self.net1 = resnet20_aux(num_classes=num_classes)
        self.net2 = resnet20_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]


class resnet56_resnet56(nn.Module):
    def __init__(self,num_classes):
        super(resnet56_resnet56, self).__init__()
        self.net1 = resnet56_aux(num_classes=num_classes)
        self.net2 = resnet56_aux(num_classes=num_classes)


    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]


class resnet32x4_resnet32x4(nn.Module):
    def __init__(self,num_classes):
        super(resnet32x4_resnet32x4, self).__init__()
        self.net1 = resnet32x4_aux(num_classes=num_classes)
        self.net2 = resnet32x4_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]


class resnet8x4_resnet8x4(nn.Module):
    def __init__(self,num_classes):
        super(resnet8x4_resnet8x4, self).__init__()
        self.net1 = resnet8x4_aux(num_classes=num_classes)
        self.net2 = resnet8x4_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]

class vgg13_vgg13(nn.Module):
    def __init__(self,num_classes):
        super(vgg13_vgg13, self).__init__()
        self.net1 = vgg13_bn_aux(num_classes=num_classes)
        self.net2 = vgg13_bn_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]


class shufflev1_shufflev1(nn.Module):
    def __init__(self,num_classes):
        super(shufflev1_shufflev1, self).__init__()
        self.net1 = ShuffleV1_aux(num_classes=num_classes)
        self.net2 = ShuffleV1_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]


class shufflev2_shufflev2(nn.Module):
    def __init__(self,num_classes):
        super(shufflev2_shufflev2, self).__init__()
        self.net1 = ShuffleV2_aux(num_classes=num_classes)
        self.net2 = ShuffleV2_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]


class WRN_40_1_WRN_40_1(nn.Module):
    def __init__(self,num_classes):
        super(WRN_40_1_WRN_40_1, self).__init__()
        self.net1 = wrn_40_1_aux(num_classes=num_classes)
        self.net2 = wrn_40_1_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]




class resnet18_resnet18(nn.Module):
    def __init__(self,num_classes):
        super(resnet18_resnet18, self).__init__()
        self.net1 = resnet18_imagenet_aux(num_classes=num_classes)
        self.net2 = resnet18_imagenet_aux(num_classes=num_classes)

    def forward(self, x, grad=True):
        logit1, ss_logits1 = self.net1(x, grad=grad)
        logit2, ss_logits2 = self.net2(x, grad=grad)
        return [logit1, logit2], [ss_logits1, ss_logits2]



if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = mbv2_mbv2(100)
    logit, ss_logits = net(x)
    print(logit)
    print(ss_logits)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))