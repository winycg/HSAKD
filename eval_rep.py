import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np


import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds


from bisect import bisect_right
import time
import math



parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='STL-10', type=str, help='Dataset name')
parser.add_argument('--arch', default='mobilenetV2', type=str, help='network architecture')
parser.add_argument('--s-path', default='ShuffleV1', type=str, help='pre-trained checkpoint dir')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')
parser.add_argument('--milestones', default=[30, 60, 90], type=list, help='milestones for lr-multistep')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number workers')
parser.add_argument('--gpu-id', type=str, default='1')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint dir')


# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
log_txt = 'result/'+ str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch' + '_' +  args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'seed'+ str(args.manual_seed) +'.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch'+ '_' + args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'seed'+ str(args.manual_seed)


args.checkpoint_dir = os.path.join(args.checkpoint_dir, log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


if args.resume is False and args.evaluate is False:
    with open(log_txt, 'a+') as f:
        f.seek(0)
        f.truncate()
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.set_printoptions(precision=4)


if args.dataset == 'STL-10':
    num_classes = 10
    trainset = torchvision.datasets.STL10(root=args.data,  split='train', download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomResizedCrop(32),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ]))

    testset = torchvision.datasets.STL10(root=args.data, split='test', download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(32),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                        pin_memory=(torch.cuda.is_available()))

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                        pin_memory=(torch.cuda.is_available()))

elif args.dataset == 'TinyImageNet':
    num_classes = 200
    train_set = torchvision.datasets.ImageFolder(
    root=os.path.join(args.data, "train"),
    transform=transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]))

    test_set = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data, "val"), 
        transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]))

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
# --------------------------------------------------------------------------------------------
# Model
print('==> Building model..')
net = getattr(models, args.arch)(num_classes=num_classes)
net.eval()
resolution = (1, 3, 32, 32)
print('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.arch, cal_param_size(net)/1e6, cal_multi_adds(net, resolution)/1e9))
del(net)


print('load pre-trained weights from: {}'.format(args.s_path))    

model = getattr(models, args.arch)
net = model(num_classes=num_classes).cuda()
net_dict = net.state_dict()
checkpoint = torch.load(args.s_path,
                        map_location=torch.device('cpu'))['net']
for key in net.state_dict().keys():
    if key.startswith('classifier'):
        continue
    net_dict[key] = checkpoint['backbone.'+key]


net.load_state_dict(net_dict)
optimizer = optim.SGD(net.classifier.parameters(), lr=args.init_lr,
                      momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
net.eval()
for name, p in net.named_parameters():
    if name.startswith('classifier') is False:
        p.requires_grad = False

net = torch.nn.DataParallel(net)
cudnn.benchmark = True


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0, eta_min=0.):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch*all_iters_per_epoch)/(args.warmup_epoch *all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = cur_lr
    return cur_lr


# Training
def train(epoch, criterion_list, optimizer):
    train_loss = 0.
    train_loss_cls = 0.

    top1_num = 0
    top5_num = 0
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    net.module.classifier.train()
    for batch_idx, (input, target) in enumerate(trainloader):
        batch_start_time = time.time()
        input = input.float().cuda()
        target = target.cuda()

        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        optimizer.zero_grad()
        logits = net(input)

        loss_cls = criterion_cls(logits, target)

        loss = loss_cls
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(trainloader)
        train_loss_cls += loss_cls.item() / len(trainloader)

        top1, top5 = correct_num(logits, target, topk=(1, 5))
        top1_num += top1
        top5_num += top5

        total += target.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}'.format(epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time))

    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.4f}\t duration:{:.3f}'
                '\n train_loss:{:.5f}\t train_loss_cls:{:.5f}'
                '\nTrain Top-1 accuracy: {}\n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_cls,
                        str(top1_num.item()/total)))



def test(epoch, criterion_cls):
    global best_acc
    test_loss_cls = 0.
    
    top1_num = 0
    top5_num = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            batch_start_time = time.time()
            input, target = inputs.cuda(), target.cuda()
            
            logits = net(input)
            loss_cls = torch.tensor(0.).cuda()
            loss_cls = criterion_cls(logits, target)

            test_loss_cls += loss_cls.item()/ len(testloader)

            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)
            

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}'.format(epoch, batch_idx, len(testloader), time.time()-batch_start_time))

        with open(log_txt, 'a+') as f:
            f.write('test epoch:{}\t test_loss_cls:{:.5f}\nTest Top-1 accuracy: {}\n'
                    .format(epoch, test_loss_cls, str(top1_num.item()/total)))

    return top1_num.item()/total


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)

    if args.evaluate: 
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))     
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                map_location=torch.device('cpu'))
        exit()
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        test(start_epoch, criterion_cls)
    else:
        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.cuda()

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls)
            if acc > best_acc:
                best_acc = acc
            
        with open(log_txt, 'a+') as f:
            f.write('best_acc:{}'.format(best_acc))
        print('best_acc:{}'.format(best_acc))
