import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

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
parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet18_imagenet_aux', type=str, help='student network architecture')
parser.add_argument('--tarch', default='resnet34_imagenet_aux', type=str, help='teacher network architecture')
parser.add_argument('--tcheckpoint', default='resnet34_imagenet_aux.pth.tar', type=str, help='pre-trained weights of teacher')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--milestones', default=[30,60,90], type=list, help='milestones for lr-multistep')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory fot storing checkpoints')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')                    


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    args.log_txt = 'result/'+ str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch' + '_' +  args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'seed'+ str(args.manual_seed) +'.txt'


    args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch'+ '_' + args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'seed'+ str(args.manual_seed)


    args.traindir = os.path.join(args.data, 'train')
    args.valdir = os.path.join(args.data, 'val')

    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.set_printoptions(precision=4)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(args.log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')

    print('==> Building model..')
    num_classes = 1000

    net = getattr(models, args.tarch)(num_classes=num_classes)
    net.eval()
    print('Teacher Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.tarch, cal_param_size(net)/1e6, cal_multi_adds(net, (1, 3, 224, 224))/1e9))
    del(net)

    net = getattr(models, args.arch)(num_classes=num_classes)
    net.eval()
    print('Student Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.tarch, cal_param_size(net)/1e6, cal_multi_adds(net, (1, 3, 224, 224))/1e9))
    del(net)

    args.distributed = args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node)
    args.world_size = ngpus_per_node * args.world_size
    print('multiprocessing_distributed')
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                        world_size=args.world_size, rank=args.rank)

    print('load pre-trained teacher weights from: {}'.format(args.tcheckpoint))     
    checkpoint = torch.load(args.tcheckpoint, map_location='cuda:{}'.format(args.gpu))
    torch.cuda.set_device(args.gpu)

    num_classes = 1000
    model = getattr(models, args.arch)
    net = model(num_classes=num_classes).cuda(args.gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

    tmodel = getattr(models, args.tarch)
    tnet = tmodel(num_classes=num_classes).cuda(args.gpu)
    tnet = torch.nn.parallel.DistributedDataParallel(tnet, device_ids=[args.gpu])
    tnet.module.load_state_dict(checkpoint['net'])
    tnet.eval()

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    _, ss_logits = net(torch.randn(2, 3, 224, 224))
    num_auxiliary_branches = len(ss_logits)
    cudnn.benchmark = True

    criterion_cls = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_div = DistillKL(args.kd_T).cuda(args.gpu)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(net)
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=args.weight_decay)


    if args.resume:
        print('load intermediate weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1


    train_set = torchvision.datasets.ImageFolder(
    args.traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    test_set = torchvision.datasets.ImageFolder(
        args.valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ]))

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    
    def train(epoch, criterion_list, optimizer):
        train_loss = 0.
        train_loss_cls = 0.
        train_loss_div = 0.
        train_loss_kd = 0.

        ss_top1_num = [0] * num_auxiliary_branches
        ss_top5_num = [0] * num_auxiliary_branches
        class_top1_num = [0] * num_auxiliary_branches
        class_top5_num = [0] * num_auxiliary_branches
        top1_num = 0
        top5_num = 0
        total = 0

        if epoch >= args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args)

        start_time = time.time()
        criterion_cls = criterion_list[0]
        criterion_div = criterion_list[1]

        net.train()
        for batch_idx, (input, target) in enumerate(trainloader):
            batch_start_time = time.time()
            input = input.float().cuda()
            target = target.cuda()

            size = input.shape[1:]
            input = torch.stack([torch.rot90(input, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
            labels = torch.stack([target*4+i for i in range(4)], 1).view(-1)

            if epoch < args.warmup_epoch:
                lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

            optimizer.zero_grad()
            logits, ss_logits = net(input, grad=True)
            with torch.no_grad():
                t_logits, t_ss_logits = tnet(input)

            loss_cls = torch.tensor(0.).cuda()
            loss_div = torch.tensor(0.).cuda()

            loss_cls = loss_cls + criterion_cls(logits[0::4], target)
            
            for i in range(len(ss_logits)):
                loss_div = loss_div + criterion_div(ss_logits[i], t_ss_logits[i].detach())
            
            loss_div = loss_div + criterion_div(logits, t_logits.detach())
            
                
            loss = loss_cls + loss_div
            loss.backward()
            optimizer.step()


            train_loss += loss.item() / len(trainloader)
            train_loss_cls += loss_cls.item() / len(trainloader)
            train_loss_div += loss_div.item() / len(trainloader)

            for i in range(len(ss_logits)):
                top1, top5 = correct_num(ss_logits[i], labels, topk=(1, 5))
                ss_top1_num[i] += top1
                ss_top5_num[i] += top5
            
            class_logits = [torch.stack(torch.split(ss_logits[i], split_size_or_sections=4, dim=1), dim=1).sum(dim=2) for i in range(len(ss_logits))]
            multi_target = target.view(-1, 1).repeat(1, 4).view(-1)
            for i in range(len(class_logits)):
                top1, top5 = correct_num(class_logits[i], multi_target, topk=(1, 5))
                class_top1_num[i] += top1
                class_top5_num[i] += top5

            logits = logits.view(-1, 4, num_classes)[:, 0, :]
            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)

            print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, (top1_num/(total)).item()))


        ss_acc1 = [round((ss_top1_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)]
        ss_acc5 = [round((ss_top5_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)]
        class_acc1 = [round((class_top1_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)] + [round((top1_num/(total)).item(), 4)]
        class_acc5 = [round((class_top5_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)] + [round((top5_num/(total)).item(), 4)]

        with open(args.log_txt, 'a+') as f:
            f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                    '\n train_loss:{:.5f}\t train_loss_cls:{:.5f}\t train_loss_div:{:.5f}'
                    '\nTrain Top-1 ss_accuracy: {}\nTrain Top-5 ss_accuracy: {}\n'
                    'Train Top-1 class_accuracy: {}\nTrain Top-5 class_accuracy: {}\n'
                    .format(epoch, lr, time.time() - start_time,
                            train_loss, train_loss_cls, train_loss_div,
                            str(ss_acc1), str(ss_acc5), str(class_acc1), str(class_acc5)))


    def test(epoch, criterion_cls, net):
        test_loss_cls = 0.

        ss_top1_num = [0] * (num_auxiliary_branches)
        ss_top5_num = [0] * (num_auxiliary_branches)
        class_top1_num = [0] * num_auxiliary_branches
        class_top5_num = [0] * num_auxiliary_branches
        top1_num = 0
        top5_num = 0
        total = 0
        
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(testloader):
                batch_start_time = time.time()
                input, target = inputs.cuda(), target.cuda()

                size = input.shape[1:]
                input = torch.stack([torch.rot90(input, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
                labels = torch.stack([target*4+i for i in range(4)], 1).view(-1)
                
                logits, ss_logits = net(input)
                loss_cls = torch.tensor(0.).cuda()
                for i in range(len(ss_logits)):
                    loss_cls = loss_cls + criterion_cls(ss_logits[i], labels)

                test_loss_cls += loss_cls.item()/ len(testloader)

                batch_size = logits.size(0) // 4
                for i in range(len(ss_logits)):
                    top1, top5 = correct_num(ss_logits[i], labels, topk=(1, 5))
                    ss_top1_num[i] += top1
                    ss_top5_num[i] += top5
                    
                class_logits = [torch.stack(torch.split(ss_logits[i], split_size_or_sections=4, dim=1), dim=1).sum(dim=2) for i in range(len(ss_logits))]
                multi_target = target.view(-1, 1).repeat(1, 4).view(-1)
                for i in range(len(class_logits)):
                    top1, top5 = correct_num(class_logits[i], multi_target, topk=(1, 5))
                    class_top1_num[i] += top1
                    class_top5_num[i] += top5

                logits = logits.view(-1, 4, num_classes)[:, 0, :]
                top1, top5 = correct_num(logits, target, topk=(1, 5))
                top1_num += top1
                top5_num += top5
                total += target.size(0)
                

                print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(testloader), time.time()-batch_start_time, (top1_num/(total)).item()))

            ss_acc1 = [round((ss_top1_num[i]/(total*4)).item(), 4) for i in range(len(ss_logits))]
            ss_acc5 = [round((ss_top5_num[i]/(total*4)).item(), 4) for i in range(len(ss_logits))]
            class_acc1 = [round((class_top1_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)] + [round((top1_num/(total)).item(), 4)]
            class_acc5 = [round((class_top5_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)] + [round((top5_num/(total)).item(), 4)]
            with open(args.log_txt, 'a+') as f:
                f.write('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 ss_accuracy: {}\nTop-5 ss_accuracy: {}\nTop-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'
                        .format(epoch, test_loss_cls, str(ss_acc1), str(ss_acc5), str(class_acc1), str(class_acc5)))
            print('test epoch:{}\nTest Top-1 ss_accuracy: {}\nTest Top-5 ss_accuracy: {}\nTest Top-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'.format(
                epoch, str(ss_acc1), str(ss_acc5), str(class_acc1), str(class_acc5)))

        return class_acc1[-1]

    if args.evaluate: 
        print('load pre-trained weights from: {}'.format(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))     
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        test(start_epoch, criterion_cls, net) 
    else:
        print('Evaluate Teacher:')
        acc = test(0, criterion_cls, tnet)
        print('teacher accuracy:{}'.format(acc))
        with open(args.log_txt, 'a+') as f:
            f.write('teacher accuracy:{}'.format(acc))

        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls, net)

            if args.rank == 0:
                state = {
                    'net': net.module.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

                is_best = False
                if best_acc < acc:
                    best_acc = acc
                    is_best = True

                if is_best:
                    shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls, net)

        with open(args.log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))
        os.system('cp ' + args.log_txt + ' ' + args.checkpoint_dir)


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

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

if __name__ == '__main__':
    main()
    
        


        

        
