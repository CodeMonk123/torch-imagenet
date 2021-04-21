from torch.utils.data import dataloader
from torchvision.models.inception import inception_v3
from inception_v4 import inceptionv4
import torch
import torch.distributed as dist
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.models import resnet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import os
from utils import *
import time

model_names = ['alexnet', 'inception_v3', 'resnet50', 'resnet152', 'vgg16', 'inception_v4'] # TODO: implement inception v4

parser = argparse.ArgumentParser(description="Pytorch imagenet distributed training")
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')   
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')         
parser.add_argument('--dist-url', default='tcp://localhost:7890', type=str,
                    help='url used to set up distributed training') 
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 5)')
parser.add_argument('--fast', action='store_true', help='if setted, run only 100 mini batches.' )


best_acc1 = 0
args = parser.parse_args()


def join_process_group():
    print('==> Join process group')
    if dist.is_available() and dist.is_nccl_available():
        dist.init_process_group(backend='nccl',init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        print('==> Process[{}] is ready.'.format(args.rank))
    else:
        raise RuntimeError("Error: Pytorch distributed framework or NCCL is unavailable.")


def main_worker():
    global best_acc1
    join_process_group()
    # create model 
    if args.arch != 'inception_v4':
        model = models.__dict__[args.arch]()
    else:
        model = inceptionv4(num_classes=1000, pretrained=None)
    
    device = torch.device('cuda', 0) # Set reasonable CUDA_VISIBLE_DEVICES 
    model = model.to(device)
    # ddp
    model = nn.parallel.DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # model size
    total_params = sum([torch.numel(p) for p in model.parameters()])
    print('==> Model({}): {:.2f} MB'.format(args.arch, total_params * 4 / (1024 * 1024)))
    
    # optionally resume from a checkpoint
    if args.resume:
        pass # TODO

    cudnn.benchmark = True

    # data loading
    print('==> Create Data Loader')
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  
    
    input_size = 224 if args.arch != 'inception_v3' else 299

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # train & val iteration
    print('==> Train and Val')

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer=optimizer, epoch=epoch, args=args)
        if not args.fast:
            train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, args=args)
        else:
            fast_test(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, args=args)
        # acc1 = validate(val_loader=val_loader, model=model, criterion=criterion, args=args)
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        ## TODO: save checkpoint
        

def fast_test(train_loader, model, criterion, optimizer,  args):
    speed_meter = SpeedMerter()
    model.train()
    end = time.time()
    for i,(images, target) in enumerate(train_loader):
        if i == 100:
            break
        images = images.cuda(0, non_blocking=True)
        target = target.cuda(0, non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed_time = time.time() - end
        speed = args.batch_size * dist.get_world_size() / elapsed_time
        end = time.time()
        speed_meter.update(speed)
        if i % args.print_freq == 0:
            print('batch[{}/100]: {} images/sec'.format(i, speed))
    speed_meter.output()



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(0, non_blocking=True)
        target = target.cuda(0, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        elapsed_time = time.time() - end
        batch_time.update(elapsed_time)
        end = time.time()
        


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    main_worker()