import argparse
import os
import shutil
import time

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.parallel

import models.nonlocal_net as i3d
import transforms as t
from datasets.VideoDataset import VideoDataset


def main():
    global args, best_prec1
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', type=str)
    parser.add_argument('--root_data_path', type=str,
                        default="/media/v-pakova/New Volume/Datasets/Kinetics/400/val_frames_256")
    parser.add_argument('--base_model', type=str, default="resnet50")
    parser.add_argument('--weights_file', type=str, default="/media/v-pakova/New Volume1/"
                        "OnlineActionRecognition/models/pre-trained/resnet50_nl_i3d_kinetics.pt")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--sample_frames', type=int, default=32)
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    batch_size = 64
    iterations = 400000
    start_epoch = 0

    start = time.time()
    i3d_model = i3d.resnet50()
    i3d_model.load_state_dict(torch.load(args.weights_file))
    print('Loading model took {}'.format(time.time() - start))

    i3d_model = torch.nn.DataParallel(i3d_model).cuda()
    cudnn.benchmark = False

    train_transforms = t.get_default_transforms('train')
    val_transforms = t.get_default_transforms('val')

    train_loader = torch.utils.data.DataLoader(
            VideoDataset(args.root_data_path, args.map_file, sample_frames=args.sample_frames,
                         image_tmpl="frame_{:06d}.jpg",
                         train_mode=False, test_clips=args.test_clips,
                         transform=train_transforms),
            batch_size=batch_size, shuffle=True, num_workers=args.workers)  # , pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            VideoDataset(args.root_data_path, args.map_file, sample_frames=args.sample_frames,
                         image_tmpl="frame_{:06d}.jpg",
                         train_mode=False, test_clips=args.test_clips,
                         transform=val_transforms),
            batch_size=batch_size, shuffle=False, num_workers=args.workers)

    epoch_size = len(train_loader.dataset)
    num_epochs = round(iterations / epoch_size)  # Check number of epochs!!!

    # loaders = {'train': train_loader, 'val': val_loader}

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            i3d_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    for epoch in range(start_epoch, num_epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, i3d_model, criterion, optimizer, epoch, log_training)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, i3d_model, criterion, epoch, log_training)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': i3d_model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, top5=top5,
                          lr=optimizer.param_groups[-1]['lr'] * 0.1))
            print(output)
            log.write(output + '\n')
            log.flush()


def validate(val_loader, model, criterion, epoch, log=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              i, len(val_loader), batch_time=batch_time, loss=losses,
                              top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    return top1.avg


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
