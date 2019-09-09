import argparse
import logging
import os
import time

import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import metrics.metrics as m
import utils
from models.get import get_model


def save_checkpoint(checkpoint_path, model, meta, optimizer, scheduler):
    if isinstance(model, nn.DataParallel):
        model = model.module
    chkpt_dict = {
        'model': model.state_dict(),
        'meta': meta,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(chkpt_dict, checkpoint_path)


def load_checkpoint(checkpoint_path, model, meta, optimizer, scheduler):
    chkpt_dict = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(chkpt_dict['model'])
    meta.update(chkpt_dict['meta'])
    optimizer.load_state_dict(chkpt_dict['optimizer'])
    scheduler.load_state_dict(chkpt_dict['scheduler'])


def run_epoch(model, dataloader, epoch, num_epochs, criterion, metric, is_train, optimizer=None):
    """
    Iterate over the dataloader computing losses and metrics and, if `is_train=True`,
    updating model weights.
    """
    model.train(is_train)  # Set model to train/eval mode
    if is_train:
        dataloader.sampler.set_epoch(epoch)

    offset = 0
    metric.reset()
    running_loss = m.AverageMeter('running_loss')

    with tqdm(
            desc='\nEpoch {}/{} - {}'.format(epoch, num_epochs, 'Train' if is_train else 'Val'),
            total=len(dataloader.dataset),
            leave=False,
            maxinterval=3600,
            disable=(hvd.rank() != 0)) as t:
        with torch.set_grad_enabled(is_train):
            for (data, label) in dataloader:  # Iterate on the data
                batch_samples = torch.tensor(data.shape[0]).item()

                # Get outputs
                outputs = model(data.view(-1, *data.shape[2:]).cuda())
                targets = label['target'].cuda()

                # Compute loss and metrics
                loss = criterion(outputs, targets)
                running_loss.update(loss.item(), batch_samples)
                metric.add(outputs, targets)

                if running_loss.count - offset >= t.total // 10:  # Update progressbar every 10%
                    t.set_postfix({
                        'loss': '{:.4f}'.format(running_loss.avg),
                        metric.name: str(metric)
                    }, refresh=False)
                    t.update(running_loss.count - offset)
                    offset = running_loss.count

                # Update weights
                if is_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

    return running_loss


def train(config_json, train_file, val_file, train_data, val_data, sample_frames, dataset,
          checkpoint_path, restart=False, num_workers=4, arch='nonlocal_net', backbone='resnet50',
          pretrained_weights=None, fine_tune=True, pos_weight_file=False, subset=False):

    config = utils.parse_json(config_json)

    chkpt_name, chkpt_ext = os.path.splitext(checkpoint_path)
    backup_path = chkpt_name + '_bkp' + chkpt_ext

    # Data loaders
    train_loader, val_loader = utils.get_dataloaders(
        dataset, train_file, val_file, train_data, val_data, config['batch_size'],
        sample_frames=sample_frames, num_workers=num_workers, distributed=True, subset=subset)
    num_classes = train_loader.dataset.num_classes
    multi_label = train_loader.dataset.multi_label

    # Metrics
    train_metric = m.mAP() if multi_label else m.TopK()
    val_metric = m.mAP() if multi_label else m.TopK()

    # Model
    model = get_model(arch=arch, backbone=backbone, pretrained_weights=pretrained_weights,
                      mode='train', num_classes=num_classes, non_local=config['nonlocal'],
                      frame_num=sample_frames, fine_tune=fine_tune, log_name='training')

    # Epochs, optimizer, scheduler, criterion
    num_epochs, optimizer, scheduler = utils.get_optimizer(
        model, config['learning_rate'], config['weight_decay'], distributed=True)

    if multi_label:
        pos_weight = torch.load(pos_weight_file).cuda() if pos_weight_file else None

        def criterion(outputs, labels):
            return F.binary_cross_entropy_with_logits(
                outputs, labels.type_as(outputs), pos_weight=pos_weight)
    else:

        def criterion(outputs, labels):
            return F.cross_entropy(outputs, labels)

    # Info to restore from checkpoint
    meta = {}
    for k in ['loss', 'val_loss', 'metric', 'val_metric']:
        meta.setdefault(k, [])
    meta['config'] = config
    meta['train_file'] = os.path.basename(train_file)
    meta['val_file'] = os.path.basename(val_file)

    # Logging info
    LOG = logging.getLogger(name='training')
    if hvd.rank() == 0:
        if os.path.exists(checkpoint_path) and not restart:
            try:
                LOG.info('Loading checkpoint from {}.'.format(checkpoint_path))
                load_checkpoint(checkpoint_path, model, meta, optimizer, scheduler)
            except RuntimeError as e:
                LOG.info(e)
                LOG.info('Could not load checkpoint in {}. Trying backup checkpoint.'.format(
                    checkpoint_path))
                load_checkpoint(backup_path, model, meta, optimizer, scheduler)

        else:
            LOG.info('Learning rate configuration: {}'.format(config['learning_rate']))
            LOG.info('Weight decay: {:g}\n'.format(config['weight_decay']))
            LOG.info(train_loader.dataset)
            LOG.info(val_loader.dataset)
            if multi_label:
                LOG.info('Criterion with{} balanced weights (pos_weight)'.format(
                    'out' if pos_weight is None else ''))
            LOG.info('Batch size: {:d}'.format(config['batch_size']))
            LOG.info('Using {:d} workers for data loading\n'.format(num_workers))
            LOG.info('Saving results to {}\n'.format(os.path.abspath(checkpoint_path)))
            LOG.info(model)
            LOG.info('\nNumber of trainable parameters: {}'.format(
                sum(p.numel() for p in model.parameters() if p.requires_grad)))
            LOG.info('Using {:d} horovod process{}'.format(hvd.size(),
                                                           'es' if hvd.size() > 1 else ''))

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    utils.broadcast_scheduler_state(scheduler, root_rank=0)

    initial_epoch = scheduler.last_epoch

    for epoch in range(initial_epoch, num_epochs):
        b = time.time()
        # Train for one epoch
        train_loss = run_epoch(model, train_loader, epoch, num_epochs, criterion,
                               train_metric, True, optimizer)
        e1 = time.time()

        meta['loss'].append(train_loss.avg)
        meta['metric'].append(train_metric.value)

        # Validate
        b2 = time.time()
        val_loss = run_epoch(model, val_loader, epoch, num_epochs, criterion,
                             val_metric, False)
        e = time.time()

        meta['val_loss'].append(val_loss.avg)
        meta['val_metric'].append(val_metric.value)

        scheduler.step()

        if hvd.rank() == 0:
            # Checkpoint with redundancy
            if os.path.exists(checkpoint_path):
                os.rename(checkpoint_path, backup_path)
            save_checkpoint(checkpoint_path, model, meta, optimizer, scheduler)

            train_time = e1 - b
            val_time = e - b2

            prefix = 'Epoch {}/{} - '.format(epoch + 1, num_epochs)
            tmp = ('{prefix}{phase:>5}: loss = {loss:.4f}, {metric.name} = {metric}.'
                   ' {phase} time: {time:.2f}s ({rate:.2f} samples/s)')
            train_string = tmp.format(
                prefix=prefix,
                phase='Train',
                loss=train_loss.avg,
                metric=train_metric,
                time=train_time,
                rate=train_loss.count / train_time)
            val_string = tmp.format(
                prefix=' ' * len(prefix),
                phase='Val',
                loss=val_loss.avg,
                metric=val_metric,
                time=val_time,
                rate=val_loss.count / val_time)
            LOG.info('{}\n{}'.format(train_string, val_string))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        required=True,
                        help='JSON configuration file')
    parser.add_argument('--train_map_file', type=str,
                        required=True,
                        help='Full path to the file that maps train data')
    parser.add_argument('--val_map_file', type=str,
                        required=True,
                        help='Full path to the file that maps val data')
    parser.add_argument('--train_data_path', type=str,
                        required=True,
                        help='Full path to the training videos directory')
    parser.add_argument('--val_data_path', type=str,
                        help=('Full path to the validation videos directory.'
                              'If None, will be used val_data_path = train_data_path'),
                        default=None)
    parser.add_argument('--arch', type=str, default='nonlocal_net')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--pos_weights', type=str, default=None)
    parser.add_argument('--outputdir',
                        help='Output directory for checkpoints and models',
                        default=None)
    parser.add_argument('--sample_frames', type=int, default=32,
                        help='Number of frames to be sampled in the input.')
    parser.add_argument('--dataset', type=str, default='charades')
    parser.add_argument('--filename',
                        help='Checkpoint and logging filenames',
                        default=None)
    parser.add_argument('--restart',
                        help=('Restart from scratch (instead of restarting from checkpoint ',
                              'file by default)'),
                        action='store_true')
    parser.add_argument('--subset',
                        action='store_true')
    parser.add_argument('--fine_tune',
                        help=('Fine-tune the model from the weights stored in pretrained_weights.'),
                        action='store_true')
    parser.add_argument('--workers', type=int,
                        help='Number of workers on the data loading subprocess',
                        default=4)

    args = parser.parse_args()

    assert args.dataset in ['kinetics', 'charades'], (
        'Dataset {} not available. Choose between "kinetics" or "charades".'.format(args.dataset))

    # STUFF
    val_data_path = args.val_data_path if args.val_data_path else args.train_data_path
    filename = args.filename if args.filename else os.path.splitext(
        os.path.basename(args.config_file))[0]
    outputdir = os.path.join(
        args.outputdir,
        os.path.splitext(os.path.basename(args.filename))[0]) if args.outputdir else os.path.join(
            os.path.dirname(__file__), '..', 'outputs',
            os.path.splitext(os.path.basename(args.filename))[0])
    checkpoint_path = os.path.join(outputdir, filename + '.pth')
    os.makedirs(outputdir, exist_ok=True)

    # Initialize horovod
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # Setting loggers
    if hvd.rank() == 0:
        log_file = os.path.join(outputdir, filename + '.log')
        utils.setup_logger('training', log_file)

    train(args.config_file, args.train_map_file, args.val_map_file, args.train_data_path,
          val_data_path, args.sample_frames, args.dataset, checkpoint_path, args.restart,
          args.workers, args.arch, args.backbone, args.pretrained_weights, args.fine_tune,
          args.pos_weights, args.subset)


if __name__ == '__main__':
    main()
