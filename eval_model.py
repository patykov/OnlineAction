import argparse
import logging
import os
import time

import horovod.torch as hvd
import torch.nn.parallel
from torch.nn import AvgPool1d, MaxPool1d

import metrics.metrics as m
from datasets.get import get_dataloader, get_dataset
from models.get import get_model
from utils import setup_logger

torch.backends.cudnn.benchmarks = True


def eval(map_file, root_data_path, pretrained_weights, arch, backbone, baseline, mode,
         dataset_name, sample_frames, workers):
    start_time = time.time()

    LOG = logging.getLogger(name='eval')
    RESULTS = logging.getLogger(name='results')

    # Loading data
    dataset = get_dataset(dataset_name, list_file=map_file, root_path=root_data_path, mode=mode,
                          sample_frames=sample_frames)
    data_loader = get_dataloader(dataset, list_file=map_file, root_path=root_data_path, mode=mode,
                                 sample_frames=sample_frames, batch_size=1, num_workers=workers,
                                 distributed=True)

    total_num = len(dataset)
    num_classes = dataset.num_classes
    data_gen = enumerate(data_loader, start=1)

    data_time = time.time()
    LOG.info('Loading dataset took {:.3f}s'.format(data_time - start_time))
    LOG.debug(dataset)

    # Loading model
    model = get_model(arch=arch, backbone=backbone, pretrained_weights=pretrained_weights,
                      mode=mode, num_classes=num_classes, non_local=baseline,
                      frame_num=sample_frames, log_name='eval')
    model.eval()
    model_time = time.time()

    if dataset.multi_label:
        def video_output(outputs):
            max_pool = MaxPool1d(dataset.test_clips)
            avg_pool = AvgPool1d(3)

            data = outputs.view(1, -1, num_classes).contiguous()
            data = data.permute(0, 2, 1).contiguous()

            data = max_pool(data)
            if mode == 'test':
                # During test, fullyconv transform takes 3 random crops of each clip
                data = avg_pool(data)
            video_data = data.view(-1, num_classes).contiguous()

            return video_data

    else:
        def video_output(outputs):
            return outputs.mean(0)

    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    LOG.info('Loading model took {:.3f}s'.format(model_time - data_time))
    LOG.debug(model)

    video_metric = m.VideoMAP(m.mAP()) if data_loader.dataset.multi_label else m.VideoAccuracy(
        m.TopK(k=(1, 5)))
    batch_time = m.AverageMeter('batch_time')
    data_time = m.AverageMeter('data_time')
    with torch.no_grad():

        end = time.time()
        for i, (data, label) in data_gen:
            # measure data loading time
            data_time.update(time.time() - end)

            data = data.squeeze(0).cuda()
            output = model(data).cpu()
            video_metric.add(video_output(output), label)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 50 == 0:
                LOG.info('Video {}/{} ({:.02%}) | '
                         'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s avg.) | '
                         'Data {data_time.val:.3f}s ({data_time.avg:.3f}s avg.) | '
                         '{metric.name}: {metric}'.format(
                            i, total_num, i/total_num, batch_time=batch_time,
                            data_time=data_time, metric=video_metric.metric))
                RESULTS.debug(video_metric.to_text())

    RESULTS.debug(video_metric.to_text())
    LOG.info('\n{metric.name}: {metric}'.format(metric=video_metric.metric))


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', type=str)
    parser.add_argument('--root_data_path', type=str,
                        help="Full path to the videos directory")
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--log_file',
                        help='Results, log and metrics filenames.',
                        type=str)
    parser.add_argument('--outputdir',
                        help='Output directory for logs and results.',
                        default=None)
    parser.add_argument('--arch', type=str, default='nonlocal_net')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--baseline', action='store_false')
    parser.add_argument('--mode', type=str, default='val')
    parser.add_argument('--dataset', type=str, default='kinetics')
    parser.add_argument('--sample_frames', type=int, default=8,
                        help='Number of frames to be sampled in the input.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of workers on the data loading subprocess.')

    args = parser.parse_args()
    assert args.mode in ['test', 'val'], (
        'Mode {} does not exist. Choose between "val" or "test" for evaluation'.format(args.mode))
    assert args.dataset in ['kinetics', 'charades'], (
        'Dataset {} not available. Choose between "kinetics" or "charades".'.format(args.dataset))

    torch.multiprocessing.set_sharing_strategy('file_system')

    outputdir = args.outputdir if args.outputdir else os.path.join(
        os.path.dirname(__file__), '..', 'outputs',
        os.path.splitext(os.path.basename(args.log_file))[0])
    os.makedirs(outputdir, exist_ok=True)
    base_name = os.path.join(outputdir, os.path.splitext(os.path.basename(args.log_file))[0])
    log_file = base_name + '.log'
    results_file = base_name + '.txt'

    # Initialize horovod
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    if hvd.rank() == 0:
        setup_logger('eval', log_file)
        setup_logger('results', results_file)

    eval(args.map_file, args.root_data_path, args.pretrained_weights, args.arch, args.backbone,
         args.baseline, args.mode, args.dataset, args.sample_frames, args.workers)


if __name__ == '__main__':
    main()
