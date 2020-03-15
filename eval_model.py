import argparse
import logging
import os
import time

import torch.nn.parallel

import horovod.torch as hvd
import metrics.metrics as m
from datasets.get import get_dataloader
from models.get import get_model
from utils import setup_logger

torch.backends.cudnn.benchmarks = True


def eval(map_file, root_data_path, pretrained_weights, arch, backbone, non_local, mode, dataset,
         sample_frames, workers, selected_classes_file=None, verb_classes_file=None):
    start_time = time.time()

    LOG = logging.getLogger(name='eval')
    RESULTS = logging.getLogger(name='results')

    # Loading data
    data_loader = get_dataloader(dataset, list_file=map_file, root_path=root_data_path, mode=mode,
                                 sample_frames=sample_frames, batch_size=1, num_workers=workers,
                                 distributed=True, selected_classes_file=selected_classes_file,
                                 verb_classes_file=verb_classes_file)

    total_num = len(data_loader.dataset)
    num_classes = data_loader.dataset.num_classes
    data_gen = enumerate(data_loader, start=1)

    data_time = time.time()
    LOG.info('Loading dataset took {:.3f}s'.format(data_time - start_time))
    LOG.debug(data_loader.dataset)

    # Loading model
    model = get_model(arch=arch, backbone=backbone, pretrained_weights=pretrained_weights,
                      fullyConv=False if 'centerCrop' in mode else True,
                      num_classes=num_classes, non_local=non_local,
                      frame_num=sample_frames, log_name='eval')
    model.eval()
    model_time = time.time()

    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    LOG.info('Loading model took {:.3f}s'.format(model_time - data_time))
    LOG.debug(model)

    video_metric = m.VideoMAP(
        m.mAP(video=True)) if data_loader.dataset.multi_label else m.VideoAccuracy(
        m.TopK(k=(1, 5), video=True))
    batch_time = m.AverageMeter('batch_time')
    data_time = m.AverageMeter('data_time')
    with torch.no_grad():

        end = time.time()
        for i, (data, label) in data_gen:
            # measure data loading time
            data_time.update(time.time() - end)

            data = data.squeeze(0).cuda()
            output = model(data).cpu()
            video_metric.add(output, label)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 50 == 0:
                LOG.info('Video {}/{} ({:.02%}) | '
                         'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s avg.) | '
                         'Data {data_time.val:.3f}s ({data_time.avg:.3f}s avg.) | '
                         '{metric.name}: {metric}'.format(
                             i, total_num, i / total_num, batch_time=batch_time,
                             data_time=data_time, metric=video_metric.metric))

                RESULTS.debug(video_metric.to_text())

            # Trying to empty gpu cache
            torch.cuda.empty_cache()

    RESULTS.debug(video_metric.to_text())
    LOG.info('\n{metric.name}: {metric}'.format(metric=video_metric.metric))


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', type=str)
    parser.add_argument('--root_data_path', type=str, help="Full path to the videos directory")
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--log_file', help='Results, log and metrics filenames.', type=str)
    parser.add_argument('--outputdir', help='Output directory for logs and results.', default=None)
    parser.add_argument('--arch', type=str, default='nonlocal_net')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--non_local', action='store_true')
    parser.add_argument('--mode', type=str, default='video_centerCrop',
                        choices=['video_centerCrop', 'video_fullyConv', 'video_3crops', 'val'])
    parser.add_argument('--dataset', type=str, default='kinetics', choices=['kinetics', 'charades'])
    parser.add_argument('--sample_frames', type=int, default=8,
                        help='Number of frames to be sampled in the input.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of workers on the data loading subprocess.')
    parser.add_argument('--selected_classes_file', type=str, default=None,
                        help='Full path to the file with the classes to be used in training')
    parser.add_argument('--verb_classes_file', type=str, default=None,
                        help='Full path to the file with the classes to verbs mapping')

    args = parser.parse_args()

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
         args.non_local, args.mode, args.dataset, args.sample_frames, args.workers,
         args.selected_classes_file, args.verb_classes_file)


if __name__ == '__main__':
    main()
