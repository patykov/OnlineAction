import argparse
import logging
import os
import time

import torch.nn.parallel

import datasets
import metric_tools.metrics as m
import models.nonlocal_net as i3d
from utils import setup_logger


def eval(map_file, root_data_path, weights_file, baseline, causal, mode, dataset,
         sample_frames, workers):
    start_time = time.time()

    LOG = logging.getLogger(name='log')
    RESULTS = logging.getLogger(name='results')

    # Loading data
    Dataset = getattr(datasets, dataset.capitalize())
    dataset = Dataset(root_data_path, map_file, sample_frames=sample_frames,
                      mode=mode, causal=causal)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=workers)

    total_num = len(dataset)
    data_gen = enumerate(data_loader, start=1)
    data_time = time.time()
    LOG.info('Loading dataset took {:.3f}s'.format(data_time - start_time))
    LOG.debug(data_loader.dataset)

    # Loading model
    model = i3d.resnet50(weights_file=weights_file, mode=mode,
                         num_classes=dataset.num_classes, non_local=baseline,
                         frame_num=sample_frames)
    model.eval()
    model_time = time.time()
    LOG.info('Loading model took {:.3f}s'.format(model_time - data_time))
    LOG.debug(model)

    metric = m.Video_mAP(save_text=True) if dataset.multi_label else m.Video_Accuracy(
        save_text=True)
    batch_time = m.AverageMeter()
    data_time = m.AverageMeter()
    with torch.no_grad():

        end = time.time()
        for i, (data, label) in data_gen:
            # measure data loading time
            data_time.update(time.time() - end)

            data = data.squeeze(0).cuda()
            rst = model(data).cpu()
            metric.add(rst, label)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                LOG.info('Video {}/{} ({:.02f}%) | '
                         'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s avg.) | '
                         'Data {data_time.val:.3f}s ({data_time.avg:.3f}s avg.) | '
                         '{metric}'.format(
                             i, total_num, i*100/total_num, batch_time=batch_time,
                             data_time=data_time, metric=metric))
                RESULTS.debug(metric.to_text())

    RESULTS.debug(metric.to_text())
    LOG.info('\n{}'.format(metric))


if __name__ == '__main__':
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', type=str)
    parser.add_argument('--root_data_path', type=str,
                        help="Full path to the videos directory")
    parser.add_argument('--weights_file', type=str, default=None)
    parser.add_argument('--log_file',
                        help='Path to file name that will be used for the results, log and metrics',
                        type=str)
    parser.add_argument('--baseline', action='store_false')
    parser.add_argument('--causal', action='store_true')
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

    outputdir = os.path.dirname(args.log_file)
    base_name = os.path.join(outputdir, os.path.splitext(os.path.basename(args.log_file))[0])
    log_file = base_name + '.log'
    results_file = base_name + '.txt'

    setup_logger('log', log_file)
    setup_logger('results', results_file)

    eval(args.map_file, args.root_data_path, args.weights_file, args.baseline, args.causal,
         args.mode, args.dataset, args.sample_frames, args.workers)
