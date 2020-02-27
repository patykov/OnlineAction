import argparse
import logging
import os
import time

import torch.nn.parallel
from torch.nn import AvgPool1d
from tqdm import tqdm

import horovod.torch as hvd
import metrics.metrics as m
from datasets.get import get_dataloader, get_dataset, get_distributed_sampler
from models.get import get_model
from utils import setup_logger

torch.backends.cudnn.benchmarks = True


def eval(map_file, root_data_path, pretrained_weights, arch, backbone, baseline, mode, subset,
         dataset, sample_frames, workers, selected_classes_file=None, verb_classes_file=None):
    start_time = time.time()

    LOG = logging.getLogger(name='eval')
    RESULTS = logging.getLogger(name='results')

    # Loading data
    data_sampler = get_distributed_sampler(dataset, list_file=map_file, root_path=root_data_path,
                                           subset=subset, mode=mode,
                                           sample_frames=sample_frames,
                                           selected_classes_file=selected_classes_file,
                                           verb_classes_file=verb_classes_file)
    video_dataset = data_sampler.dataset
    total_per_gpu = data_sampler.num_samples
    num_classes = video_dataset.num_classes
    data_time = time.time()
    LOG.info('Loading dataset took {:.3f}s'.format(data_time - start_time))
    LOG.info('Sampler total_size: {} | Sampler num_samples: {}'.format(
        data_sampler.total_size, total_per_gpu))
    LOG.debug(video_dataset)

    # Logging stream transforms
    # video_path, label = video_dataset[0]
    # stream_dataset = get_dataset((dataset, 'stream'), video_path=video_path, label=label,
    #                              num_classes=num_classes, mode=mode)
    # LOG.debug('Stream dataset sample:')
    # LOG.debug(stream_dataset)

    # Loading model
    model = get_model(arch=arch,
                      backbone=backbone,
                      pretrained_weights=pretrained_weights,
                      num_classes=num_classes,
                      non_local=baseline,
                      fullyConv=True if mode == 'fullyConv' else False,
                      frame_num=sample_frames,
                      log_name='eval')
    model.eval()
    model_time = time.time()

    if video_dataset.multi_label:

        def frame_output(outputs):
            avg_pool = AvgPool1d(3)

            outputs = torch.sigmoid(outputs)
            data = outputs.view(1, -1, num_classes).contiguous()
            data = data.permute(0, 2, 1).contiguous()

            if '3crops' in mode:
                # 3crops transform takes 3 random crops of each clip
                data = avg_pool(data)
            video_data = data.view(-1, num_classes).contiguous()

            return video_data

    else:
        softmax = torch.nn.Softmax(dim=1)

        def frame_output(outputs):
            return softmax(outputs)

    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    LOG.info('Loading model took {:.3f}s'.format(model_time - data_time))
    LOG.debug(model)

    video_metric = m.VideoPerFrameMAP(
        m.mAP()) if data_sampler.dataset.multi_label else m.VideoPerFrameAccuracy(m.TopK(k=(1, 5)))
    batch_time = m.AverageMeter('batch_time')
    data_time = m.AverageMeter('data_time')
    with torch.no_grad():

        end = time.time()
        count = 0
        offset = 0
        total = data_sampler.total_size
        with tqdm(desc='{} videos of {} total'.format(total_per_gpu, total),
                  total=total_per_gpu,
                  leave=True,
                  maxinterval=3600) as t:

            for i, vid in enumerate(data_sampler, start=1):
                video_path, label = video_dataset[vid]
                # measure data loading time
                data_time.update(time.time() - end)

                video_stream = get_dataloader((dataset, 'stream'), video_path=video_path,
                                              label=label, batch_size=None, num_classes=num_classes,
                                              mode=mode, distributed=False, num_workers=0)

                for j, (chunk_data, chunk_target) in enumerate(video_stream):
                    chunk_data = chunk_data.to('cuda')
                    output = model(chunk_data)

                    video_metric.add(output, {
                        'target': chunk_target['target'],
                        'video_path': chunk_target['video_path']
                    },
                        synchronize=((i == data_sampler.num_samples)
                                     and (j == len(video_stream) - 1)))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                count += 1
                if (count - offset) >= total // 100:  # Update progressbar every 1%
                    t.update(count - offset)
                    offset = count

                    # Saving results and log info
                    LOG.info('Video {}/{} ({:.02%}) | '
                             'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s avg.) | '
                             'Data {data_time.val:.3f}s ({data_time.avg:.3f}s avg.) | '
                             '{metric.name}: {metric}'.format(i,
                                                              total_per_gpu,
                                                              i / total_per_gpu,
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              metric=video_metric.metric))
                    RESULTS.debug(video_metric.to_text())
                    # video_metric.reset()

                # Trying to empty gpu cache
                torch.cuda.empty_cache()

    RESULTS.debug(video_metric.to_text())
    # LOG.info('\n{metric.name}: {metric}'.format(metric=video_metric.metric))


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
    parser.add_argument('--baseline', action='store_false')
    parser.add_argument('--mode', type=str, default='fullyConv')
    parser.add_argument('--dataset', type=str, default='kinetics')
    parser.add_argument('--sample_frames',
                        type=int,
                        default=8,
                        help='Number of frames to be sampled in the input.')
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--workers',
                        default=4,
                        type=int,
                        help='Number of workers on the data loading subprocess.')
    parser.add_argument('--selected_classes_file',
                        type=str,
                        default=None,
                        help='Full path to the file with the classes to be used in training')
    parser.add_argument('--verb_classes_file',
                        type=str,
                        default=None,
                        help='Full path to the file with the classes to verbs mapping')

    args = parser.parse_args()
    assert args.dataset in [
        'kinetics', 'charades'
    ], ('Dataset {} not available. Choose between "kinetics" or "charades".'.format(args.dataset))

    torch.multiprocessing.set_sharing_strategy('file_system')

    outputdir = args.outputdir if args.outputdir else os.path.join(
        os.path.dirname(__file__), '..', 'outputs',
        os.path.splitext(os.path.basename(args.log_file))[0])
    os.makedirs(outputdir, exist_ok=True)
    base_name = os.path.join(outputdir, os.path.splitext(os.path.basename(args.log_file))[0])

    # Initialize horovod
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    setup_logger('eval', base_name + '_{}.log'.format(hvd.rank()))
    setup_logger('results', base_name + '_{}.txt'.format(hvd.rank()))

    eval(args.map_file, args.root_data_path, args.pretrained_weights, args.arch, args.backbone,
         args.baseline, args.mode, args.subset, args.dataset, args.sample_frames, args.workers,
         args.selected_classes_file, args.verb_classes_file)


if __name__ == '__main__':
    main()
