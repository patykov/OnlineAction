import argparse
import time

import torch.nn.parallel

import datasets
import metric_tools.metrics as m
import models.nonlocal_net as i3d


def eval(map_file, root_data_path, weights_file, output_file, baseline, causal, mode, dataset,
         sample_frames, workers):
    start = time.time()

    # Loading data
    Dataset = getattr(datasets, dataset.capitalize())
    dataset = Dataset(root_data_path, map_file, sample_frames=sample_frames,
                      mode=mode, causal=causal)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=workers)

    total_num = len(data_loader.dataset)
    data_gen = enumerate(data_loader, start=1)
    data_time = time.time()
    print('Loading dataset took {:.3f}s'.format(data_time - start))

    # Loading model
    model = i3d.resnet50(weights_file=weights_file, mode=mode,
                         num_classes=dataset.num_classes, non_local=baseline,
                         frame_num=sample_frames)
    model.eval()
    model_time = time.time()
    print('Loading model took {:.3f}s'.format(model_time - data_time))

    log = dataset.set_log(args.output_file)

    batch_time = m.AverageMeter()
    data_time = m.AverageMeter()
    with torch.no_grad():

        end = time.time()
        for i, (data, label) in data_gen:
            # measure data loading time
            data_time.update(time.time() - end)

            data = data.squeeze(0).cuda()
            rst = model(data).cpu()
            log.update(rst, label)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Video {}/{} ({:.02f}%) | '
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s avg.) | '
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s avg.)'.format(
                         i, total_num, i*100/total_num, batch_time=batch_time, data_time=data_time))
                if i % 20 == 0:
                    # Saving as the program goes in case of error
                    log.save_partial()

            if i % 20 == 0:
                break

    log.get_metrics(batch_time, data_time)


if __name__ == '__main__':
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', type=str)
    parser.add_argument('--root_data_path', type=str,
                        help="Full path to the videos directory")
    parser.add_argument('--weights_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
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

    eval(args.map_file, args.root_data_path, args.weights_file, args.output_file, args.baseline,
         args.causal, args.mode, args.dataset, args.sample_frames, args.workers)
