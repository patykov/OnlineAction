import argparse
import time

import torch.nn.parallel

import datasets
import metric_tools.metrics as m
import models.nonlocal_net as i3d

# options
parser = argparse.ArgumentParser()
parser.add_argument('--map_file', type=str)
parser.add_argument('--root_data_path', type=str, default=("../../../"
                                                           "Datasets/Kinetics/400/val_frames_256"))
parser.add_argument('--base_model', type=str, default="resnet50")
parser.add_argument('--weights_file', type=str, default=None)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--baseline', action='store_false')
parser.add_argument('--causal', action='store_true')
parser.add_argument('--mode', type=str, default='val')
parser.add_argument('--dataset', type=str, default='kinetics')
parser.add_argument('--sample_frames', type=int, default=32)
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()
assert args.mode in ['test', 'val'], ('Mode {} does not exist. Choose between "val" or "test" for'
                                      ' evaluation'.format(args.mode))
assert args.dataset in ['kinetics', 'charades'], (
    'Dataset {} not available. Choose between "kinetics" or "charades".'.format(args.dataset))

torch.multiprocessing.set_sharing_strategy('file_system')

start = time.time()

i3d_model = i3d.resnet50(weights_file=args.weights_file, mode=args.mode, dataset=args.dataset,
                         non_local=args.baseline, frame_num=args.sample_frames)
i3d_model.eval()
load_model_time = time.time()
print('Loading model took {:.3f}s'.format(load_model_time - start))

Dataset = getattr(datasets, args.dataset.capitalize())
dataset = Dataset(args.root_data_path, args.map_file, sample_frames=args.sample_frames,
                  mode=args.mode, causal=args.causal)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=args.workers)  # , pin_memory=True)

total_num = len(data_loader.dataset)
data_gen = enumerate(data_loader, start=1)
print('Loading dataset took {:.3f}s'.format(time.time() - load_model_time))

# i3d_model = torch.nn.DataParallel(i3d_model)

log = dataset.set_log(args.output_file)

batch_time = m.AverageMeter()
data_time = m.AverageMeter()
with torch.no_grad():

    end = time.time()
    for i, (data, label) in data_gen:
        # measure data loading time
        data_time.update(time.time() - end)

        data = data.squeeze(0).cuda()
        rst = i3d_model(data).cpu()
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
