import argparse
import os
import time

import numpy as np
import torch.nn.parallel

import eval_utils as eu
import models.nonlocal_net as i3d
import transforms as t
from datasets.VideoDataset import VideoDataset

# options
parser = argparse.ArgumentParser()
parser.add_argument('--map_file', type=str)
parser.add_argument('--root_data_path', type=str, default=("../../../"
                                                           "Datasets/Kinetics/400/val_frames_256"))
parser.add_argument('--base_model', type=str, default="resnet50")
parser.add_argument('--weights_file', type=str, default="/../models/"
                    "pre-trained/resnet50_nl_i3d_kinetics.pth")
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--baseline', action='store_false')
parser.add_argument('--causal', action='store_true')
parser.add_argument('--mode', type=str, default='val')
parser.add_argument('--test_clips', type=int, default=10)
parser.add_argument('--sample_frames', type=int, default=32)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()
assert args.mode in ['test', 'val'], ('Mode {} does not exist. Choose between "val" or "test" for'
                                      ' evaluation'.format(args.mode))

torch.multiprocessing.set_sharing_strategy('file_system')

start = time.time()

i3d_model = i3d.resnet50(non_local=args.baseline, frame_num=args.sample_frames)
i3d_model.load_state_dict(torch.load(args.weights_file))
i3d_model.set_mode(args.mode)
i3d_model.eval()
print('Loading model took {}'.format(time.time() - start))

transforms = t.get_default_transforms(i3d_model.mode)

stride = 2 if args.sample_frames == 32 else 8
data_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_data_path, args.map_file, sample_frames=args.sample_frames,
                     image_tmpl="frame_{:06d}.jpg", stride=stride,
                     train_mode=False, test_clips=args.test_clips,
                     transform=transforms),
        batch_size=1, shuffle=False, num_workers=args.workers)  # , pin_memory=True)

total_num = len(data_loader.dataset)
data_gen = enumerate(data_loader, start=1)

# i3d_model = torch.nn.DataParallel(i3d_model)


def eval_video(data):
    data = data.squeeze(0).cuda()
    rst = i3d_model(data).cpu()

    if args.causal:
        return [rst[:i].mean(0) for i in range(1, 11)]

    else:
        return rst.mean(0)


if args.output_file is not None:
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(args.output_file, 'w') as file:
        if args.causal:
            file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(
                'Label', 'Top5 - 10%', 'Top5 - 20%', 'Top5 - 30%', 'Top5 - 40%', 'Top5 - 50%',
                'Top5 - 60%', 'Top5 - 70%', 'Top5 - 80%', 'Top5 - 90%', 'Top5 - 100%'))
        else:
            file.write('{:^10} | {:^20}\n'.format('Label', 'Top5 predition'))

score_text = ''
video_pred = []
video_labels = []
batch_time = eu.AverageMeter()
data_time = eu.AverageMeter()
with torch.no_grad():

    end = time.time()
    for i, (data, label) in data_gen:
        # measure data loading time
        data_time.update(time.time() - end)

        rst = eval_video(data)

        if args.causal:
            top5_pred = []
            top5_str = ''
            for j in range(10):
                _, t5_p = torch.topk(rst[j], 5)
                top5_pred.append(t5_p)
                top5_str += '{}\t'.format(np.array2string(t5_p.numpy(), separator=' ')[1:-1])
            score_text += ('{}\t{}\n').format(label[0], top5_str)
        else:
            _, top5_pred = torch.topk(rst, 5)
            score_text += '{:^10} | {:^20}\n'.format(label[0], np.array2string(
                top5_pred.numpy(), separator=', ')[1:-1])

        video_pred.append(top5_pred)
        video_labels.append(label[0])

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
                if args.output_file is not None:
                    with open(args.output_file, 'a') as file:
                        file.write(score_text)
                score_text = ''

# Saving last < 100 lines
if args.output_file is not None:
    with open(args.output_file, 'a') as file:
        file.write(score_text)

if args.causal:
    eu.save_causal_metrics(video_pred, video_labels, args.output_file, batch_time, data_time)
else:
    eu.save_metrics(video_pred, video_labels, args.output_file, batch_time, data_time)
