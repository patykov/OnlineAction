import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision

import eval_utils as eu
import models.nonlocal_net as i3d
import transforms as t
from datasets.I3D import I3DDataSet

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--map_file', type=str)
parser.add_argument('--root_data_path', type=str, default=("/media/v-pakova/New Volume/"
                                                           "Datasets/Kinetics/400/val_frames_256"))
parser.add_argument('--base_model', type=str, default="resnet50")
parser.add_argument('--weights_file', type=str, default="/media/v-pakova/New Volume/"
                    "OnlineActionRecognition/models/pre-trained/resnet50_i3d_kinetics.pt")
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--test_clips', type=int, default=10)
parser.add_argument('--sample_frames', type=int, default=32)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--workers', default=6, type=int,
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

num_class = 400

start = time.time()
i3d_model = i3d.resnet50(is_train=False)
i3d_model.load_state_dict(torch.load(args.weights_file))
i3d_model.flatten_features_output = False
i3d_model.set_test()
print(i3d_model)
print('Loading model took {}'.format(time.time() - start))

# i3d_model = torch.nn.DataParallel(i3d_model).to('cuda')
device = torch.device("cuda")
# i3d_model.to(device)

cropping = torchvision.transforms.Compose([
    t.GroupResize(256),
    # t.GroupCenterCrop(224),
])

# input_mean = [0.485, 0.456, 0.406]  # ?
# input_std = [0.225, 0.225, 0.225]  # std is on conv1

data_loader = torch.utils.data.DataLoader(
        I3DDataSet(args.root_data_path, args.map_file, sample_frames=args.sample_frames,
                   image_tmpl="frame_{:06d}.jpg",
                   train_mode=False, test_clips=args.test_clips,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       t.GroupToTensorStack(),
                    ])),
        batch_size=1, shuffle=False, num_workers=args.workers)  # , pin_memory=True)

total_num = len(data_loader.dataset)
data_gen = enumerate(data_loader, start=1)

num_channel = 3
num_depth = 32


def eval_video(data):
    data = data.to(device)
    data = data.squeeze(0)
    data = data.view(num_channel, -1, num_depth, data.size(2), data.size(3)).contiguous()
    data = data.permute(1, 0, 2, 3, 4).contiguous()

    if data.size(0) > 10:
        sub_data = data.view(10, -1, num_channel, num_depth, data.size(3), data.size(4))
        results = torch.stack([i3d_model(sub_data[i]).mean(0).cpu() for i in range(10)])

        return results.mean(0)

    else:
        return i3d_model(data).mean(0).cpu()


if args.output_file is not None:
    with open(args.output_file, 'w') as file:
        file.write('{:^10} | {:^20}\n'.format('Label', 'Top5 predition'))

proc_start_time = time.time()

score_text = ''
video_pred = []
video_labels = []
with torch.no_grad():
    for i, (data, label) in data_gen:
        rst = eval_video(data)
        cnt_time = time.time() - proc_start_time

        _, top5_pred = torch.topk(rst, 5)
        video_pred.append(top5_pred)
        video_labels.append(label[0])
        score_text += '{:^10} | {:^20}\n'.format(label[0], np.array2string(
            top5_pred.numpy(), separator=', ')[1:-1])

        if i % 10 == 0:
            print('video {}/{} done, {:.02f}%, average {:.5f} sec/video'.format(
                i, total_num, i*100/total_num, float(cnt_time)/i))
            # if i % 50 == 0:
            # Saving as the program goes in case of error
            if args.output_file is not None:
                with open(args.output_file, 'a') as file:
                    file.write(score_text)
            score_text = ''

# Saving last < 100 lines
if args.output_file is not None:
    with open(args.output_file, 'a') as file:
        file.write(score_text)

eu.save_metrics(video_pred, video_labels, args.output_file)
