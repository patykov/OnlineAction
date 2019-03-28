import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision
from sklearn.metrics import confusion_matrix

import transforms as tf
from datasets import I3DDataSet
from models.i3d_nonlocal import *

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--test_list', type=str)
# parser.add_argument('weights', type=str)
parser.add_argument('--root_data_path', type=str, default=("/media/v-pakova/New Volume/"
                    "Datasets/Kinetics/400/val_frames_256"))
parser.add_argument('--base_model', type=str, default="resnet50")
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--test_clips', type=int, default=10)
parser.add_argument('--sample_frames', type=int, default=32)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()

num_class = 400

# i3d_model = I3DResNet(num_class, args.sample_frames, base_model=args.base_model,
# # dropout=args.dropout)
# pre_trained_file = ("/media/v-pakova/New Volume/OnlineActionRecognition/models/pre-trained/"
#                     "non-local/i3d_nonlocal_32x2_IN_pretrain_400k.pkl")
# blank_resnet_i3d = i3d.resnet50()
# i3d_model = i3d.copy_weigths_i3dResNet(pre_trained_file, blank_resnet_i3d)
i3d_model = torch.load('/media/v-pakova/New Volume/OnlineActionRecognition/models/'
                       'pre-trained/resnet50_i3d_pre_trained.pt')

device = torch.device("cuda:0")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
i3d_model = torch.nn.DataParallel(i3d_model)
i3d_model.to(device)
# i3d_model.eval()

cropping = torchvision.transforms.Compose([
    tf.GroupResize(256),
    tf.GroupCenterCrop(224),
])

input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

data_loader = torch.utils.data.DataLoader(
        I3DDataSet(args.root_data_path, args.test_list, sample_frames=args.sample_frames,
                   image_tmpl="frame_{:06d}.jpg",
                   train_mode=False, test_clips=args.test_clips,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       tf.Stack(),
                       tf.ToTorchFormatTensor(),
                       #  tf.GroupNormalize(input_mean, input_std),
                    ])),
        batch_size=1, shuffle=False, pin_memory=True, num_workers=0)  # args.workers * 2)

data_gen = enumerate(data_loader)
total_num = len(data_loader.dataset)

num_channel = 3
num_depth = 32


def eval_video(data):
    data = data.squeeze(0)
    data = data.view(num_channel, -1, num_depth, data.size(2), data.size(3)).contiguous()
    data = data.permute(1, 0, 2, 3, 4).contiguous()

    input_var = torch.Tensor(data).to(device)
    rst = i3d_model(input_var).data.cpu().numpy().copy()
    return rst.reshape((args.test_clips*args.test_crops, num_class)).mean(axis=0).reshape(
        (1, num_class))[0]


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

        top5_pred = rst.argsort()[-5:][::-1]
        top1_pred = top5_pred[0]
        video_pred.append(top1_pred)
        video_labels.append(label[0])
        score_text += '{:^10} | {:^20}\n'.format(label[0], np.array2string(
            top5_pred, separator=', ')[1:-1])

        if i % 10 == 0:
            print('video {}/{} done, {:.02f}%, average {:.5f} sec/video'.format(
                i+1, total_num, (i+1)*100/total_num, float(cnt_time) / (i+1)))
            # if i % 100 == 0:
            # Saving as the program goes in case of error
            if args.output_file is not None:
                with open(args.output_file, 'a') as file:
                    file.write(score_text)
            score_text = ''

# Saving last < 100 lines
if args.output_file is not None:
    with open(args.output_file, 'a') as file:
        for l, p in zip(video_labels, video_pred):
            file.write(score_text)

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = [h/c if c > 0 else 0.00 for (h, c) in zip(cls_hit, cls_cnt)]

print('\n\nAccuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
