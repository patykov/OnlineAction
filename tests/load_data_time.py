import time

import torch
import torchvision

import models.i3d_nonlocal as i3d
import transforms as t
from datasets import I3DDataSet

root_data_path = "/media/v-pakova/New Volume/Datasets/Kinetics/400/val_frames_256"
test_list = "/media/v-pakova/New Volume/Datasets/Kinetics/400/val_list_short.txt"
sample_frames = 32
test_clips = 10
num_workers = 4

cropping = torchvision.transforms.Compose([
    t.GroupResize(256),
    t.GroupCenterCrop(224),
])

input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
dataset = I3DDataSet(root_data_path, test_list, sample_frames=sample_frames,
                     image_tmpl="frame_{:06d}.jpg",
                     train_mode=False, test_clips=test_clips,
                     transform=torchvision.transforms.Compose([
                       cropping,
                       t.GroupToTensorStack(),
                       t.GroupNormalize(input_mean, input_std),
                     ]))

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=num_workers)  # , pin_memory=True)

start = time.time()
pre_trained_file = ("/media/v-pakova/New Volume/OnlineActionRecognition/models/pre-trained/"
                    "non-local/i3d_nonlocal_32x2_IN_pretrain_400k.pkl")
blank_resnet_i3d = i3d.resnet50()
i3d_model = i3d.copy_weigths_i3dResNet(pre_trained_file, blank_resnet_i3d)

i3_model = torch.nn.DataParallel(i3d_model).to('cuda')
print(' loading nodel took {}'.format(time.time() - start))

start = time.time()
# with torch.no_grad():
for i in range(len(dataset)):
    s = dataset[i]
#     b = b[0].to('cuda')
    print('{} took {}, len = {}'.format(i, time.time() - start, len(s)))
    start = time.time()

    if i == 30:
        break
