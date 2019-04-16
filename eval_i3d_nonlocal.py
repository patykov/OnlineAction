import argparse
import time

import torch.nn.parallel
import torch.optim
import torchvision

import eval_utils as eu
import models.i3d_nonlocal as i3d
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
i3d_model = i3d.resnet50()
i3d_model.load_state_dict(torch.load(args.weights_file))
# i3d_model.set_test()
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

    result = i3d_model(data)
    print(result.shape)
    raise Exception
    causal_results = [result[0]]
    for i in range(1, 10):
        causal_results.append(result[:i, :])
    return i3d_model(data).mean(0).cpu()


def causal_video_eval(data):
    data = data.to(device)
    data = data.squeeze(0)
    data = data.view(num_channel, -1, num_depth, data.size(2), data.size(3)).contiguous()
    data = data.permute(1, 0, 2, 3, 4).contiguous()

    return i3d_model(data).mean(0).cpu()


eu.evaluate_model(i3d_model, eval_video, data_gen, total_num, args.output_file)
