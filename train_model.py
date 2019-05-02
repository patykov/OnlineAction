import argparse
import time

import torch.nn.parallel

import models.nonlocal_net as i3d
import transforms as t
from datasets.VideoDataset import VideoDataset

# options
parser = argparse.ArgumentParser()
parser.add_argument('--map_file', type=str)
parser.add_argument('--root_data_path', type=str, default=("/media/v-pakova/New Volume/"
                                                           "Datasets/Kinetics/400/val_frames_256"))
parser.add_argument('--base_model', type=str, default="resnet50")
parser.add_argument('--weights_file', type=str, default="/media/v-pakova/New Volume/"
                    "OnlineActionRecognition/models/pre-trained/resnet50_i3d_kinetics.pt")
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--sample_frames', type=int, default=32)
parser.add_argument('--workers', default=6, type=int,
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

num_class = 400
batch_size = 64
iterations = 400000
# epoch_num = ?

start = time.time()
i3d_model = i3d.resnet50()
i3d_model.load_state_dict(torch.load(args.weights_file))
i3d_model.set_mode('train')
print('Loading model took {}'.format(time.time() - start))

device = torch.device("cuda")
# i3d_model = torch.nn.DataParallel(i3d_model).to(device)
# i3d_model.to(device)

transforms = t.get_default_transforms(i3d_model.mode)

data_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_data_path, args.map_file, sample_frames=args.sample_frames,
                     image_tmpl="frame_{:06d}.jpg",
                     train_mode=False, test_clips=args.test_clips,
                     transform=transforms),
        batch_size=batch_size, shuffle=True, num_workers=args.workers)  # , pin_memory=True)

epoch_size = len(data_loader.dataset)
data_gen = enumerate(data_loader, start=1)

num_channel = 3
num_depth = 32
