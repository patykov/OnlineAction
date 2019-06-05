# OnlineAction
Pytorch implementation of [**Non-local Neural Networks**](https://arxiv.org/abs/1711.07971).

## Dataset
After downloading the Kinetics dataset version from [DATASET.md](https://github.com/facebookresearch/video-nonlocal-net/blob/master/DATASET.md), use the following script to create a classes.txt file and a dir_name_list.txt file. The first one maps each class name found in the directory to an ID, while the second lists each video of the directory with its respective class ID.
```bash
python prepare_dataset.py --dir_path [path_to_video_dir]
```
For instance, you should call this script for the folders: 'train_clips', 'test_clips' and 'val_clips'; generating the files: 'train_clips_list.txt', 'test_clips_list.txt', 'val_clips_list.txt' and 'classes.txt'.


## Evaluation

**Pretrained Weights**  
Download the pretrained weights from [https://github.com/facebookresearch/video-nonlocal-net](https://github.com/facebookresearch/video-nonlocal-net) repo:
- i3d_baseline_32x2_IN_pretrain_400k.pkl
- i3d_nonlocal_32x2_IN_pretrain_400k.pkl
- i3d_baseline_8x8_IN_pretrain_400k.pkl
- i3d_nonlocal_8x8_IN_pretrain_400k.pkl

Convert these weights from caffe2 to pytorch:
```bash
python nonlocal_net.py --type [baseline, nonlocal] --frame_num [8, 32]
```
Remember to change the path to the downloaded weights in the code to match yours.


**Evaluate**                                                                     
The evaluation script saves the top5 predictions for each video as they are evaluated and provide at the end of the evaluation a report with some important metrics.
```bash
python eval_model.py --map_file [path_to_map_file] --root_data_path [path_to_videos_dir] --output_file [output_file_path] --mode [test, val] --weights_file [path_to_pretrained_weights] [--baseline]
```

## Results
| <sub>pre-trained weights file</sub> | <sub>input frames</sub> | <sub>non-local?</sub> | <sub>fully-conv?</sub> | <sub>top1 / top5</sub> | <sub>reported by authors</sub> |
| ----------------------------------- | ----------------------- | --------------------- | ---------------------- | ----------------------- |--------------------------------|
| <sub>i3d_baseline_32x2_IN_pretrain_400k.pkl</sub> | 32 |  -  |  -  | 72.3 / 90.5 | <sub>73.3 / 90.7 (in paper)</sub> |
| <sub>i3d_baseline_32x2_IN_pretrain_400k.pkl</sub> | 32 |  -  | Yes | 72.4 / 90.3 | <sub>73.3 / 90.7 (in paper)</sub> |
| <sub>i3d_nonlocal_32x2_IN_pretrain_400k.pkl</sub> | 32 | Yes |  -  | 73.9 / 91.2 | <sub>74.9 / 91.6 (in paper)</sub> |
| <sub>i3d_nonlocal_32x2_IN_pretrain_400k.pkl</sub> | 32 | Yes | Yes | 74.0 / 91.3 | <sub>74.9 / 91.6 (in paper)</sub> |
| <sub>i3d_baseline_8x8_IN_pretrain_400k.pkl</sub>  |  8 |  -  |  -  | 67.9 / 87.7 | <sub>73.4 / 90.9 (in github page)</sub> |
| <sub>i3d_baseline_8x8_IN_pretrain_400k.pkl</sub>  |  8 |  -  | Yes | 68.7 / 87.9 | <sub>73.4 / 90.9 (in github page)</sub> |
| <sub>i3d_nonlocal_8x8_IN_pretrain_400k.pkl</sub>  |  8 | Yes |  -  | 70.3 / 89.2 | <sub>74.7 / 91.6 (in github page)</sub> |
| <sub>i3d_nonlocal_8x8_IN_pretrain_400k.pkl</sub>  |  8 | Yes | Yes | 70.8 / 89.3 | <sub>74.7 / 91.6 (in github page)</sub> |

## Problems
- Models with 8x8 input achieve lower results than expected.

## References:
- [https://github.com/facebookresearch/video-nonlocal-net](https://github.com/facebookresearch/video-nonlocal-net)
- [https://github.com/feiyunzhang/i3d-non-local-pytorch](https://github.com/feiyunzhang/i3d-non-local-pytorch)
- [https://github.com/Tushar-N/pytorch-resnet3d](https://github.com/Tushar-N/pytorch-resnet3d) 
- [https://github.com/17Skye17/Non-local-Neural-Networks-Pytorch](https://github.com/17Skye17/Non-local-Neural-Networks-Pytorch)
