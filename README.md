# OnlineAction
Pytorch implementation of [**Non-local Neural Networks**](https://arxiv.org/abs/1711.07971).

## Results
Pre-trained weights downloaded from [https://github.com/facebookresearch/video-nonlocal-net](https://github.com/facebookresearch/video-nonlocal-net).

| <sub>pre-trained weights file</sub> | <sub>input frames</sub> | <sub>non-local?</sub> | <sub>top1 / top5</sub> | <sub>reported by authors</sub> | Ok? |
| ------------------------------ | ----------------------- | --------------------- | ---------------------- | ------------------- | ------|
| <sub>i3d_baseline_32x2_IN_pretrain_400k.pkl</sub> | 32 | - | 72.3 / 90.5 | <sub>73.3 / 90.7 (in paper)</sub> | Ok! |
| <sub>i3d_nonlocal_32x2_IN_pretrain_400k.pkl</sub> | 32 | Yes | 63.0 / 84.4 | <sub>74.9 / 91.6 (in paper)</sub> | - |
| <sub>i3d_baseline_8x8_IN_pretrain_400k.pkl</sub> | 8 | - | 68.0 / 87.5 | <sub>73.4 / 90.9 (in github page)</sub> | - |
| <sub>i3d_nonlocal_8x8_IN_pretrain_400k.pkl</sub> | 8 | Yes | 55.7 / 78.8 | <sub>74.7 / 91.6 (in github page)</sub> | - |

## Problems
- Non-local models can not achieve expected results
- Models with 8x8 input can not achieve expected results

## References:
- [https://github.com/facebookresearch/video-nonlocal-net](https://github.com/facebookresearch/video-nonlocal-net)
- [https://github.com/feiyunzhang/i3d-non-local-pytorch](https://github.com/feiyunzhang/i3d-non-local-pytorch)
- [https://github.com/Tushar-N/pytorch-resnet3d](https://github.com/Tushar-N/pytorch-resnet3d) 
- [https://github.com/17Skye17/Non-local-Neural-Networks-Pytorch](https://github.com/17Skye17/Non-local-Neural-Networks-Pytorch)
