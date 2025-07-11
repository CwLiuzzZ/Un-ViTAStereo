# Un-ViTAStereo
This is the official repo for our work 'Integrating Disparity Confidence Estimation into Relative Depth Priors-Guided Unsupervised Stereo Matching'.

## Setup
We built and ran the repo with CUDA 12.1, Python 3.10.12, and Pytorch 2.4.0. For using this repo, please follow the instructions below:
```
pip install -r requirements.txt
```

If you have any problem with installing mmsegmentation, xFormers packages, please follow the guidance in [DINOv2](https://github.com/facebookresearch/dinov2).

## Pre-trained models

To use our models, you have to first download the DINOv2 pretrained weights "ViT-L/14 distilled" from [DINOV2](https://github.com/facebookresearch/dinov2), and is supposed to be under dir like: ```toolkit/models/dinoV2/dinov2_vitl14_pretrain.pth```

and the DepthAnything pretrained weights "Depth-Anything-V2-Large" from [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2), and is supposed to be under dir like:```toolkit/models/Depth_anything_v2/depth_anything_v2_vitl.pth```

Pretrained models leading to our [SoTA KITTI benchmark results](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) can be downloaded from [google drive](https://drive.google.com/file/d/1KWsLsyMlF9JkN-qnQt2BInaUSOzkV8dO/view?usp=sharing), and is supposed to be under dir like: `toolkit/models/UnViTAStereo/unsupervised_benchmark.ckpt`.

## Dataset Preparation
To train/evaluate ViTAStereo, you will need to download the required datasets.

* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#:~:text=on%20Academic%20Torrents-,FlyingThings3D,-Driving) (Includes FlyingThings3D, Driving)
* [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
* [Middlebury](https://vision.middlebury.edu/stereo/data/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)
* [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
* [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

You can create symbolic links to wherever the datasets were downloaded in the `$root/datasets` folder:

```shell
ln -s $YOUR_DATASET_ROOT datasets
```

Our folder structure is as follows:

```
├── datasets
    ├── ETH3D
    │   ├── testing
    │       ├── lakeside_1l
    │       └── ...
    │   └── training
    │       ├── delivery_area_1l
    │       └── ...
    │
    ├── KITTI
    │   ├── 2012
    │   │   ├── testing
    │   │   └── training
    │   └── 2015
    │       ├── testing
    │       └── training
    ├── middlebury
    │   ├── 2005
    │   ├── 2006
    │   ├── 2014
    │   ├── 2021
    │   └── MiddEval3
    └── sceneflow
        ├── disparity
        │   ├── 15mm_focallength
        │   │   ├── scene_backwards
        │   │   └── scene_fowwards
        │   ├── 35mm_focallength
        │   ├── monka
        │   ├── TEST
        │   └──-TRAIN
        ├── frames_cleanpass
        │   ├── 15mm_focallength
        │   │   ├── scene_backwards
        │   │   └── scene_fowwards
        │   └── 35mm_focallength
        ├── frames_finalpass
        │   ├── TRAIN
        │   │   ├── A
        │   │   ├── B
        │   │   └── C
        │   ├── TEST
        │   └── monka
```
## Training & Evaluation
The inferences of depth priors generating, model training and model evaluating are integrated in the  `toolkit/depth_main.py`

To use these inferences, you should be under the '''toolkit''' directory, and use the following instruction:

```python toolkit.py```

## Acknowledgment

Some of this repo come from [IGEV-Stereo](https://github.com/gangweiX/IGEV),[GMStereo](https://github.com/autonomousvision/unimatch), and [DINOv2](https://github.com/facebookresearch/dinov2).
