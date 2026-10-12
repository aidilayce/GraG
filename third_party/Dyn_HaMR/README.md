<p align="center">

  <h1 align="center">Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera</h1>
  <p align="center">
    <a href="https://github.com/ZhengdiYu"><strong>Zhengdi Yu</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=QKOH5iYAAAAJ&hl=en&oi=ao"><strong>Stefanos Zafeiriou</strong></a>
    ·
    <a href="https://tolgabirdal.github.io/"><strong>Tolga Birdal</strong></a>
  </p>
  <p align="center">
    <strong>Imperial College London</strong>
  </p>
  </p>
    <p align="center">
    <strong>CVPR 2025 (Highlight)</strong>
  </p>

  <div align="center" style="display: flex; justify-content: center; gap: 20px;">
    <img src="./assets/wild.gif" alt="Wild GIF" width="45%" style="border-radius: 10px;">
    <img src="./assets/global.gif" alt="Global GIF" width="45%" style="border-radius: 10px;">
  </div>

  <p align="center">
    <a href='https://arxiv.org/abs/2412.12861'>
      <img src='https://img.shields.io/badge/Arxiv-2412.12861-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
    </a>
    <a href='https://arxiv.org/pdf/2412.12861'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green'>
    </a>
    <a href='https://dyn-hamr.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue'></a>
    <a href='https://youtu.be/n25NGIWiA7M'>
      <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a>
      <img src="https://visitor-badge.laobi.icu/badge?page_id=ZhengdiYu.Dyn-HaMR&left_color=gray&right_color=orange">
    </a>
  </p>
</p>


## Introduction
<img src="./assets/teaser.png">

We propose **Dyn-HaMR** to reconstruct 4D global hand motion from monocular videos recorded by dynamic cameras in the wild, as a remedy for the motion entanglement in the wild.

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#get-started">Get Started</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>
<br/>

## News :triangular_flag_on_post:
- [2025/11/20] 🚀 **Major Update**: 
  - **Integrated [VIPE](https://github.com/nv-tlabs/vipe)** for camera estimation, significantly improving reconstruction quality over DROID-SLAM
  - **Enhanced Hand Tracker** with robust hallucination prevention and handedness correction for better hand tracking and **significantly** Improved temporal consistency. Please `pip install ultralytics==8.1.34` since YOLO is using in this version (Thanks to [WiloR](https://github.com/rolpotamias/WiLoR)). Please download the checkpoint from [here](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt) and put it under `third-party/hamer/pretrained_models`.

  See comparison below:

  <table>
    <tr>
      <th>Before: Jitter from DROID-SLAM</th>
      <th>New: VIPE + enhanced HaMeR (Recommended)</th>
    </tr>
    <tr>
      <td align="center">
        <img src="./assets/droid_result.gif" width="100%">
      </td>
      <td align="center">
        <img src="./assets/vipe_result.gif" width="100%">
      </td>
    </tr>
  </table>

  <table>
    <tr>
      <th>Before: Handedness shifting with original HaMeR</th>
      <th>New: Enhanced hand tracker (Recommended)</th>
    </tr>
    <tr>
      <td align="center">
        <img src="./assets/handedness1.gif" width="100%">
      </td>
      <td align="center">
        <img src="./assets/handedness2.gif" width="100%">
      </td>
    </tr>
  </table>

  <table>
    <tr>
      <th>Before: Handedness shifting with original HaMeR</th>
      <th>New: Enhanced hand tracker (Recommended)</th>
    </tr>
    <tr>
      <td align="center">
        <img src="./assets/handedness3.gif" width="100%">
      </td>
      <td align="center">
        <img src="./assets/handedness4.gif" width="100%">
      </td>
    </tr>
  </table>

- [2025/06/04] Code released.
- [2024/12/18] [Paper](https://arxiv.org/abs/2412.12861) is now available on arXiv. ⭐

## Installation

### Environment setup
1. Clone the repository with submodules with the following command:
   ```bash
    git clone --recursive https://github.com/ZhengdiYu/Dyn-HaMR.git
    cd Dyn-HaMR
    ```
    You can also run the following command to fetch the submodules:
    ```bash
    git submodule update --init --recursive .
    ```
  
2. To set up the virtual environment for Dyn-HaMR, we provide the integrated commands in `\scripts`. You can create the environment from
    ```bash
    source install_pip.sh
    ```

   Or, alternatively, create the environment from conda:   
    ```bash
    source install_conda.sh
    ```

### Model checkpoints download
Please run the following command to fetch the data dependencies. This will create a folder in `_DATA`:
  ```bash
  source prepare.sh
  ```
After processing, the folder layout should be:
```
|-- _DATA
|   |-- data/  
|   |   |-- mano/
|   |   |   |-- MANO_RIGHT.pkl
|   |   |-- mano_mean_params.npz
|   |-- BMC/
|   |-- hamer_ckpts/
|   |-- vitpose_ckpts/
|   |-- <SLAM model .pkl>
```

### Prerequisites
We use [MANO](https://mano.is.tue.mpg.de) model for hand mesh representation. Please visit the [MANO website](https://mano.is.tue.mpg.de) for registration and the model downloading. Please download `MANO_RIGHT.pkl` and put under the `_DATA/data/mano` folder.

## Get Started🚀

### Preparation
Please follow the instructions [here](https://github.com/MengHao666/Hand-BMC-pytorch) to calculate the below `.npz` files in order `dyn-hamr/optim/BMC/`:
```
|-- BMC
|   |-- bone_len_max.npy
|   |-- bone_len_min.npy
|   |-- CONVEX_HULLS.npy
|   |-- curvatures_max.npy
|   |-- curvatures_min.npy
|   |-- joint_angles.npy
|   |-- PHI_max.npy
|   |-- PHI_min.npy
```

> [!NOTE]
> If accurate camera parameters are available, please follow the format of `Dyn-HaMR/test/dynhamr/cameras/demo/shot-0/cameras.npz` to prepare the camera parameters for loading. Similarly, you can use Dyn-HaMR to refine and recover the hand mesh in the world coordinate system initializing from your own 2D & 3D motion data.

### Customize configurations
| Config | Operation |
|--------|-----------------|
| GPU | Edit in [`<CONFIG_GPU>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/config.yaml#L56) |
| Video info | Edit in [`<VIDEO_SEQ>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/data/video.yaml#L5) |
| Interval | Edit in [`<VIDEO_START_END>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/data/video.yaml#L16-L17) |
| Optimization configurations | Edit in [`<OPT_WEIGHTS>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/optim.yaml#L29-L49) |
| General configurations | Edit in [`<GENERAL_CONFIG>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/config.yaml) |

### Fitting on RGB-(D) videos 🎮
To run the optimization pipeline for fitting on arbitrary RGB-(D) videos, please first edit the path information here in `dyn-hamr/confs/data/video.yaml`, where `root` is the root folder to all of your datasets. `video_dir` is the corresponding folder that contains the videos. The key `seq` represents the video name you wanted to process. For example, you can run the following command to recover the global motion for `test/videos/demo1.mp4`:


#### 🌟 Using VIPE for Camera Estimation (Recommended)
For significantly better camera estimation quality, use VIPE instead of DROID-SLAM:

```bash
python run_opt.py data=video_vipe run_opt=True data.seq=demo1 is_static=False
```

#### 🌟 Using original DROID-SLAM for Camera Estimation
```bash
python run_opt.py data=video run_opt=True data.seq=demo1 is_static=<True or False>
```

VIPE will automatically run if results are not found. Make sure you have:
1. Installed VIPE in `third-party/vipe/` with conda environment named `vipe`
2. Set `src_path` in `dyn-hamr/confs/data/video_vipe.yaml` to your video file

By default, the camera parameters will be predicted during the process and assumes a moving camera (`is_static=False`). If your video is recorded with a static camera, you can add `is_static=True` for more stable optimization. The result will be saved to `outputs/logs/video-custom/<DATE>/<VIDEO_NAME>-<tracklet>-shot-<shot_id>-<start_frame_id>-<end_frame_id>`. After optimization, you can specify the output log dir and visualize the results by running the following command:
```
python run_vis.py --log_root <LOG_ROOT>
```
This will visualize all log subdirectories and save the rendered videos and images, as well as saved 3D meshes in the world space in `<LOG_ROOT>`. Please visit `run_vis.py` for further details. Alternatively, you can also use the following command to run and visualize the results in one-stage:
```
python -u run_opt.py data=video_vipe run_opt=True run_vis=True is_static=<True of False>
```
As a multi-stage pipeline, you can customize the optimization process. Add `is_static=True` for static camera videos. Adding `run_prior=True` can activate the motion prior in stage III. Please note that in the current version, each motion chunk size needs to be set to 128 to be compatible with the original setting of HMP only when the prior module is activated.

### Blender Addon
Coming soon.

## Acknowledgements
The PyTorch implementation of MANO is based on [manopth](https://github.com/hassony2/manopth). Part of the fitting and optimization code of this repository is borrowed from [SLAHMR](https://github.com/vye16/slahmr). For data preprocessing and observation, [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [HaMeR](https://github.com/geopavlakos/hamer/) is used for 2D keypoints detection and MANO parameter initilization. For camera motion estimation, we support [VIPE](https://github.com/facebookresearch/vipe) (recommended), [DPVO](https://github.com/princeton-vl/DPVO), and [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM). For biomechanical constraints and motion prior, we use the code from [here](https://github.com/MengHao666/Hand-BMC-pytorch) and [HMP](https://hmp.is.tue.mpg.de/). We thank all the authors for their impressive work!

## License
Please see [License](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/LICENSE) for details of Dyn-HaMR. This code and model are available only for non-commercial research purposes as defined in the LICENSE (i.e., MIT LICENSE). Note that, for MANO you must agree with the LICENSE of it. You can check the LICENSE of MANO from https://mano.is.tue.mpg.de/license.html.

## Citation
```bibtex
@inproceedings{yu2025dynhamr,
  title={Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera},
  author={Yu, Zhengdi and Zafeiriou, Stefanos and Birdal, Tolga},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
}
```

## Contact
For any technical questions, please contact z.yu23@imperial.ac.uk or ZhengdiYu@hotmail.com.
