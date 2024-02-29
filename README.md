# Cal_PnP

This repository contains our Python implementation of semi-automatic camera calibration based on Perspective-n-Point (PnP). It can be used to compute the homography matrix and manually correct radial distortion. 

## Introduction

This package is designed for computing camera homography matrix. The input includes a set of 3D points on the 3D ground plane and the corresponding 2D pixel points. We adopt the Perspective-n-Point (PnP) for camera calibration. The user can choose from different approaches such as (1) a regular method using all the points, (2) RANSAC-based robust method or (3) Least-Median robust method. The package also includes tools for manual 2D points selection. 

## Coding Structure

1. `./src/` folder: Source code
2. `./data/` folder: Example input files
   1. `cfg.json`: Configuration parameters in JSON format
   2. `frm.jpg`: Input 2D frame image
   3. `pic0.jpg`: Input 3D frame image

## How to Build

1. Download the OpenCV library.
 
   ```
   pip3 install opencv-python
   ```
2. Download the NumPy library.

   ```
   pip3 install numpy
   ```

## How to Use
1. Set the corresponding input/output paths in the configuration file if necessary. 

2. If the user want to select the 2D and 3D pixel points on the image, set `calSel2dPtFlg` to 1. A new window called `selector of 2D and 3D points` pops out. The user can click on the image to select each 2D and 3D points. A blue or red circle stands for each click. After the selection is done, click `o`. Please make sure the number of selected 2D points should be the same as the 3D points (in the same order). During the selection, if mis-clicking on wrong places, the user can press `r`. All the markers will be cleared, and s/he can start over.

<div align="center">
    <img src="/pic/pic3.jpg", width="900">
</div>

3. If there exists radial distortion, the frame image can be manually corrected by by setting `calDistFlg` to 1 and providing the distortion coefficients (`calDistCoeff`) and intrinsic camera parameters (`calFocLen` and `calPrinPt`). 
4. The user can choose to use different methods for camera calibration by setting `calTyp`, whose introduction is at this [link](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findhomography). Note that when using RANSAC-based robust method, the threshold parameter needs to be provided at  `calRansacReprojThld`. When setting `calTyp` as -1, all the methods will be conducted and evaluated, the one with the minimum reprojection error will be chosen. 
5. The output text file shows the 3x3 homography matrix at the first line. If the correction of radial distortion is conducted, the 3x3 intrinsic parameter matrix and 1x4 distortion coefficients are also printed. Finally, the reprojection error in pixels is printed as well. 
6. The user can choose to output the display image (with a colorful grid on the 3D plane) by setting `outCalDispFlg` to 1. The blue circles show the input 2D pixel points. The red circles show the corresponding points back projected from 3D. The distance between their centers indicates the reprojection error. 

<div align="center">
    <img src="/pic/pic2.jpg", width="640">
</div>

7. Run the main.py program.

   ```
   python3 main.py
   ```

## References

The code was applied to the generation of baseline camera calibration results for the CityFlow benchmark used in the [AI City Challenge Workshops](https://www.aicitychallenge.org/). Please consider to cite these papers in your publications if it helps your research:

    @inproceedings{Tang18AIC,
      author = {Zheng Tang and Gaoang Wang and Hao Xiao and Aotian Zheng and Jenq-Neng Hwang},
      title = {Single-camera and inter-camera vehicle tracking and {3D} speed estimation based on fusion of visual and semantic features},
      booktitle = {Proc. CVPR Workshops},
      pages = {108--115}, 
      year = {2018}
    }

    @misc{Tang17AIC,
      author = {Zheng Tang and Gaoang Wang and Tao Liu and Young-Gun Lee and Adwin Jahn and Xu Liu and Xiaodong He and Jenq-Neng Hwang},
      title = {Multiple-kernel based vehicle tracking using {3D} deformable model and camera self-calibration},
      howpublished = {arXiv:1708.06831},
      year = {2017}
    }

Forked and modified the code from [Zheng (Thomas) Tang](https://github.com/zhengthomastang).

## Disclaimer

For any question you can contact [William Chen](https://github.com/will211).
