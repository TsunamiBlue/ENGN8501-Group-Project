# Motion Transfer with SIM-GAN Refinement

Written by Jieli Zheng, Xuan Feng and Siwei Lin, including temporal smoothing and SIM-GAN.<br>
<br>
We train and evaluate on Ubuntu 16.04, so if you are under other environment, you should set `nThreads=0` in `./src/config/train_opt.py`.

## Reference:

[nyoki-mtl](https://github.com/nyoki-mtl) pytorch-EverybodyDanceNow

[Lotayou](https://github.com/Lotayou) everybody_dance_now_pytorch

[yanx](https://github.com/yanx27/EverybodyDanceNow_reproduce_pytorch) EverybodyDanceNow_reproduce_pytorch

[AlexHex7](https://github.com/AlexHex7/SimGAN_pytorch) SimGAN_pytorch

## Loading pre-trained models and source video
* Download vgg19-dcbb9e9d.pth [here](https://drive.google.com/file/d/1JG-pLXkPmyx3o4L33rG5WMJKMoOjlXhl/view?usp=sharing) and put it in `./src/pix2pixHD/models/`  <br>

* Download pose_model.pth [here](https://drive.google.com/file/d/1DDBQsoZ94N4NRKxZbwyEXt7Tz8KqgS_w/view?usp=sharing) and put it in `./src/PoseEstimation/network/weight/`   <br>

* Source video can be download from [here](https://drive.google.com/file/d/1drRBJypNGqOZV9WFutEzDYXkEelUjZXh/view?usp=sharing) or use your own source video

* Download pre-trained vgg_16 for face enhancement [here](https://drive.google.com/file/d/180WgIzh0aV1Aayl_b1X7mIhVhDUcW3b1/view?usp=sharing) and put in `./face_enhancer/`

## How to use our project

#### Make source pictures
* Put source video mv.mp4 in `./data/source/` and run `A_make_source_images.py`, the label images and coordinate of head will save in `./data/source/test_label_ori/` and `./data/source/pose_souce.npy` (will use in step6). If you want to capture video by camera, you can directly run `./src/utils/save_img.py`
#### Make target pictures
* Rename your own target video as mv.mp4 and put it in `./data/target/` and run `B_make_target_images.py`, `pose.npy` will save in `./data/target/`, which contain the coordinate of faces (will use in step F).

#### Train and use generative network
* Run `C_train_target_images.py` and check loss and full training process in `./checkpoints/`

* If you break the training and want to continue last training, set `load_pretrain = './checkpoints/target/` in `./src/config/train_opt.py`
* Run `D_normalization.py` rescale the label images, you can use two sample images from `./data/target/train/train_label/` and `./data/source/test_label_ori/` to complete normalization between two skeleton size
* Run `E_transfer.py` and get results in `./results`
#### Face enhancement network

#### Train and use face enhancement network
* Find the directory `./face_enhancer`.
* Under the face enhancer directory, run `prepare.py` and check the results in `data` directory at the root of the repo (`data/face/test_sync` and `data/face/test_real`).
* Under the face enhancer directory, run `main.py` to rain the face enhancer. Then run `enhance.py` to obtain the results <br>
This is comparision in original (left), generated image before face enhancement (median) and after enhancement (right). FaceGAN can learn the residual error between the real picture and the generated picture faces.


#### SIM-GAN global enhancement
* Find the directory `./src/simGAN`.
* Under the simGAN directory, run `train.py` to train the simGAN network for current target person.
* Under the simGAN directory, run `main.py` to evaluate the simGAN network results.

#### Gain results
* Go back to the root dir and run `F_img2vid.py` to create a gif out of the resulting images.


## Contained modules:

- Pose estimation
    - [x] Pose
    - [x] Face
    - [ ] Hand
- [x] pix2pixHD
- [x] Local enhancer based on FaceGAN
- [x] Temporal smoothing
- [x] Glabal enhancer based on SIM-GAN

## Environments
Ubuntu 16.04 <br>
Python 3.7.3 <br>
Pytorch 1.9.1  <br>
OpenCV 4.5.1.48  <br>