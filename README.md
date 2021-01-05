# yolov3-hart
Code implementation of the work "Real-time adaptive object detection and tracking for autonomous vehicles"

## Adaptive Stage Switch Model (YOLOv3 + HART) [2D/3D] | Keras/Tensorflow v2.0

> If you consider using this code, consult our paper on the references tab for additional information on our model and proposed adaptive system.

## Quick start
1. Clone this file
```bashrc
$ git clone https://github.com/Hffmann/yolov3-hart.git
```
2.  You are supposed  to install some dependencies before getting out hands with these codes.
```bashrc
$ cd yolov3-hart
$ pip install -r ./data/requirements.txt
```
3. The yolo.h5 file can be generated using the YAD2K repository here: https://github.com/allanzelener/YAD2K

Steps how to do it on windows:

- Clone the above repository to your computer

- Download the yolo.weights file from here: http://pjreddie.com/media/files/yolo.weights

- Download the yolo.cfg file form here: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg

- Copy the downloaded weights and cfg files to the YAD2K master directory

- Run the following command on the shell and the h5 file will be generated.
```bashrc
$ python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5 
```

- The YOLOv3 model training can be performed as described on https://github.com/YunYang1994/tensorflow-yolov3

4. Download the HART weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy and put the file in the tracker/checkpoints folder. 

- The HART approach training can be performed as described on https://github.com/akosiorek/hart

5. [3D-Deepbox] Download the weights file from https://drive.google.com/file/d/1yAFCmdSEz2nbYgU5LJNXExtsS0Gvt66U/view?usp=sharing or follow the training process from https://github.com/smallcorgi/3D-Deepbox and put it in the model_data folder.

6. After downloading or training all weights on your own dataset, run the testing script :
```bashrc
$ python main.py
```

- Parse the arguments from the file main.py upon the main command execution to adjust the model to your use.

- Opt. Download the video input from https://drive.google.com/file/d/1enfe_tcHLkAJGL0Y-PPNw4Hb6FYg8_om/view?usp=sharing and put it in the video_input folder

<p align="center">
    <img width="100%" src="" style="max-width:100%;">
    </a>
</p>

## References

[- **`Real-time adaptive object detection and tracking for autonomous vehicles`**](https://ieeexplore.ieee.org/document/9259200)<br>

[-**`tensorflow-yolov3`**](https://github.com/YunYang1994/tensorflow-yolov3)<br>

[-**`hart`**](https://github.com/akosiorek/hart)<br>

[- **`3D-Deepbox`**](https://github.com/smallcorgi/3D-Deepbox)<br>


