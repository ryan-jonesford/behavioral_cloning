# Behaviorial Cloning Project

Overview
---
This repository contains files for implementing the Behavioral Cloning Project for Udacity's Self Driving Car NanoDegree program.

### Dependencies

Files: 
* README (This file)
* model.py (Script used to create and train the model)
* drive.py (Script to drive the car)
* model.h5 (A trained Keras model ran with dropout enabled)
* image_augmentation.py (File holding functions to augment images and calculate steering angles)
* video.mp4 (A video recording of the vehicle driving autonomously around the track for at least one full lap)

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The simulator can be downloaded from the Udacity classroom.

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

### Usage

Collect training data with the simulator and store them in a new directory that is in the same directory as model.py (ex: ./model.py, ./new_dir/simdata). Then to run the model use
```sh
usage: model.py [-h] [--no_augmentation] [--center_camera_only] [-trained_model_path TRAINED_MODEL_PATH] [--use_dropout] datadir

Driving Neural Network Model

positional arguments:
  datadir               Folder name containing driving_log.csv and dir IMG
                        with training images

optional arguments:
  -h, --help            show this help message and exit
  --no_augmentation     Augment data.
  --center_camera_only  Three images are saved, use all three or just the center camera only?
  -trained_model_path   TRAINED_MODEL_PATH
                        Path to a trained model. Use the weights from this model
  --use_dropout         Use dropout layers in the training model?
```
This runs the model with image augmentation generation, using all three cameras and **NOT** using dropout in the model.

### Training Data

Collecting good training data is important. The simulator provides two pre-made tracks, an easy one and hard one. The way that I collected my training data was to record three forward laps and three laps in the reverse direction on the easy track and two forward laps and two laps in the reverse direction on the hard track. 
I also recorded correctional runs. For example: have the car start by looking like it is going off the road, begin recording with the steering angle correcting the car and accelerate until the car is back on the road. End the recording and do it again many, many times. 

### Model Architecture

I used the [PilotNet Architecture](https://arxiv.org/pdf/1704.07911.pdf) developed by the Nvidia team with some added dropout layers. Prior to putting the data into the architecture the images where cropped to get rid of extranioius data (the hood of the car, skyline, and pixels off the sides to accomodate for image shifts). 

[//]: # (Image References)

[bridge]: ./writeup/bridge.PNG "Example of crossing bridge"
[aug_image]: ./writeup/Augmented_example.jpg
[side_camera_offset]: ./writeup/side_camera_offset.png

Using the side cameras has some challenges to it.
First, the system uses the side cameras as if they are the front camera to predict the steering angle. 
Second,  they aren’t on the same axis as the center camera, they look off to the side, rotated some amount from normal. 

![bridge][bridge]

To determine what a reasonable estimate for that rotational constant is I picked a rather homogeneous piece of track to train the car on, the bridge. What’s good about the bridge (pictured above) is that it is straight, and has a couple of well defined features that are constant for many frames. Passing over the bridge thrice forward and thrice backward gave me a good dataset to work off of. 

If you think of the side cameras as shifted to one side by a number of pixels, one could use some simple geometric math to determine that the new angle is equal to 

You get this by using the origin as the center, and the steering angle is the angle formed by the angle originating from the origin to the target. 
This method of finding the steering angle shifted some pixels is really useful also when augmenting data. I augmented my data with random shifts in brightness and random shifts of the image left to right. The image below is exemplary of the augmentation that I performed.

![Brightness and pixel shifted image][aug_image]

![Why there is an offset][side_camera_offset]

Back to the bridge. Now that we have a formula for figuring out the change in angle when the image is offset we need to figure out the constants, namely how far the camera axis’ are rotated from the origin compared to the center; this is what I'm calling the constant offset and the pixels sifted. I guessed that the pixels shifted is 160 based on what the images looked like and to get an initial guess of the constant offset of the side cameras I ran the trainer without an offset and then took the car to the bridge and watched how it performed.

With no offset it performed okay, but it had a high amplitude, high frequency oscillation. Taking note of the amplitude of the oscillation I used that to come up with an offset of .2° or .008 after normalizing by 25°.  

I retrained the model and found that the oscillation had all but disappeared and so I used the .008 number as my constant offset that I added back into my steering calculation for the side cameras (positive for the left camera and negative for the right). 

After this, it was down to training the model on the full dataset. The images collected from the training/recording session were saved into a folder and the augmentation routine duplicated these images with a random brightness or image shift applied. A new csv file was created that listed the paths and new steering angle to the augmented images which was added to the overall dataset before running the model. 

## PilotNet Model Used ##
*PilotNet Architecture With Dropout layers added*
**Layer (type)** | **Output Shape** | **Param #**
--- | --- | ---
cropping2d_1 (Cropping2D)  |  (None, 75, 270, 3)    |    0
lambda_1 (Image Normalization)     |       (None, 75, 270, 3)    |    0
lambda_2 (YUV Conversion)     |       (None, 75, 270, 3)    |    0
conv2d_1 (Conv2D)     |       (None, 38, 135, 3)    |    228
conv2d_2 (Conv2D)     |       (None, 19, 68, 24)    |    1824
conv2d_3 (Conv2D)     |       (None, 10, 34, 36)    |    21636
conv2d_4 (Conv2D)     |       (None, 10, 34, 48)    |    15600
conv2d_5 (Conv2D)     |       (None, 10, 34, 64)    |    27712
flatten_1 (Flatten)     |     (None, 21760)      |       0
dense_1 (Dense)       |       (None, 100)        |       2176100
dropout_1 (Dropout 25%)     |     (None, 100)     |          0
dense_2 (Dense)      |        (None, 50)        |        5050
dropout_2 (Dropout 25%)     |     (None, 50)       |         0
dense_3 (Dense)       |       (None, 10)       |         510
dropout_3 (Dropout 25%)     |     (None, 10)       |         0
dense_4 (Dense)      |        (None, 1)       |          11


https://hoganengineering.wixsite.com/randomforest/single-post/2017/03/14/How-To-Triangle