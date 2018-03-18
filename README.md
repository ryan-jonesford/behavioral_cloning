# Behaviorial Cloning Project

Overview
---
This repository contains files for implementing the Behavioral Cloning Project for Udacity's Self Driving Car NanoDegree program.

I used Nvidia developed PilotNet CNN model architecture and training datasets gathered from Udacity’s driving simulator to successfully train the car to autonomously drive around two provided tracks. 
---

[//]: # (Image References)
[pilotnet_arch]: ./writeup/pilotnet_arch.png "PilotNet Architecture"
[bridge]: ./writeup/bridge.PNG "Example of crossing bridge"
[aug_image]: ./writeup/Augmented_example.jpg "Example of augmented image"
[side_camera_offset]: ./writeup/side_camera_offset.png "Side Camera Offset"

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

Collect training data with the simulator and store them in a new directory that is in the same directory as model.py (ex: ./model.py, ./new_dir/simdata). 
```sh
python model.py

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
I also recorded correctional runs. For example: have the car start by looking like it is going off the road, begin recording with the steering angle correcting the car and accelerate until the car is back on the road. End the recording and repeating these steps again many, many times on both tracks. This is to help teach the car how to stay on the road when something goes wrong and it starts to veer off. 

After augmentation, 25% of this training data was split off as validation data.

### Model Architecture

![PilotNet][pilotnet_arch]

I used the [PilotNet Architecture](https://arxiv.org/pdf/1704.07911.pdf) developed by the Nvidia team with some added dropout layers to help prevent overfitting. Prior to putting the data into the architecture the images where cropped to get rid of extraneous data (the hood of the car, skyline, and pixels off the sides to accomodate for image shifts), each color channel normalized and then converted to YUV color-space. 

A generator was used which allowed batches of images to be put in memory instead of holding all the training data in memory all at once. 

The model was trained with an Adam optimizer and contained an early exit function that stops training once the val_loss variable stops decreasing for two times in a row. This normally kicked in after 23-26 epochs. 

The PilotNet architecture is comprised of 5 convolutional layers and 3 fully connected layers that converge to output a single value, the steering angle. The first 3 convolutional layers are 5x5 kernels with a stride of 2, followed by 2 convolutional layers with a stride of 1 followed by a flatting layer and three fully connected layers. Between the fully connected layers I added the option to enable a dropout of 25% (which I used for my model).  The dropout layers help prevent over-fitting, which the Nvidia team didn’t really need due to running on real world data instead of simulated track data that produces very simular data over multiple runs. 

### Determining Steering Angle

To determine the steering angle of the vehicle, it takes more than just running the training data through the CNN. A couple variables that need to be delt with are which cameras to use (the system has 3) and how to account for all the straight sections of the track where the model doesn’t learn anything about controlling the car because the steering angle doesn’t change much.

#### Side Cameras and Augmenting Images

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

With no offset it performed okay, but it had a high-ish amplitude, high frequency oscillation. Taking note of the amplitude of the oscillation I used that to come up with an offset of .2° or .008 after normalizing by 25°.  

I retrained the model and found that the oscillation had all but disappeared and so I used the .008 number as my constant offset that I added back into my steering calculation for the side cameras (positive for the left camera and negative for the right). 

After this, it was down to training the model on the full dataset. The images collected from the training/recording session were saved into a folder and the augmentation routine duplicated these images with a random brightness or image shift applied. A new csv file was created that listed the paths and new steering angle to the augmented images which was added to the overall dataset before running the model. 

### Other Modifications

drive.py uses a simple PI controller for regulating the speed of the car, but didn’t have anything for smoothing out the steering angle. Since the steering angle is computed from the PilotNet model, it can be a little jerky and oscillations are present. To help alleviate that, I added in a weighted moving average  filter with a window of 15 samples. 

### Results

The model created a workable control structure that enabled the simulated car to successfully drive around both tracks without going off road or taking unsafe actions (e.g. veering off into the opposing lane). The harder track, turned out to have better results than the easy track. On the easy track the car had some major oscillations compared to the harder one. I think the reason for this is that since the track is much wider and I drove the center of the track, there were less features for the model to work off of, meaning that it had to rely more on the correction data that I collected. 

#### Summary Of PilotNet Model

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