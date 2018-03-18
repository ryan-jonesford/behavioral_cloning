import csv
import sys
import cv2
import os.path
import os
import random
import math
from image_augmentation import gen_augmented_images, steering_adjustment
from matplotlib.mlab import csv2rec
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Activation, \
                         MaxPool2D, Dropout, Cropping2D, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import TruncatedNormal
import argparse
            
def normalize_pixel(pixel):
    # normalize color channels 
    import tensorflow as tf
    return tf.scalar_mul((1/255),pixel)

def per_image_standardization(image):
    # per image standardization option
    import tensorflow as tf
    return tf.image.per_image_standardization(image)

def yuv_conversion(x):
    # convert image to yuv colorspace
    import tensorflow as tf    
    return tf.image.rgb_to_yuv(x)

def generator(samples, batch_size=512, center_camera_only=False):
    # generator for the model. This keeps the memory usage down when running the model
    num_samples = len(samples)
    side_camera_offset = 0.008
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images_g = []
            angles_g = []
            loop = 3
            for batch_sample in batch_samples:
                source_path = []
                filenames = []
                this_path = []
                image = []
                steering_center = float(batch_sample[3])
                if center_camera_only:
                    loop = 1
                    angles_g.append(steering_center)
                else:
                    # Adjust the steering angles for the side cameras
                    steering_left = steering_adjustment(steering_center,160,320) + side_camera_offset 
                    steering_right = steering_adjustment(steering_center,-160,320) - side_camera_offset
                    angles_g.extend([steering_center, steering_left, steering_right])
                for i in range(loop):
                    source_path.append(batch_sample[i])
                    # the host system delimiter was used when adding agmented images, split with it
                    if "aug" in source_path[i]:
                        fname = source_path[i].split(host_delimiter)[-1]
                    else:
                        fname = source_path[i].split(delimiter)[-1]
                    filenames.append(fname)
                    assert "jpg" in filenames[i], "%s is not am image file" % filenames[i]
                    this_path.append(Imgdir+filenames[i])
                    image.append(cv2.cvtColor(cv2.imread(this_path[i]), cv2.COLOR_BGR2RGB))
                images_g.extend(image)              

            X_train = np.array(images_g)
            y_train = np.array(angles_g)
            yield shuffle(X_train, y_train)

def create_pilotnet_model(with_dropout=False):
    # https://arxiv.org/pdf/1704.07911.pdf
    print("running PilotNet")
    model = Sequential()
    # Crop the image, the top 60 pixels are mostly sky and mountains (aka distractions)
    # the bottom 25 pixels is fairly constant (the hood of the car), so not useful
    model.add(Cropping2D(cropping=((60,25),(25,25)), input_shape=(160,320,3)))
    # normalize the image color channels
    model.add(Lambda(normalize_pixel))
    # convert the image you yuv colorspace
    model.add(Lambda(yuv_conversion))
    # PilotNet model: conv2d x3 with stride 2x2 and kernal 5x5, conv2d x2 with stride 1x1 and kernal 3x3, fc layers x3 
    model.add(Conv2D(3, 5, strides=(2, 2), padding='same', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(Conv2D(24, 5, strides=(2, 2), padding='same', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(Conv2D(36, 5, strides=(2, 2), padding='same', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(Conv2D(48, 3, strides=(1, 1), padding='same', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu', kernel_initializer='TruncatedNormal'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # options for dropout
    if with_dropout:
        model.add(Dropout(.25))
    model.add(Dense(50, activation='relu'))
    if with_dropout:
        model.add(Dropout(.25))
    model.add(Dense(10, activation='relu'))
    if with_dropout:
        model.add(Dropout(.25))
    model.add(Dense(1))
    return model

def train(model, model_name_and_path="./model.h5"):
    # compile the model with mse and the adam optimizer
    model.compile(loss='mse', optimizer='adam')
    # create a callback for saving the model after every epoch
    checkpointer = ModelCheckpoint(filepath=model_name_and_path, verbose=1, save_best_only=False)
    # create a callback to stop the model once the val_loss stops dropping 
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # Train the model, getting the images from the disk as they are needed, so memory is preserved
    model.fit_generator(trainer, steps_per_epoch=len(train_samples)/generator_batch_size, epochs=epochs,
                        verbose=1, callbacks=[checkpointer, early_stopping],
                        validation_data=validator, validation_steps=len(validation_samples)/generator_batch_size)
    print("DONE")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driving Neural Network Model')
    parser.add_argument(
        'datadir',
        type=str,
        help='Folder name containing driving_log.csv and dir IMG with training images'
    )
    parser.add_argument(
        '--no_augmentation',
        action='store_false',
        default=True,
        help='Augment data.'
    )
    parser.add_argument(
        '--center_camera_only',
        action='store_true',
        default=False,
        help='Three images are saved, use all three or just the center camera only?'
    )
    parser.add_argument(
        '-trained_model_path',
        type=str,
        default='',
        help='Path to a trained model. Use the weights from this model'
    )
    parser.add_argument(
        '--use_dropout',
        action='store_true',
        default=False,
        help='Use dropout layers in the training model?'
    )
    args = parser.parse_args()
    augment = args.no_augmentation
    center_camera_only = args.center_camera_only
    use_dropout = args.use_dropout
    Imgdir = './'+args.datadir+'/IMG/'
    driving_log = './'+args.datadir+'/driving_log.csv'
    aug_data = './'+args.datadir+'/aug_data.csv'
    trained_model_path = args.trained_model_path
    # if a trained model path is specified, we are loading the model
    if len(trained_model_path) > 0:
        load_trained = True
    else:
        load_trained = False

    # create the model variable
    model = None

    # Path delimiter definition
    delimiter = '/'
    host_delimiter = '/'

    # Batch size to pass the generator
    generator_batch_size = 512
    # potential number of epochs, since the model quits once the loss_val stops decreasing
    # it doesn't really matter, as long as there is enough
    epochs=250
    # percentage of data taken for testing
    test_split = .25
    # adjusted steering measurements for the side camera images
    steering_correction_offset = .041

    # different path delimiter in windows
    if "win" in sys.platform: 
        host_delimiter = '\\'
        Imgdir= Imgdir.replace('/',host_delimiter)
        driving_log = driving_log.replace('/',host_delimiter)
        aug_data = aug_data.replace('/',host_delimiter)

    gen_aug = augment
    lines = []
    images = []
    m_steering_angle = []
    X_train = []
    y_train = []
    train_samples=[]
    validation_samples=[]

    with open(driving_log) as csvfile:
        reader = csv2rec(csvfile)
        for line in reader:
            lines.append(line)

    # get the delimiter used in driving_log.csv
    if len(line[0].split('\\')):
        delimiter = '\\'
    else:
        delimiter = '/'

    # check if we have already generated the augmented images
    if augment:
        import fnmatch
        # get the image directory and see if any files have "aug" in their name
        for file in os.listdir(args.datadir+host_delimiter):
            if fnmatch.fnmatch(file, 'aug*'):
                print("Augmented data already generated")
                gen_aug = False
                break
        if not gen_aug:
            with open(aug_data) as csvfile:
                reader = csv2rec(csvfile, delimiter=',')
                for row in reader:
                    lines.append(row)

    # make sure the cvs file was imported
    assert len(lines) != 0, "Array lines is empty"
    from sklearn.model_selection import train_test_split
    if augment and gen_aug:
        data = gen_augmented_images(lines, Imgdir, delimiter, host_delimiter, args.datadir, center_camera_only=center_camera_only)
    else:
        data = lines
    # split the data into training and validation sets
    train_samples, validation_samples = train_test_split(data, test_size=test_split)

    # create the training and validation generators
    trainer = generator(train_samples, center_camera_only=center_camera_only)
    validator = generator(validation_samples, center_camera_only=center_camera_only)

    # Remove old model data if it isn't being used
    if os.path.isfile('model.h5') and 'model.h5' not in trained_model_path:
        os.remove('model.h5')
    # load a trained model or create a new model
    if load_trained:
        model = load_model(trained_model_path)
    else:
        model = create_pilotnet_model(with_dropout=use_dropout)        
    # Train the model
    train(model)
    model.summary()
