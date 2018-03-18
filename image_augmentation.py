import math
import cv2
import numpy as np
import sys 
import os.path
import os
import random
import csv

def change_brightness(img, value):
    # modify image birghtness by value
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    if value > 0:
        # avoid saturation
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        # 0 is the min, v channel is type uint8
        lim = np.int8(abs(value)).astype('uint8')
        v[v < lim] = 0
        v[v >= lim] -= np.int8(abs(value)).astype('uint8')

    # merge the channels back together
    final_hsv = cv2.merge((h, s, v))
    # convert the image back to RGB colorspace
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img

def steering_adjustment(steering_angle, pixel_shift, image_width):
    # normalize the pixel shift to the width of the image
    norm_pixel_shift = pixel_shift/image_width
    # Calculate and return the new angle
    return math.atan((math.tan(steering_angle)+norm_pixel_shift))

def gen_augmented_images(samples, Imgdir, delimiter, host_delimiter, data_dir, center_camera_only=False):
    # Generate images to augment the data
    # default to using all three cameras
    data_dir = '.'+host_delimiter+data_dir+host_delimiter
    loop = 3
    aug_samples = []
    # image generation progress counter
    progress_count = 0
    if center_camera_only:
        # only using 1 camera so no need to loop through the loop more than once per line
        loop = 1
    for sample in samples:
        # output to inform user of the progress
        sys.stdout.write("\rGenerating Augmented Images: %d%%" % (progress_count/len(samples)*100))
        sys.stdout.flush()
        aug_line = ()
        source_path = []
        filenames = []
        this_path = []
        image = []
        augname = []
        # Get the steering angle
        steering_center = float(sample[3])
        aug_x, aug_y = None, None
        for i in range(loop):
            # Get the path to the file
            source_path.append(sample[i])
            fname = source_path[i].split(delimiter)[-1]
            filenames.append(fname)
            assert "jpg" in filenames[i], "%s is not am image file" % filenames[i]
            aug_y = steering_center
            this_path.append(Imgdir+filenames[i])
            # cv2.imread reads the file in as BGR, need to convert it to RGB
            image.append(cv2.cvtColor(cv2.imread(this_path[i]), cv2.COLOR_BGR2RGB))
            

            # brightness augmentation
            # change the brightness randomly (-100, 75 in multiples of 5)
            new_img = change_brightness(image[i], random.randrange(-100, 75, 5))

            # horizontal translation (shift) augmentation
            # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#translation
            # translate (shift) the image horizontally by a random amount of pixels (-30 to 30 even amount only)
            pixel_shift = random.randrange(-25,25,2)
            rows,cols,channels = new_img.shape
            M = np.float32([[1,0,pixel_shift],[0,1,0]])
            aug_x = cv2.warpAffine(new_img,M,(cols,rows))

            # adjust the steering angle due to the shift
            # aug_y = steering_center -pixel_shift
            aug_y = steering_adjustment(steering_center,pixel_shift,cols)
            assert len(aug_x) != 0 and aug_x is not None, "augmented data fail, image"
            augname.append(Imgdir+"aug_" + filenames[i])
            cv2.imwrite(augname[i], aug_x)
            # elif not gen_aug:
            #     augname.append(Imgdir+"aug_" + filenames[i])
            
        assert type(aug_y) == float, "augmented data fail, angle"
        if loop == 3:
            # Add the augmented data information to the rest of the data
            aug_line = (augname[0],augname[1],augname[2],aug_y,sample[4],sample[5],sample[6])
        else:
            aug_line = (augname[0],"dummy","dummy",aug_y,sample[4],sample[5],sample[6])
        aug_samples.append(aug_line)
        progress_count += 1
    assert len(aug_samples) > 0, "returning no augmentations"
    with open(data_dir+'aug_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in aug_samples:
            writer.writerow(line)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return aug_samples
   
