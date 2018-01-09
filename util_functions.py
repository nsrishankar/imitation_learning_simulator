# Utility functions to go with model.py
import os
import csv
import cv2
import math
import datetime
import keras
import numpy as np


# Loading images using OpenCV using local paths
def cvimg(global_image_path):
    local_preposition='./data/IMG/'
    image_filename=global_image_path.split('/')
    return (local_preposition+str(image_filename[1]))

# Equalizes classes across dataset
def equalize_dataset(X_data,y_data):
    number_of_bins=25
    average_bin_length=len(y_data)/number_of_bins
    histogram,bin_edges=np.histogram(y_data,number_of_bins)

    remove_data_index=[]
    keep_data=[]
    keep_all_data=1.0
    target_samples=0.75*average_bin_length

    for i in range(number_of_bins):
        if histogram[i]<average_bin_length:
            keep_data.append(keep_all_data)
        else:
            keep_samples=target_samples/histogram[i]
            keep_data.append(keep_samples)

    for i in range(len(y_data)):
        for j in range(len(bin_edges)):
            if y_data[i]>bin_edges[j] and y_data[i]<=bin_edges[j+1]:
                scale_probability=np.random.uniform()
                if scale_probability>=keep_data[j]:
                    remove_data_index.append(i)
    X_pruned=np.delete(X_data,remove_data_index, axis=0)
    y_pruned=np.delete(y_data,remove_data_index, axis=0)
    return X_pruned,y_pruned

# Converts from RGB space to YUV space (needed by NVIDIA model)
def bgr2yuv(raw_image):
    return cv2.cvtColor(raw_image,cv2.COLOR_BGR2YUV)

# Adds a x- and y- direction translation to a given image
def random_translate(raw_image,trans_range_x,trans_range_y,steering_angle):
    rows,cols,ch=raw_image.shape
    x_translation=trans_range_x*np.random.uniform()-0.5*trans_range_x
    y_translation=trans_range_y*np.random.uniform()-0.5*trans_range_y
    steering_angle+=x_translation*2e-3
    Trans_M=np.float32([[1,0,x_translation],
                        [0,1,y_translation]])

    image=cv2.warpAffine(raw_image,Trans_M,(cols,rows))
    return image, steering_angle

# Vertically flips a given image, and associates the augmented image with the negative of the steering angle
def random_flips(raw_image,trans_range_x,trans_range_y,steering_angle):
    threshold=np.random.rand()

    #if threshold<=0.50:
    image=cv2.flip(raw_image,1) # Vertical Flip
    steering_angle*=-1.0
    return image, steering_angle

def flip(raw_image,steering_angle):
    image=cv2.flip(raw_image,1) # Definite Vertical Flip
    steering_angle*=-1.0
    return image, steering_angle

# Modifies brightness of a given image by a random factor to mimic changes in brightness/patches in the road surface
def random_contrast(raw_image,trans_range_x,trans_range_y,steering_angle):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    random_brightness=1.+0.20*(np.random.uniform()-0.5)
    image[:,:,2]=image[:,:,2]*random_brightness
    image=cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
    return image, steering_angle

# Obtains a given image, steering angle pair, and adds a random gaussian noise to the steering angle
def random_noise(raw_image,trans_range_x,trans_range_y,steering_angle):
    mu,sigma=0,0.1
    noise=np.random.normal(mu,sigma)
    steering_angle+=noise
    return raw_image,steering_angle

# Creates an augmented image using various 
def augment(raw_image,trans_range_x,trans_range_y,steering_angle):
    function_list=['random_translate','random_flips','random_contrast','random_noise']
    n_augments=np.random.choice(1,len(function_list)+1)
    function_choice=np.random.choice(function_list,n_augments,replace=False)
    
    for function in function_choice:
        raw_image,steering_angle=eval(function)(raw_image,trans_range_x,trans_range_y,steering_angle)
        # "Raw_image" in this instance, is the output from a function evaluation, which is fed into the next function (if any
    return (raw_image, steering_angle)

# Generator to create batches of processed data
def generate_processed_data(X,y,im_h,im_w,im_ch,batch_size,train_flag):
# Gets local image paths (not images!), steering angles for preprocessing and batch size.

    features=np.empty([batch_size,im_h,im_w,im_ch])
    labels=np.empty([batch_size])
    augment_threshold=np.random.rand()

    while 1:
        count=0
        index=np.random.permutation(len(X))
        
        #X_temp=np.empty([len(X),im_h,im_w,im_ch])
        #y_temp=np.empty([len(X)])
        
        #for i in range(len(X)):
        #   temp=cv2.imread(X[i])
        #    temp_steering=y[i]
            
        #    if y[i]>0.3:
        #        X_temp[i], y_temp[i]=flip(temp,temp_steering)
                
        #    else:
        #        X_temp[i],y_temp[i]=temp,temp_steering
                
       
        for i in range(len(X)):
            temp_image=cv2.imread(X[index[i]]) # Until this point we have been dealing with a path, now temp_image is an actual image
            temp_steering_angle=y[index[i]]
            
            augment_count=0
            if train_flag==True and augment_threshold<=.75:
                augment_count+=1
                image, steering_angle=augment(raw_image=temp_image,
                    steering_angle=temp_steering_angle,
                    trans_range_x=10,
                    trans_range_y=10)
            else: # Only augment if the image is in the training set, and only random images
                image, steering_angle=temp_image, temp_steering_angle
            features[i]=bgr2yuv(np.array(image, dtype=np.uint8)) # Features--> Image in YUV space
            labels[i]=steering_angle # Labels--> Steering angles

            count+=1
            
            if count==batch_size:
                break
        yield features, labels