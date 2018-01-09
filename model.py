import os
import os.path
import csv
import cv2
import math
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Input, Conv2D, Dropout, Activation, Flatten, Lambda, Dense
from keras.layers.convolutional import Cropping2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from util_functions import generate_processed_data, cvimg, equalize_dataset

# Load training dataset (driving simulation csv file)
def load_data(file_path, tune):
    rows=[]
    with open(file_path) as f:
        read=csv.reader(f)
        next(read)
            
        X_images=[]
        y_steering_measurements=[]
            
        for row in read:
            rows.append(row)
           
        for row in rows:
            if float(row[6])<0.5 and abs(float(row[3]))>.5:
                continue
            if abs(float(row[3]))<.015: 
                # Ignore rows where speed< 2 mph, steering angle is less than 3 degrees or more than ~50 degrees
                    continue
            else:
                # Everything denoted with _image here is still a path
                center_image=cvimg(row[0])
                left_image=cvimg(row[1])
                right_image=cvimg(row[2])

                center_steering_measurement=float(row[3])
                steering_correction=0.25
                X_images.append(center_image)
                y_steering_measurements.append(center_steering_measurement)

                X_images.append(left_image)
                y_steering_measurements.append(center_steering_measurement+steering_correction)

                X_images.append(right_image) 
                y_steering_measurements.append(center_steering_measurement-steering_correction)
    
    X_raw, y_raw=shuffle(X_images,y_steering_measurements)
    X_equalize, y_equalize=equalize_dataset(X_raw,y_raw)
    X_train, X_test, y_train, y_test=train_test_split(X_equalize,y_equalize,test_size=0.1)
    
    return X_train, X_test, y_train, y_test


# NVDIA End-to-End Driving CNN, with additional dropout layers
def NVIDIA_model(input_shape):
    model=Sequential()
    model.add(Cropping2D(cropping=((65,20),(0,0)), input_shape=input_shape, name='Image_crop'))
    # model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_shape, name='Normalization'))
    model.add(Lambda(lambda x: x/127.5-1.0, name='Normalization'))
    
    # Layer One- Convolution
    model.add(Conv2D(24,5,5, subsample=(2,2), border_mode="valid", W_regularizer=l2(0.001), name='Convolution_1'))
    model.add(ELU())
    
    # Layer Two- Colution
    model.add(Conv2D(36,5,5, subsample=(2,2), border_mode="valid", W_regularizer=l2(0.001), name='Convolution_2'))
    model.add(ELU())
    
    # Layer Three- Convolution
    model.add(Conv2D(48,5,5, subsample=(2,2), border_mode="valid", W_regularizer=l2(0.001), name='Convolution_3'))
    model.add(ELU())
    
    # Layer Four- Convolution
    model.add(Conv2D(64,3,3, border_mode="valid", W_regularizer=l2(0.001), name='Convolution_4'))
    model.add(ELU())
    
    # Layer Five- Convolution
    model.add(Conv2D(64,3,3, border_mode="valid", W_regularizer=l2(0.001), name='Convolution_5'))
    model.add(ELU())
    
    # Layer Six- Flatten
    model.add(Flatten())
    
    # Layer Seven- Full-Connected
    model.add(Dense(100, W_regularizer=l2(0.001), name='Dense_1'))
    model.add(ELU())
        
    # Layer Eight- Full-Connected
    model.add(Dense(50, W_regularizer=l2(0.001), name='Dense_2'))
    model.add(ELU())
    model.add(Dropout(0.50))
    
    # Layer Nine- Full-Connected
    model.add(Dense(10, W_regularizer=l2(0.001), name='Dense_3'))
    model.add(ELU())
    model.add(Dropout(0.50))
    
    # Layer Ten- Full-Connected
    model.add(Dense(1, name='Steering_Angle'))

    model.summary()

    return model

# Implementation
driving_data=[('Udacity','./data/udacity_driving_log.csv'),
                ('Self_T1_forward','./data/trackone_forward_driving_log.csv'),
                ('Self_T2_forward','./data/tracktwo_forward_driving_log.csv'),
                ('Self_T1_reverse','./data/trackone_reverse_driving_log.csv'),
                ('Self_T2_reverse','./data/tracktwo_reverse_driving_log.csv'),
                ('Self_T1_recovery','./data/trackone_recovery_driving_log.csv'),
                ('Self_T2_recovery','./data/tracktwo_recovery_driving_log.csv'),
                ('Self_T2_forward_lap2','./data/tracktwo_forward_driving_log.csv'), 
                ('Self_T1_reverse_lap2','./data/trackone_reverse_driving_log.csv'),
                ('Self_T1_forward_lap2','./data/trackone_forward_driving_log.csv'),
                ('Udacity_lap2','./data/udacity_driving_log.csv')]

valid_driving_data=[True, True, False, True, False, True, False, False, False, False, False] # To use or not to use?
batch_size=[64,64,64,64,64,64,64,64,64,64,64]

epochs=[10,5,20,5,10,6,3,5,6,6,6]

#valid_driving_data=[True, False, True, False, True, False, False, True, False, False, False] # To use or not to use?
#epochs=[10,5,20,3,10,3,3,5,6,6,6]

print("Model initialized.")
input_shape=(160,320,3)
model=NVIDIA_model(input_shape)
opt=Adam(lr=1e-4)
model.compile(optimizer=opt, loss='mean_squared_error')

if os.path.exists('./model_weights_01.h5'):
    model.load_weights('./model_weights_01.h5')
    
for i in range(len(driving_data)):
    
    if valid_driving_data[i]==True:
        print("Using {} dataset".format(driving_data[i][0]))

        X_train, X_test, y_train, y_test=load_data(driving_data[i][1], tune[i])
        
        generate_train=generate_processed_data(X_train,y_train,
            input_shape[0],input_shape[1],input_shape[2],batch_size[i],train_flag=True)
    
        generate_test=generate_processed_data(X_test,y_test,
            input_shape[0],input_shape[1],input_shape[2],batch_size[i],train_flag=False)
    
        early_stopping=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='min')

        history=model.fit_generator(generator=generate_train, samples_per_epoch=360*batch_size[i], nb_epoch=epochs[i], verbose=1, validation_data=generate_test, nb_val_samples=40*batch_size[i])
        
    else:
        print("Ignoring {} dataset".format(driving_data[i][0]))
    model.save_weights('./model_weights_01.h5')
print("\n")
print("Loss {}".format(history.history['loss']))
print("Validation Loss {}".format(history.history['val_loss']))
print("Saving NVIDIA model.")

model.save('./model_01.h5')

print('Model Trained and saved. Script terminated.')
