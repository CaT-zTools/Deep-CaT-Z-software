# -*- coding: utf-8 -*-
"""
@author: DeepCaT_Z
"""

#%% ############################################
######### IMPORTS: DO NOT TOUCH ##############
################################################

import numpy as np
import cv2
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import os
import time
from matplotlib import pyplot as plt
from keras.losses import binary_crossentropy
from keras import backend as K

from config_Segmentation_DeepCaT_Z import *

# Activate GPU - if needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


#%% ############################################
######### MAIN CODE: DO NOT TOUCH ##############
################################################

# number of batches for each set:
n_batch_train = int(N_TRAIN_AUG / TIME_SEQUENCE_FRAMES)
n_batch_val = int(N_VAL_AUG / TIME_SEQUENCE_FRAMES)
n_batch_test = int(N_TEST_AUG / TIME_SEQUENCE_FRAMES)

# Architectures
from segmentationModel_functions import UNet_simple, UNet_ConvLSTM

if ARCHITECTURE == 'unet':
    model = UNet_simple(IMG_SIZE)
    print(model.summary())
    
else:
    model = UNet_ConvLSTM(ARCHITECTURE, IMG_SIZE)
    print(model.summary())

#%% Load images -  1 channel grayscale
from skimage.io import imread

def load_images(dirname, n_batch, frames_time_sequence, architecture):
    
    if architecture == 'unet': 
    
        X = np.array([imread('%s/frames/%05d_%05d.png' % (dirname, i, j), True).astype(np.float32)/255
            for j in range(frames_time_sequence) for i in range(n_batch)])[..., np.newaxis]
        
        Y = np.array([imread('%s/masks/%05d_%05d_seg.png' % (dirname, i, j), True).astype(np.float32)/255
            for j in range(frames_time_sequence) for i in range(n_batch)])[..., np.newaxis]
    else:
         
        X = np.array([[imread('%s/frames/%05d_%05d.png' % (dirname, i, j), True).astype(np.float32)/255
            for j in range(frames_time_sequence)] for i in range(n_batch)])[..., np.newaxis]
        
        Y = np.array([[imread('%s/masks/%05d_%05d_seg.png' % (dirname, i, j), True).astype(np.float32)/255
            for j in range(frames_time_sequence)] for i in range(n_batch)])[..., np.newaxis]
         
    return X, Y

# Load training set - already augmented and with correct image size
X_train, Y_train = load_images(TRAIN_DIR, n_batch_train, TIME_SEQUENCE_FRAMES, ARCHITECTURE)

# Load validation set - already augmented and with correct image size
X_val, Y_val = load_images(VAL_DIR, n_batch_val, TIME_SEQUENCE_FRAMES, ARCHITECTURE)

#%% Losses - definition
from segmentationModel_functions import acc0, acc1, balanced_ce, loss, jaccard_coef, dice_coef, dice_loss, bce_dice_loss

#%% Model compilation & parameters
from tensorflow.keras.optimizers import Adam

model.compile(optimizer = Adam(lr = LR), loss = bce_dice_loss, metrics = [acc0, acc1, dice_coef, jaccard_coef])

#%% Training parameters + Fit model

training_history = model.fit(X_train, Y_train, BATCH_SIZE, EPOCHS, 1, validation_data=(X_val, Y_val), callbacks=[model_checkpoint])



