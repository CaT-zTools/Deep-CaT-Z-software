"""
@author: DeepCaT_Z
"""

#%% ######################################################################
######### MODELS' ARCHITECTURES AND PARAMETERS DO NOT TOUCH ##############
##########################################################################

#%% Imports
import tensorflow as tf
from tensorflow.keras import layers
from skimage.io import imread
import numpy as np
import os
from keras import backend as K
from keras.losses import binary_crossentropy
tf.get_logger().setLevel('INFO')

#%% Segmentation Model - UNet backbone architecture
# UNet
# UNet + ConvLSTM v1
# UNet + ConvLSTM v2
# UNet + ConvLSTM v3

def UNet_ConvLSTM(architecture = 'unet', image_size = 128, n_convs = 4, filters = 32, activation_func = 'relu'):

    x = input_layer = layers.Input((None, image_size, image_size, 1))
    
    # ENCODER
    encoder = []
    for f in range(n_convs):
        
        if f > 2:
            x = layers.TimeDistributed(layers.Conv2D(filters, 3, 2, 'same', activation = activation_func))(x)
            x = layers.TimeDistributed(layers.Dropout(0.5))(x)
            encoder.append(x)
        else:
            x = layers.TimeDistributed(layers.Conv2D(filters, 3, 2, 'same', activation = activation_func))(x)
            encoder.append(x)
    
    if architecture in ('convlstm-v1', 'convlstm-v3'):
        x = layers.ConvLSTM2D(filters, 3, padding='same', return_sequences=True)(x)
    
    # DECODER
    for i in range(n_convs):
        x = layers.Concatenate()([x, encoder[-i-1]])
        x = layers.TimeDistributed(layers.Conv2DTranspose(filters, 3, 2, 'same', activation = activation_func))(x)
    
    if architecture in ('convlstm-v2', 'convlstm-v3'):
        x = layers.ConvLSTM2D(filters, 3, padding='same', return_sequences=True)(x)
    
    x = layers.TimeDistributed(layers.Conv2DTranspose(1, 1, activation='sigmoid'))(x)
    model = tf.keras.Model(input_layer, x)

    return model

#%% UNet - smaller architecture

def UNet_simple(image_size = 128, n_convs = 4, filters = 32, activation_func = 'relu'):

    x = input_layer = layers.Input((image_size, image_size, 1))
    
    # ENCODER
    encoder = []
    for f in range(n_convs):
        
        if f > 2:
            x = (layers.Conv2D(filters, 3, 2, 'same', activation = activation_func))(x)
            x = (layers.Dropout(0.5))(x)
            encoder.append(x)
        else:
            x = (layers.Conv2D(filters, 3, 2, 'same', activation = activation_func))(x)
            encoder.append(x)   
    
    # DECODER
    for i in range(n_convs):
        x = layers.Concatenate()([x, encoder[-i-1]])
        x = (layers.Conv2DTranspose(filters, 3, 2, 'same', activation = activation_func))(x)
        
    x = layers.Conv2DTranspose(1, 1, activation='sigmoid')(x)
    model = tf.keras.Model(input_layer, x)

    return model

#%% UTILS/FUNCTIONS

def acc0(Y, Yhat):
    Yhat = tf.math.round(Yhat)
    Y0 = tf.cast(Y == 0, tf.float32)
    Yhat0 = tf.cast(Yhat == 0, tf.float32)
    return tf.reduce_sum(Y0 * Yhat0) / tf.reduce_sum(Y0)

def acc1(Y, Yhat):
    Yhat = tf.math.round(Yhat)
    return tf.reduce_sum(Y*Yhat) / tf.reduce_sum(Y)

def balanced_ce(Y, Yhat):
    alpha = 0.995
    return -tf.reduce_mean(alpha*Y*tf.math.log(Yhat+1e-6) + (1-alpha)*(1-Y)*tf.math.log(1-Yhat+1e-6))

def focal_ce(Y, Yhat):
    gamma = 2
    return -tf.reduce_mean(((1-Yhat)**gamma)*Y*tf.math.log(Yhat+1e-6) + (Yhat**gamma)*(1-Y)*tf.math.log(1-Yhat+1e-6))

def loss(Y, Yhat):
    return balanced_ce(Y, Yhat)#/2 + dice_loss(Y, Yhat)/2

def jaccard_coef(Y, Yhat):
    # __author__ = Vladimir Iglovikov
    smooth = 1.
    y_pred_bin=tf.dtypes.cast(Yhat>0.5, tf.float32)
    y_true_f = K.flatten(Y)
    y_pred_f = K.flatten(y_pred_bin)
    intersection = K.sum(y_true_f * y_pred_f)
    sum_ = K.sum(y_true_f + y_pred_f)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def dice_coef(Y, Yhat):
    smooth = 1.
    y_pred_bin=tf.dtypes.cast(Yhat>0.5, tf.float32)
    y_true_f = K.flatten(Y)
    y_pred_f = K.flatten(y_pred_bin)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(Y, Yhat, smooth=1):
    numerator = 2 * tf.reduce_sum(Y * Yhat) + smooth
    denominator = tf.reduce_sum(Y + Yhat) + smooth
    return 1 - numerator / denominator

def bce_dice_loss(Y, Yhat):
    return 0.5 * binary_crossentropy(Y, Yhat) + 0.5*(dice_loss(Y, Yhat))