# -*- coding: utf-8 -*-
"""
@author: DeepCaT_Z
"""

#%% ############################################
######### IMPORTS: DO NOT TOUCH ##############
################################################

from keras.callbacks import ModelCheckpoint


#%% ############################################
######### PARAMETERS: CHANGE IF NECESSARY ######
################################################

IMG_SIZE = 128 #256

EPOCHS = 100
BATCH_SIZE = 1

# Architecture parameters
ARCHITECTURE = 'unet' #or 'convlstm-v3'

STEP = 1
STEPS_LIST = list(range(-10, 1, STEP)) # Format: (ID of the last frame to be included, id of the current frame, step between frames)
TIME_SEQUENCE_FRAMES = len(STEPS_LIST) # number of frames per sequence - it should be

# Number of files in augmented folders
N_TRAIN_AUG = 1980
N_VAL_AUG = 440
N_TEST_AUG = 440

# Training parameters
LR = 1e-4

# Model name to save to file
model_to_save = 'model_' + ARCHITECTURE + '_bcediceloss_batch_' + str(BATCH_SIZE) + '_' + str(EPOCHS) + 'epochs'

model_checkpoint = ModelCheckpoint(model_to_save +'_bestModel.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

# Folders with train, val and test data - augmented folder
TRAIN_DIR = 'datasetAugmented_' + str(TIME_SEQUENCE_FRAMES) + 'framesPerSequence_'  + 'step' + str(STEP) + '/train_aug/' 
VAL_DIR = 'datasetAugmented_' + str(TIME_SEQUENCE_FRAMES) + 'framesPerSequence_'  + 'step' + str(STEP) + '/val_aug/' 
TEST_DIR = 'datasetAugmented_' + str(TIME_SEQUENCE_FRAMES) + 'framesPerSequence_'  + 'step' + str(STEP) + '/test_aug/' 

