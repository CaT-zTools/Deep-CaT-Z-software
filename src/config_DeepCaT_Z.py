# -*- coding: utf-8 -*-
"""
@author: DeepCaT_Z
"""

import  numpy as np

################### CAMERA ##################################################
# Parameters
# pixel resolution
ORIGINAL_SIZE_Y = 480 #424
ORIGINAL_SIZE_X = 640 #512

IMG_SIZE = 128 #256

CHANNELS_SIZE = 1

################### ARDUINO ##################################################
TIMEOUT_ARDUINO = 0.01
BAUDRATE = 1000000

################## DEEP MODEL PARAMETERS - AUTOMATIC CLASSIFICATION ###########
GPU_AVAILABLE = False

# Input frames' sequence - (past id frame, id current frame, step) 
# steps_list = list(range(-10, 1, 1))
STEPS_LIST = list(range(-20, 1, 1))

NCLASSES = 4

################## DEEP MODEL PARAMETERS - AUTOMATIC SEGMENTATION ############
# For erosion - pre processing
KERNEL_EROSION = np.ones((2,2), np.uint8)

################## BACKGROUND SUBTRACTION ###################################
NUM_FRAMES_TO_CREATE_MODEL = 200

#Defaults values: minimum and maximum rat range for background cleaning
MINRATRANGE = 5
MAXRATRANGE = 300
















