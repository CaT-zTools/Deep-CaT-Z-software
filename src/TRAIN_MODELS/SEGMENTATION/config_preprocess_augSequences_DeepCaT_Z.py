"""
@author: DeepCaT_Z
"""

#%% ############################################
######### IMPORTS: DO NOT TOUCH ##############
################################################

import numpy as np

#%% ############################################
######### PARAMETERS: CHANGE IF NECESSARY ######
################################################

np.random.seed(123)

IMAGE_SIZE = 128 # final pixel resolution
TOTAL_TRAINING_SIZE_final = 200 # number of images needed in the final training set

STEP = 1
STEPS_LIST = list(range(-10, 1, STEP)) # Format: (ID of the last frame to be included, id of the current frame, step between frames)
N_FRAMES_SEQUENCE = len(STEPS_LIST) # number of frames per sequence - it should be

# Directory of original training data
directory_original_data = 'dataset_train100_val50_test50_128_example/'
