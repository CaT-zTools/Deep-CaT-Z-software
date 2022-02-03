# -*- coding: utf-8 -*-
"""
@author: DeepCaT_Z
"""

#%% ############################################
######### PARAMETERS: CHANGE IF NECESSARY ######
################################################

NCLASSES = 4

DATADIR = 'dataset_train100_val50_test50_128_example'

STEPS_LIST = list(range(-10, 1, 1))

BATCHSIZE = 16
EPOCHS = 100

lr = 1e-4

IMG_SIZE = 128
IMG_LARGE_SIZE = int(IMG_SIZE * 1.10)  # ~110%

CHANNELS_SIZE = 1
