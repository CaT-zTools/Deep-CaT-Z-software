"""
@author: DeepCaT_Z
"""

#%% PRE-PROCESSING (mandatory):
    
# Resizing all frames to pre-defined pixel's resolution
# OBS: augmentation operations will be carried out while training the model.

#%% ############################################
######### IMPORTS: DO NOT TOUCH ##############
################################################

import os, shutil
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize
from config_preprocess_Classification_DeepCaT_Z import *

#%% ############################################
######### MAIN CODE: DO NOT TOUCH ##############
################################################

new_datadir = f'{directory_original_data}_{IMAGE_SIZE}'
if os.path.exists(new_datadir):
    shutil.rmtree(new_datadir)
os.mkdir(new_datadir)

for root, dirs, files in os.walk(directory_original_data):
    new_root = os.path.join(new_datadir, '/'.join(root.split('\\')[1:]))
    for dirname in dirs:
        os.mkdir(os.path.join(new_root, dirname))
    if len(files) == 0: continue
    print(root)
    for f in tqdm(files):
        if f.endswith('.png'):
            x = imread(os.path.join(root, f), CHANNELS)
            if root.endswith('//masks'):
                x = x >= 128
                
                x = resize(x, (IMAGE_SIZE, IMAGE_SIZE), 0)
                x = (x * 255).astype(np.uint8)
            else:
                x = resize(imread(os.path.join(root, f), CHANNELS), (IMAGE_SIZE, IMAGE_SIZE), 5)
                x = (x * 255).astype(np.uint8)
            imsave(os.path.join(new_root, f), x, check_contrast=False)
        else:
            shutil.copyfile(os.path.join(root, f), os.path.join(new_root, f))

#%% Resize with 110% IMAGE_SIZE (augmentation purposes)
IMAGE_SIZE = int(IMAGE_SIZE*1.10)

new_datadir = f'{directory_original_data}_{IMAGE_SIZE}'
if os.path.exists(new_datadir):
    shutil.rmtree(new_datadir)
os.mkdir(new_datadir)

for root, dirs, files in os.walk(directory_original_data):
    new_root = os.path.join(new_datadir, '/'.join(root.split('\\')[1:]))
    for dirname in dirs:
        os.mkdir(os.path.join(new_root, dirname))
    if len(files) == 0: continue
    print(root)
    for f in tqdm(files):
        if f.endswith('.png'):
            x = imread(os.path.join(root, f), CHANNELS)
            if root.endswith('//masks'):
                x = x >= 128
                
                x = resize(x, (IMAGE_SIZE, IMAGE_SIZE), 0)
                x = (x * 255).astype(np.uint8)
            else:
                x = resize(imread(os.path.join(root, f), CHANNELS), (IMAGE_SIZE, IMAGE_SIZE), 5)
                x = (x * 255).astype(np.uint8)
            imsave(os.path.join(new_root, f), x, check_contrast=False)
        else:
            shutil.copyfile(os.path.join(root, f), os.path.join(new_root, f))
