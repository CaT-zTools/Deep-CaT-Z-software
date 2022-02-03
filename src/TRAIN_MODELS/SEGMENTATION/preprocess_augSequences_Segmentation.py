"""
@author: DeepCaT_Z
"""

#%% PRE-PROCESSING (optional):
    
# Augmentation of train-validation-test splits:

# training augmentation operations:
# - free rotate (30 degrees maximum)
# - random crop of 75%
# - change brightness of ±25%

# validation and testing: just sizing and saving

#%% ############################################
######### IMPORTS: DO NOT TOUCH ##############
################################################

from skimage.io import imread, imsave
from skimage.transform import rotate, resize
from scipy import ndimage 
import numpy as np
from tqdm import tqdm
import os, shutil
from config_preprocess_augSequences_DeepCaT_Z import *

#%% ############################################
######### MAIN CODE: DO NOT TOUCH ##############
################################################

# Create folders to save augmented data:

name_main_folder = 'datasetAugmented_' + str(N_FRAMES_SEQUENCE) + 'framesPerSequence_' + 'step' + str(STEP)

shutil.rmtree(name_main_folder, ignore_errors=True)
os.mkdir(name_main_folder)

# subfolders
shutil.rmtree(name_main_folder + '/train_aug', ignore_errors=True)
os.mkdir(name_main_folder + '/train_aug')

# subsubfolder
shutil.rmtree(name_main_folder + '/train_aug/' + 'frames', ignore_errors=True)
os.mkdir(name_main_folder + '/train_aug/' + 'frames')
# subsubfolder
shutil.rmtree(name_main_folder + '/train_aug/' + 'masks', ignore_errors=True)
os.mkdir(name_main_folder + '/train_aug/' + 'masks')

shutil.rmtree(name_main_folder + '/val_aug', ignore_errors=True)
os.mkdir(name_main_folder + '/val_aug')

# subsubfolder
shutil.rmtree(name_main_folder + '/val_aug/' + 'frames', ignore_errors=True)
os.mkdir(name_main_folder + '/val_aug/' + 'frames')
# subsubfolder
shutil.rmtree(name_main_folder + '/val_aug/' + 'masks', ignore_errors=True)
os.mkdir(name_main_folder + '/val_aug/' + 'masks')

shutil.rmtree(name_main_folder + '/test_aug', ignore_errors=True)
os.mkdir(name_main_folder + '/test_aug')

# subsubfolder
shutil.rmtree(name_main_folder + '/test_aug/' + 'frames', ignore_errors=True)
os.mkdir(name_main_folder + '/test_aug/' + 'frames')
# subsubfolder
shutil.rmtree(name_main_folder + '/test_aug/' + 'masks', ignore_errors=True)
os.mkdir(name_main_folder + '/test_aug/' + 'masks')

#%% Load images - train, validation and test sets

def load_images(size, typeData, steps):
    
    files = sorted(os.listdir(f'{directory_original_data}{typeData}frames'))
  
    # structure files in contiguous video sequences
    videos = []
    for fname in files:
                
        if len(videos) != 0 and int(fname.split('_')[1]) == int(videos[-1][-1].split('_')[1])+2:
            videos[-1].append(fname[:-4])
        else:
            videos.append([fname[:-4]])
        
    videos = [[video[i+step] for step in steps] for video in videos for i in range(np.max(np.abs(steps)), len(video))]
    
    
    return videos

typeData_tr = 'train/'
videos_train = load_images(IMAGE_SIZE, typeData_tr, STEPS_LIST)

typeData_val = 'val/'
videos_val = load_images(IMAGE_SIZE, typeData_val, STEPS_LIST)

typeData_te = 'test/'
videos_test = load_images(IMAGE_SIZE, typeData_te, STEPS_LIST)


#%% TRAINING SET:
# Apply N_AUGMENT augmentation operations to N_FRAMES_SEQUENCE frames (the same operation for the N_FRAMES_SEQUENCE frames each time)

N_AUGMENT = int(TOTAL_TRAINING_SIZE_final / len(videos_train))  # how many times to augment each sequence of N_FRAMES_SEQUENCE

it_begin = range(TOTAL_TRAINING_SIZE_final)

counter_train = 0
counter_to_save = 0

# tqdm: progress bar
for it in tqdm(range(len(videos_train))): 
    
    for N_times in range(N_AUGMENT):
        
        imgs = [resize(imread(f'{directory_original_data}train/frames/{seq_name}.png', True), (IMAGE_SIZE, IMAGE_SIZE)) for seq_name in videos_train[counter_train]] 
        labels = [resize(imread(f'{directory_original_data}train/masks/{seq_name}_seg.png', True), (IMAGE_SIZE, IMAGE_SIZE)) for seq_name in videos_train[counter_train]]
        
        # free rotate
        angle = np.random.rand() * 30
        imgs = [rotate(img, angle) for img in imgs]
        labels = [rotate(label, angle) for label in labels]
    
        # random flip - horizontal + vertical
        # flip the 100 frames
        flip_bool_horiz = np.random.randint(2, size=1)
    
        if flip_bool_horiz:
            imgs = np.flip(imgs, 2)
            labels = np.flip(labels, 2)
    
        flip_bool_vert = np.random.randint(2, size=1)
    
        if flip_bool_vert:
            imgs = np.flip(imgs, 1)
            labels = np.flip(labels, 1)
    
        # Shift
        shift_bool = np.random.randint(2, size=1)  
        if shift_bool:
            shift_pix = np.random.rand() * 12
            imgs = [ndimage.shift(img, shift_pix) for img in imgs]
            labels = [ndimage.shift(label, shift_pix) for label in labels]
    
        
        # change brightness by 25% (only images, not labels, of course)
        brightness_factor = 1 - (np.random.rand()*0.5-0.25)  # 0.75-1.25
        imgs = [np.clip(img*brightness_factor, 0, 1) for img in imgs]
         
        # Save images
        for j, (img, label) in enumerate(zip(imgs, labels)):
            imsave(name_main_folder + '/train_aug/frames/%05d_%05d.png' % (counter_to_save, j), (img*255).astype(np.uint8))
            imsave(name_main_folder + '/train_aug/masks/%05d_%05d_seg.png' % (counter_to_save, j), ((label >= 0.5)*255).astype(np.uint8), check_contrast=False)
            
    
        counter_to_save = counter_to_save  + 1
    
    # At the end: when sequence was replicated N_AUGMENT times, change to another sequence
    counter_train = counter_train + 1

        
#%% VALIDATION SET
# Just save, without augmentation
counter_to_save = 0

for it in tqdm(range(len(videos_val))): 
    
    imgs = [resize(imread(f'{directory_original_data}val/frames/{seq_name}.png', True), (IMAGE_SIZE, IMAGE_SIZE)) for seq_name in videos_val[it]] 
    labels = [resize(imread(f'{directory_original_data}val/masks/{seq_name}_seg.png', True), (IMAGE_SIZE, IMAGE_SIZE)) for seq_name in videos_val[it]]
    
    # save images
    for j, (img, label) in enumerate(zip(imgs, labels)):
        imsave(name_main_folder + '/val_aug/frames/%05d_%05d.png' % (counter_to_save, j), (img*255).astype(np.uint8))
        imsave(name_main_folder + '/val_aug/masks/%05d_%05d_seg.png' % (counter_to_save, j), ((label >= 0.5)*255).astype(np.uint8), check_contrast=False)
        
    counter_to_save = counter_to_save  + 1
    
#%% TESTING SET
# Just save, without augmentation
counter_to_save = 0

for it in tqdm(range(len(videos_test))): 
    
    imgs = [resize(imread(f'{directory_original_data}test/frames/{seq_name}.png', True), (IMAGE_SIZE, IMAGE_SIZE)) for seq_name in videos_test[it]] 
    labels = [resize(imread(f'{directory_original_data}test/masks/{seq_name}_seg.png', True), (IMAGE_SIZE, IMAGE_SIZE)) for seq_name in videos_test[it]]
  
    # save images
    for j, (img, label) in enumerate(zip(imgs, labels)):
        imsave(name_main_folder + '/test_aug/frames/%05d_%05d.png' % (counter_to_save, j), (img*255).astype(np.uint8))
        imsave(name_main_folder + '/test_aug/masks/%05d_%05d_seg.png' % (counter_to_save, j), ((label >= 0.5)*255).astype(np.uint8), check_contrast=False)
        
    counter_to_save = counter_to_save  + 1
    



