"""
@author: DeepCaT_Z
"""

#%% ############################################
######### IMPORTS: DO NOT TOUCH ##############
################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
import classificationModel_functions
from config_Classification_DeepCaT_Z import *


#%% ############################################
######### MAIN CODE: DO NOT TOUCH ##############
################################################

if __name__ == '__main__':
    
    model_name = f'model_{DATADIR}_{NCLASSES}_{IMG_SIZE}'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    N_STEPS = len(STEPS_LIST)
    model = getattr(classificationModel_functions, f'Model_X_A')(N_STEPS)
    model = model.to(device)
    
    #%% Load dataset
    # Train
    tr_ds = classificationModel_functions.MiceDataset(DATADIR, 'train', STEPS_LIST, True, True)
    tr = DataLoader(tr_ds, BATCHSIZE, True, num_workers=2, prefetch_factor=10)
        
    # Validation
    ts_ds = classificationModel_functions.MiceDataset(DATADIR, 'val', STEPS_LIST, False, True)
    ts = DataLoader(ts_ds, BATCHSIZE, num_workers=2, prefetch_factor=10)
        
    # Train model
    all_losses = {'A': classificationModel_functions.ce}
    all_losses_weights = {'A': 1}
    all_metrics = {'A': [classificationModel_functions.acc, classificationModel_functions.bacc, classificationModel_functions.ce]}
    
    history = classificationModel_functions.train(model, device, EPOCHS, tr, ts, all_metrics, all_losses, all_losses_weights, lr)
    
    torch.save(model.state_dict(), model_name + '.pth')


