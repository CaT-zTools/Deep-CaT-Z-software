# -*- coding: utf-8 -*-
"""
@author: DeepCaT_Z
"""
#%% ############################################
######### IMPORTS: DO NOT TOUCH ##############
################################################

import torch
from torch import nn
from sklearn.metrics import f1_score
from torch import optim
from time import time
import numpy as np
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.morphology import erosion, disk
from torch.utils.data import Dataset
from torchvision import transforms
from config_Classification_DeepCaT_Z import *
import os
  
#%% ############################### UTILS FOR DATASET CONSTRUCTION ############

# Data augmentation operations
class BrightnessTransform:
    def __call__(self, sample):
        brightness = np.random.rand()*0.3-0.15
        if 'X' in sample:
            sample['X'] = np.clip(sample['X'] + brightness, 0, 1)
        if 'F' in sample:
            sample['F'][0] = np.clip(sample['F'][0] + brightness, 0, 1)
        return sample

class FlipTransform:
    def __call__(self, sample):
        if np.random.rand() > 0.5:
            if 'X' in sample:
                sample['X'] = np.flip(sample['X'], 2)
            if 'S' in sample:
                sample['S'] = np.flip(sample['S'], 1)
        return sample

class Rot90Transform:
    def __call__(self, sample):
        k = np.random.randint(0, 4)
        if 'X' in sample:
            sample['X'] = np.rot90(sample['X'], k, (1, 2))
        if 'S' in sample:
            sample['S'] = np.rot90(sample['S'], k, (0, 1))
        return sample

class CropTransform:
    def __call__(self, sample):
        dx = np.random.randint(0, IMG_LARGE_SIZE-IMG_SIZE)
        dy = np.random.randint(0, IMG_LARGE_SIZE-IMG_SIZE)
        if 'X' in sample:
            sample['X'] = sample['X'][:, dy:dy+IMG_SIZE, dx:dx+IMG_SIZE]
        if 'S' in sample:
            sample['S'] = sample['S'][dy:dy+IMG_SIZE, dx:dx+IMG_SIZE]
        return sample

class ShiftTransform:
    def __call__(self, sample):
        dx = np.random.randint(0, IMG_LARGE_SIZE-IMG_SIZE)
        dy = np.random.randint(0, IMG_LARGE_SIZE-IMG_SIZE)
        if dx and dy:
            xdir = np.random.randint(0, 2)
            ydir = np.random.randint(0, 2)
            if 'X' in sample:
                X = np.zeros_like(sample['X'])
                if xdir == 0 and ydir == 0:
                    X[:, dy:, dx:] = sample['X'][:, :-dy, :-dx]
                elif xdir == 1 and ydir == 0:
                    X[:, dy:, :-dx] = sample['X'][:, :-dy, dx:]
                elif xdir == 0 and ydir == 1:
                    X[:, :-dy, dx:] = sample['X'][:, dy:, :-dx]
                else:
                    X[:, :-dy, :-dx] = sample['X'][:, dy:, dx:]
                sample['X'] = X
            if 'S' in sample:
                S = np.zeros_like(sample['S'])
                if xdir == 0 and ydir == 0:
                    S[dy:, dx:] = sample['S'][:, :-dy, :-dx]
                elif xdir == 1 and ydir == 0:
                    S[dy:, :-dx] = sample['S'][:, :-dy, dx:]
                elif xdir == 0 and ydir == 1:
                    S[:-dy, dx:] = sample['S'][:, dy:, :-dx]
                else:
                    S[:-dy, :-dx] = sample['S'][:, dy:, dx:]
                sample['S'] = S
        return sample

# Functions for loading the dataset + apply augmentation operations while training
class MiceDataset(Dataset):
    def __init__(self, dirname, fold, steps, with_augment, with_sampling):
        self.dirname = dirname
        assert os.path.exists(f'{self.dirname}_{IMG_SIZE}'), f'{self.dirname}_{IMG_SIZE} does not exist'
        assert os.path.exists(f'{self.dirname}_{IMG_LARGE_SIZE}'), f'{self.dirname}_{IMG_LARGE_SIZE} does not exist'
        self.fold = fold
        self.transform_big = None
        self.transform_small = None
        
        self.steps = steps
        
        if with_augment:
            self.transform_big = transforms.Compose([
                CropTransform(),
                BrightnessTransform(),
                Rot90Transform(),
                FlipTransform(),
            ])
            self.transform_small = transforms.Compose([
                BrightnessTransform(),
                Rot90Transform(),
                FlipTransform(),
            ])
        files = sorted(os.listdir(f'{dirname}_{IMG_SIZE}/{fold}/frames'))
        
        # structure files in contiguous video sequences
        videos = []
        for fname in files:
            
            if len(videos) != 0 and int(fname.split('_')[1]) == int(videos[-1][-1].split('_')[1])+2:
                videos[-1].append(fname[:-4])
            else:
                videos.append([fname[:-4]])
        
        videos = [[video[i+step] for step in steps] for video in videos for i in range(np.max(np.abs(self.steps)), len(video))]
                
        # oversample based on activities
        if with_sampling:
            A = np.array([np.loadtxt(f'{dirname}_{IMG_SIZE}/{fold}/labels/{video[-1]}.txt', np.int32) for video in videos]) - 1
            # repeat all classes until they equalize the majority class
            reps = np.round(np.max(np.bincount(A)) / np.bincount(A)).astype(int)
            for video, a in zip(videos.copy(), A):
                videos += [video] * (reps[a]-1)
        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        # sometimes we use the smaller or the bigger images
        transform = self.transform_small
        imgsize = IMG_SIZE
        if self.transform_big:
            if np.random.rand() < 0.5:
                imgsize = IMG_LARGE_SIZE
                transform = self.transform_big

        video = self.videos[i]
        sample = {}
        
        sample['X'] = np.array([(imread(f'{self.dirname}_{imgsize}/{self.fold}/frames/{fname}.png', True)[..., np.newaxis]/255).astype(np.float32) for fname in video])
        
        fname = video[-1]
        sample['A'] = np.array(int(open(f'{self.dirname}_{imgsize}/{self.fold}/labels/{fname}.txt').read()) - 1, np.int64)
   
        if transform:
            sample = transform(sample)
        
        # swap color axis: numpy (HWC), but torch (CHW)
        sample['X'] = sample['X'].transpose((0, 3, 1, 2))
        
        # numpy array -> torch tensor
        for k in sample:
            sample[k] = torch.from_numpy(np.ascontiguousarray(sample[k]))
        return sample


#%% ############################### UTILS FOR NETWORKS ########################

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return (x.contiguous()).view(self.shape)

class SliceRNN(nn.Module):
    def forward(self, x):
        output, hn = x
        return output[:, -1]

  
#%% ############################### CLASSIFICATION NETWORK ####################

class Model_X_A(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        
        # 128 => 64, 8, 8   (4096)
        # 256 => 16, 16, 16 (4096)
        # 512 => 4, 32, 32  (4096)        
        last_kernel = {32: 1024, 64: 256, 128: 64, 256: 16, 512: 4}
        
        self.net = nn.Sequential(
            Reshape(-1, CHANNELS_SIZE, IMG_SIZE, IMG_SIZE),
            nn.Conv2d(CHANNELS_SIZE, 64, 3, 2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, padding=1), nn.ReLU(),
            nn.Conv2d(64, last_kernel[IMG_SIZE], 3, 2, padding=1), nn.ReLU(),
            
            Reshape(-1, timesteps, 64*8*8),
            nn.Dropout(),
            
            nn.RNN(64*8*8, 128, batch_first= True),
            SliceRNN(),
            
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, NCLASSES),
        )

    def forward(self, x):
        return (self.net(x),)

#%% ############################### LOSSES DEFINITION ########################

def ce(preds, labels):
    return nn.CrossEntropyLoss()(preds, labels[:, 0])

def mse(preds, labels):
    return torch.mean((preds - labels)**2)

def acc(preds, labels):
    return torch.sum(preds.argmax(1) == labels[:, 0]) / len(labels)

def bacc(preds, labels):
    labels = labels[:, 0]
    preds = preds.argmax(1)
    return torch.mean(torch.tensor([
        torch.sum((preds == k) & (labels == k)) / torch.sum(labels == k)
        for k in torch.unique(labels)]))

def mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))

def dice(preds, labels):
    smooth = 1.
    num = 2 * (preds * labels).sum()
    den = preds.sum() + labels.sum()
    return 1 - (((num+smooth) / (den+smooth)) / len(labels))

def bce_dice(preds, labels):
    return nn.BCELoss()(preds, labels) + dice(preds, labels)

def f1_k(k):
    def f(preds, labels):
        labels = (labels[:, 0] == k).numpy().astype(int)
        preds = (preds.argmax(1) == k).numpy().astype(int)
        return f1_score(labels, preds)
    f.__name__ = f'f1-{k}'
    return f



#%% ######################## TRAINING METHODS ################################

def predict(model, device, ds):
    Y = []
    preds = []
    with torch.no_grad():
        model.eval()
        for batch in ds:
            X = [batch['X'].to(device)]
            Y.append([batch['A']])
            ps = model(*X)
            preds.append([p.cpu() for p in ps])
    return [torch.cat([o[0] for o in preds])], [torch.cat([o[0] for o in Y])]

def compute_metrics(fold, predictions, labels, metrics):
    ret = {}
    with torch.no_grad():
        for m in metrics['A']:
            ret[f'{fold}_A_{m.__name__}'] = m(predictions[0], labels[0])
    return ret

def train(model, device, epochs, tr, ts, metrics, losses, losses_weights, lr):
    history = []
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        tic = time()

        # Train
        train_loss, valid_loss = 0.0, 0.0
        model.train()
        
        train_labels = []
        train_preds = []
            
        for bi, batch in enumerate(tr):
            X = [batch['X'].to(device)]
            Y = [batch['A'].to(device)]
            if bi == 0:
                X[0].requires_grad = True

            optimizer.zero_grad()
            preds = model(*X)
            loss = 0
            
            loss += losses['A'](preds[0], Y[0]) * losses_weights['A']
            train_loss += loss / len(tr)
            loss.backward()
            optimizer.step()
            
            train_labels.append([y.cpu() for y in Y])
            train_preds.append([p.cpu() for p in preds])
            
            # Calculate and save gradients to folder "grads"
            # if bi == 0:  # save gradients relative to X
            #     x = X[0]
            #     for xi in range(x.shape[0]):
            #         xx = x[xi].detach().cpu().numpy()
            #         xx = np.concatenate(list(xx), 2)  # concat horizontally
            #         gg = x.grad[xi].cpu().numpy()
            #         gg = np.abs(gg)  # normalize gradients
            #         gg = (1 - (gg-gg.min()) / (gg.max()-gg.min()))
            #         # gg = dilation(gg, disk(2))
            #         gg = np.concatenate(list(gg), 2)  # concat horizontally
            #         gg = erosion(gg[0, :, :], disk(1))
            #         gg = np.expand_dims(gg, axis=0)
            #         xx = np.transpose(xx, (1, 2, 0)) * 0.4
            #         gg = np.transpose(gg, (1, 2, 0)) * 0.6
            #         fig = np.concatenate((xx+gg, xx+0.6, xx+gg), 2)
            #         fig = (fig*255).astype(np.uint8)
            #         imsave(f'grads\\grads-epoch{epoch}-image{xi}.png', fig[:, :, :3])

        train_labels = [torch.cat([o[0] for o in train_labels])]
        train_preds = [torch.cat([o[0] for o in train_preds])]
        
        # Evaluate
        avg_metrics = dict(
            train_loss=train_loss.cpu().detach(),
            **compute_metrics('train', train_preds, train_labels, metrics),
            **compute_metrics('test', *predict(model, device, ts), metrics),
        )

        toc = time()
        print('- %ds - %s' % (toc-tic, ' - '.join(f'%s: %f' % t for t in avg_metrics.items())))
        history.append(avg_metrics)
        
    return history


