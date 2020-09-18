# All training stages are done in sequence.

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import skimage.transform

import torch
from torch.utils.data import DataLoader


class LSP_cascade_Dataset(torch.utils.data.Dataset):
  
  def __init__(self, lsp_stage1_dataset, trained_model):
    self.stage1_dataset = lsp_stage1_dataset

    self.prev_stage_prediction = np.empty([len(lsp_stage1_dataset), 14, 2], dtype=float)
    self.gt_labels = np.empty([len(lsp_stage1_dataset), 14, 3], dtype=float)

    device = torch.device("cuda" if next(trained_model.parameters()).is_cuda else "cpu")
    
    stage1_dl = DataLoader(self.stage1_dataset, batch_size=16, shuffle=False)
    trained_model.eval()
    idx = 0
    for batch_idx,(batch_imgs, batch_labels) in enumerate(stage1_dl):
      
      batch_imgs, batch_labels = batch_imgs.float().to(device),batch_labels.to(device)
      output = model(batch_imgs)
      
      batch_labels = batch_labels.permute((0,2,1))
      # Reshape the outputs of shape (batch_size x 28) -> (batch_size x 14 x 2)
      output = output.view(batch_labels[:, :, :2].shape)

      self.prev_stage_prediction[idx : idx+batch_labels.shape[0]] = output
      self.gt_labels[idx : idx+batch_labels.shape[0]] = batch_labels
      idx = idx + batch_labels.shape[0]

    # extract diam(pose) from each sample as required for cropping subimages around joint locations -  defined as the distance between opposing joints on the human torso, such as left shoulder and right hip
    #right hip is 3rd (x,y) pair and left shoulder is 10th in our pose vector
    self.diam = torch.sqrt(
      torch.square(self.prev_stage_prediction[:, 2, 0] - self.prev_stage_prediction[:, 9, 0]) + 
      torch.square(self.prev_stage_prediction[:, 2, 1] - self.prev_stage_prediction[:, 9, 1]))

    # extract mean, standard deviation of predicted displacement for generating simulated predictions
    displacements = self.gt_labels - self.prev_stage_prediction
    self.displacement_means = displacements[:,:,:2].mean(axis=0)
    self.displacement_stds = displacements[:,:,:2].std(axis=0)


  def __getitem__(self, idx):
    # one sample in stage1_dataset will generate 14*40 = 560 samples for stage2 and 3
    stage1_sample_idx = int(idx / 560)
    joint_idx = int((idx % 560) / 40)

    # simulate a random stage1_prediction on stage1_sample_idx around joint_idx joint


  def __len__(self):
    # sampling 40 locations around each of 14 joint location to simulate prev stage prediction
    return len(self.stage1_dataset)*40*14