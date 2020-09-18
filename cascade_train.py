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

    device = torch.device("cuda" if next(trained_model.parameters()).is_cuda else "cpu")
    
    stage1_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    trained_model.eval()
    idx = 0
    for batch_idx,(batch_imgs, batch_labels) in enumerate(stage1_dl):
      
      batch_imgs, batch_labels = batch_imgs.float().to(device),batch_labels.to(device)
      output = model(batch_imgs)
      
      batch_labels = batch_labels[:, :2, :].permute((0,2,1))
      # Reshape the outputs of shape (batch_size x 28) -> (batch_size x 14 x 2)
      output = output.view(batch_labels.shape)

      self.prev_stage_prediction[idx : idx+batch_labels.shape[0]] = output
      idx = idx + batch_labels.shape[0]


  def __getitem__(self, idx):
    return 0

  def __len__(self):
    # sampling 40 locations around join location to simulate prev stage prediction
    return len(self.stage1_dataset)*40