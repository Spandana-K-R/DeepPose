# All training stages are done in sequence.

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import skimage.transform

import torch

class LSP_cascade_Dataset(torch.utils.data.Dataset):
  
  def __init__(self, lsp_stage1_dataset, trained_model):
    
    self.prev_stage_prediction = np.empty([len(lsp_stage1_dataset), 2, 14], dtype=float)
    
    for i in range(len(lsp_stage1_dataset)):
      img, label = lsp_stage1_dataset.__getitem__(i)
      out = model(img)
      out = out.view(label[:2, :].shape)

      self.prev_stage_prediction[i, :2, :] = out


  def __getitem__(self, idx):