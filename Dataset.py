import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import skimage.transform

import torch

class LSP_Dataset(torch.utils.data.Dataset):
  
  def __init__(self, path="./lsp_dataset", is_lsp_extended_dataset=False): #/content/drive/My Drive
    self.path = path
    self.is_lspet = is_lsp_extended_dataset

    imgs_list = sorted(os.listdir(os.path.join(path, "images")))
    
    # Load joints data from the mat file
    self.joint_data = scipy.io.loadmat(os.path.join(path, "joints.mat"))["joints"]
    if self.is_lspet:
      self.joint_data = self.joint_data.transpose((1, 0, 2))
    
    self.dataset_size = self.joint_data.shape[2]
    
    assert len(imgs_list) == self.dataset_size

    self.max_h, self.max_w = 196, 196

    # Load and store images (float) into a list
    self.array_of_images = np.empty([self.dataset_size, self.max_h, self.max_w, 3], dtype=float)
    self.array_of_labels = np.empty([self.dataset_size, 3, 14], dtype=float) #N x (X,Y) x (14 joints)
    
    # DeepPose: 4.1: Experimental Details -> "For LSP we use the full image as initial bounding box since the humans are relatively tightly cropped by design."
    # Read Section 3.4 of "Stacked Hourglass Networks for Human Pose Estimation" and Section 4.2 of "Convolutional Pose Machines" for more augmentations, normalizations and handling of single person pose detection in a multi-person scene.
    # both padding and resizing are commonly used approaches to obtain fixed size images required by the CNN, we go with padding here to preserve human body aspect ratio.
    for file_idx, file_name in enumerate(imgs_list):
      img, labels = self.scale_and_pad( plt.imread(os.path.join(path, "images", file_name)), self.joint_data[:2,:,file_idx])
      
      self.array_of_images[file_idx] = img
      self.array_of_labels[file_idx, :2, :] = labels
      self.array_of_labels[file_idx, 2, :]  = self.joint_data[2, :, file_idx]
    
    print(f"Built Dataset: found {self.__len__()} image-target pairs")

  
  def scale_and_pad(self, img, labels):
    scale_factor = self.max_h/max(*img.shape)

    # https://scikit-image.org/docs/dev/api/skimage.transform.html#rescale -> the input image is converted according to the conventions of img_as_float
    scaled_img = skimage.transform.rescale(img, scale=scale_factor, multichannel=True) #anti_aliasing=True

    img_h, img_w, _   = scaled_img.shape
    padded_scaled_img = np.zeros([self.max_h, self.max_w, 3])
    start_h, start_w  = int((self.max_h - img_h)/2), int((self.max_w - img_w)/2)

    padded_scaled_img[start_h:start_h + img_h, start_w:start_w + img_w, :] = scaled_img
    padded_scaled_labels = (labels*scale_factor + np.array([[start_w], [start_h]]))/self.max_h - 0.5
    return padded_scaled_img, padded_scaled_labels
  
  
  def __getitem__(self,idx):
    return self.array_of_images[idx], self.array_of_labels[idx]

  
  def __len__(self):
    return self.array_of_images.shape[0]
  
  
  def print_sample(self, sample_idx):
    if self.is_lspet:
      file_name = f"im{sample_idx + 1:05d}.jpg"
    else:
      file_name = f"im{sample_idx + 1:04d}.jpg"

    original_img   = plt.imread(os.path.join(self.path, "images", file_name))
    visualized_img = plt.imread(os.path.join(self.path, "visualized", file_name))
    normalized_img = self.array_of_images[sample_idx]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 2]}) # figsize=(14,14)
    
    ax1.title.set_text('Original')
    ax1.imshow(original_img)
    for i in range(14):
      if self.joint_data[2, i, sample_idx] == 0.0: c = 'b'
      else: c = 'r'
      ax1.plot(self.joint_data[0, i, sample_idx], self.joint_data[1, i, sample_idx],'.', color=c)

    ax2.title.set_text('Visualized')
    ax2.imshow(visualized_img)

    ax3.title.set_text('Normalized')
    ax3.imshow(normalized_img)
    for i in range(14):
      if self.__getitem__(sample_idx)[1][2, i] == 0.0: c = 'b'
      else: c = 'r'
      ax3.plot(self.max_h*(0.5 + self.__getitem__(sample_idx)[1][0, i]), 
               self.max_h*(0.5 + self.__getitem__(sample_idx)[1][1, i]),
              '.', 
              color=c)
    
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
  dataset = LSP_Dataset()
  dataset.print_sample(3)

    # self.max_h, self.max_w, self.min_h, self.min_w = 0, 0, float('inf'), float('inf')
    # for file_name in imgs_list:
    #   img = plt.imread(os.path.join(path, "images", file_name))
    #   if img.shape[0] > self.max_h: self.max_h = img.shape[0]
    #   if img.shape[0] < self.min_h: self.min_h = img.shape[0]
    #   if img.shape[1] > self.max_w: self.max_w = img.shape[1]
    #   if img.shape[1] < self.min_w: self.min_w = img.shape[1]
#https://pytorch.org/docs/stable/torchvision/transforms.html
#https://discuss.pytorch.org/t/data-augmentation-in-pytorch/7925
