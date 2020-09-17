import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import Dataset
import Network

# for reproducible results
torch.manual_seed(0)
# torch.cuda.manual_seed(0) #called internally from torch.manual_seed()
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(epochs, model, train_dl, val_dl, optimizer, criterion, train_size, val_size):
  train_loss_lst, val_loss_lst, batch_epoch_loss_lst = [], [], []
  
  for e in range(epochs):
    train_loss, val_loss = 0, 0
    
    # Training
    for batch_idx,(batch_imgs, batch_labels) in enumerate(train_dl):
      model.train()
      optimizer.zero_grad()
      batch_imgs,batch_labels = batch_imgs.float().to(device),batch_labels.to(device)
      output = model(batch_imgs)
      
      batch_labels = batch_labels[:, :2, :]
      # Reshape the outputs batch_size x 28 -> batch_size x 2 x 14
      output = output.view(batch_labels.shape)
      
      loss = criterion(output,batch_labels.float())
      batch_epoch_loss_lst.append(loss.item())
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
    
    train_loss_lst.append(train_loss/train_size)
    
    # Validation
    for batch_idx,(batch_imgs,batch_labels) in enumerate(val_dl):
      model.eval()
      batch_imgs, batch_labels = batch_imgs.float().to(device),batch_labels.to(device)
      output = model(batch_imgs)
      
      batch_labels = batch_labels[:, :2, :]
      # Reshape the outputs batch_size x 28 -> batch_size x 2 x 14
      output = output.view(batch_labels.shape)
      loss = criterion(output,batch_labels.float())
      val_loss += loss.item()
    
    val_loss_lst.append(val_loss/val_size)
    
    if e%2==0:
      print("[{}/{}]: Train loss={:2.4f}, Validation loss={:2.4f}".format(e+1,epochs,train_loss_lst[-1],val_loss_lst[-1]))

    if train_loss_lst[-1]<=0.25:
      for param in optimizer.param_groups:
        param["lr"]=5e-4

    if train_loss_lst[-1]<=0.15:
      for param in optimizer.param_groups:
        param["lr"]=1e-4

  return train_loss_lst, val_loss_lst, batch_epoch_loss_lst

def main():
  # lsp_extended_dataset = Dataset.LSP_Dataset(path="./lspet_dataset", is_lsp_extended_dataset=True)
  lsp_dataset = Dataset.LSP_Dataset()
  dataset = torch.utils.data.ConcatDataset([lsp_dataset]) #lsp_extended_dataset

  batch_size = 16
  total = len(dataset)
  train_size, val_size, test_size = int(total*0.6), int(total*0.2), int(total*0.2)

  lengths = [train_size, val_size, test_size]
  train_dataset, val_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, lengths)

  train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dl   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
  test_dl  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  model = Network.DeepPose().float().to(device)

  criterion = nn.MSELoss(reduction="sum")
  optimizer = torch.optim.Adagrad(model.parameters(),lr=1e-3)

  train_loss_lst, val_loss_lst, batch_epoch_loss_lst = train( epochs=100, 
                                                              model=model, 
                                                              train_dl=train_dl, 
                                                              val_dl=val_dl, 
                                                              optimizer=optimizer, 
                                                              criterion=criterion, 
                                                              train_size=train_size, 
                                                              val_size=val_size)


if __name__ == "__main__":
  main()