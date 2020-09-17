import time

import torch
import torch.nn as nn

class DeepPose(nn.Module):
    def __init__(self):
        super(DeepPose,self).__init__()

        # CNN k 11*11, c 3 -> 96, stride 4,4
        # lrn 2, 2e-05, 0.75 
        # maxpooling k3*3, stride 2,2
        # CNN k 5*5, c -> 256 stride 1,1
        # lrn 2, 2e-05, 0.75
        # maxpooling k 3*3, stride 2,2
        # CNN k 3*3, c -> 384
        # CNN k 3*3, c -> 384
        # CNN k 3*3, c -> 256
        # Dense -> 4096
        # Dense -> 4096
        # Output layer -> num_classes
        
        # not reusing ReLU and dropout layer objects - lessons learnt from quantization
        # https://discuss.pytorch.org/t/using-same-dropout-object-for-multiple-drop-out-layers/39027/6

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.lrn1  = nn.LocalResponseNorm(size=2, alpha=2e-05, beta=0.75, k=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.lrn2  = nn.LocalResponseNorm(size=2, alpha=2e-05, beta=0.75, k=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0)
        
        self.relu_conv3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=0)
        self.relu_conv4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.relu_conv5 = nn.ReLU(inplace=True)
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.fc1 = nn.Linear(in_features=256, out_features=4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout_fc1 = nn.Dropout(p=0.6)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout_fc2 = nn.Dropout(p=0.6)
        
        self.out = nn.Linear(in_features=4096, out_features=28)
        
    def forward(self,input):
        x = input.view((input.shape[0],input.shape[3],input.shape[1],input.shape[2]))
        x = self.pool1(self.lrn1(self.relu_conv1(self.conv1(x))))
        x = self.pool2(self.lrn2(self.relu_conv2(self.conv2(x))))
        x = self.relu_conv3(self.conv3(x))
        x = self.relu_conv4(self.conv4(x))
        x = self.pool3(self.relu_conv5(self.conv5(x)))
        x = torch.flatten(x,1)
        x = self.dropout_fc1(self.relu_fc1(self.fc1(x)))
        x = self.dropout_fc2(self.relu_fc2(self.fc2(x)))
        x = self.out(x)

        return x

if __name__ == "__main__":
  start = time.time()

  model = DeepPose()
  output = model(torch.ones([10, 196, 196, 3]))
  print(f"output shape : {output.shape}")

  end = time.time()
  print(f"time taken: {end - start} ms")