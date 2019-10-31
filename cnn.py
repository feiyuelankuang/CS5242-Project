import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Conv2d(nn.Module):
    def __init__(self, inchan, outchan, downsample=False):
        super().__init__()
        self.conv2d = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        if self.downsample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        if self.downsample:
            x = self.Downsample(x)
        return x

class MalwareNet(nn.Module):
    def __init__(self, in_feature, num_classes=2):
        super().__init__()
        ics = [in_feature,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.nlays = len(ics) #number of layers

        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sps[i]) for i in range(self.nlays)])

        #Group Normalization with group = 1
        self.GN = nn.ModuleList([nn.GroupNorm(1,ocs[i]) for i in range(self.nlays)])

        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):
        # Feedforward
        for i in range(self.nlays):            
            x = F.relu(self.conv[i](x))          
            x = self.GN[i](x)
        # classifier                
        out = F.avg_pool2d(x, x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
 
        return out