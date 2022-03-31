"old version and this file is never used in this project actually"
from turtle import forward
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

IN_PROP=93
OUT_CHANNELS =128

class VisualCNN(nn.Module):
    def __init__(self):
        super(VisualCNN, self).__init__()

        self.fc1=nn.Linear(256,128)
        self.fc2=nn.Linear(128,32)
        self.fc3=nn.Linear(32,4)
        

    def forward(self, x_delta):

        x_delta=F.relu(self.fc1(x_delta))
        x_delta=F.relu(self.fc2(x_delta))
        out=F.relu(self.fc3(x_delta))

        return out

class LinearCoder(nn.Module):
    def __init__(self):
        super().__init__(LinearCoder,self)
        self.l1=torch.nn.Linear(IN_PROP,OUT_CHANNELS)
        self.l2=nn.Linear(OUT_CHANNELS,OUT_CHANNELS)
    
    def forward(self,prop):
        prop1=F.relu(self.l1(prop))
        prop2=F.relu(self.l2(prop1))
        return prop2







