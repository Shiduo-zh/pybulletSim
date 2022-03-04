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
        

    #传递函数，使用激活函数层层训练
    def forward(self, x_delta):
        "x_delta为当前层训练数据"
        #输入层到隐藏层1
        x_delta=F.relu(self.fc1(x_delta))
        #隐藏层1到隐藏层2
        x_delta=F.relu(self.fc2(x_delta))
        #隐藏层2到输出层
        out=F.relu(self.fc3(x_delta))

        return out

#本体特征的线性转换层
class LinearCoder(nn.Module):
    "两层MLP"
    def __init__(self):
        super().__init__(LinearCoder,self)

        self.l1=torch.nn.Linear(IN_PROP,OUT_CHANNELS)
        self.l2=nn.Linear(OUT_CHANNELS,OUT_CHANNELS)
    
    def forward(self,prop):
        prop1=F.relu(self.l1(prop))
        prop2=F.relu(self.l2(prop1))
        return prop2







