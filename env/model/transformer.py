
import torch
import torch.nn as nn
from env.model.self_attention import Self_Attention
from env.model.feed_foward import Feed_Forward
from env.model.add_norm import Add_Norm
from env.model.mlp import *
from env.model.conv2d import *

class transformerEncoder(nn.Module):
    def __init__(self,input_dim,dim_k,dim_v):
        super(transformerEncoder,self).__init__()

        self.SA=Self_Attention(input_dim,dim_k,dim_v)#need params
        
        self.feed_forward=Feed_Forward(input_dim)
        
        self.add_norm=Add_Norm()
    
    def forward(self,embedding):
        output=self.add_norm(embedding,self.SA)
        output=self.add_norm(output,self.feed_forward)

        return output