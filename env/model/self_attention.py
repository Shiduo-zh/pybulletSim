
from math import sqrt
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np


class Self_Attention(nn.Module):
    "input : batch_size * seq_len * input_dim"
    " q : batch_size * input_dim * dim_k"
    " k : batch_size * input_dim * dim_k"
    " v : batch_size * input_dim * dim_v"
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
        
    
    def forward(self,x):
        """
        x denotes Tm in this work
        Q,K,V denote Tq,Tk,Tv respectively
        """
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        return output