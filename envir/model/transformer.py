from operator import imod
import torch
import torch.nn as nn
from self_attention import Self_Attention
from feed_foward import Feed_Forward
from add_norm import Add_Norm

class transformerEncoder(nn.Module):
    def __init__(self,embedding):
        super(transformerEncoder).__init__()
        self.embedding=embedding

        self.SA=Self_Attention()#need params
        
        self.feed_forward=Feed_Forward()
        
        self.add_norm=Add_Norm()
    
    def forward(self):
        output=self.add_norm(self.embedding,self.SA,y=self.embedding)
        output=self.add_norm(output,self.feed_forward)

        return output