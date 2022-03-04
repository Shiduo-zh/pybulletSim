import torch
import torch.nn as nn

class Add_Norm(nn.Module):
    def __init__(self):
        self.dropout = nn.Dropout()#parameters to be added
        super(Add_Norm, self).__init__()

    def forward(self,x,sub_layer,**kwargs):
        "sub_layer can be self-attention or feed_forward"
        sub_output = sub_layer(x,**kwargs)
        x = self.dropout(x + sub_output)

        layer_norm = nn.LayerNorm(x.size()[1:])
        out = layer_norm(x)
        return out