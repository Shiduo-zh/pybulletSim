import torch
import torch.nn as nn

class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(0.1)#parameters to be added
        

    def forward(self,x,sub_layer,**kwargs):
        "sub_layer can be self-attention or feed_forward"
        sub_output = sub_layer(x,**kwargs)
        x = self.dropout(x + sub_output)

        layer_norm = nn.LayerNorm(x.size()[1:])
        layer_norm.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        out = layer_norm(x)
        return out