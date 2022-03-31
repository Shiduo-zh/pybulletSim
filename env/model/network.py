from tkinter import E
from env.model.mlp import *
from env.model.conv2d import *
from env.model.transformer import *
from env.model.utils import*
import torch
import torch.nn as nn


class localTransformer(nn.Module):
    def __init__(self,prop_size=93,hidden_size=256,output_size=128,
                input_channels=4,output_channnels=128,
                action_size=12,spatial_size=4,batch_size=31):

        """
        params:
        prop_size:the dimension of proprioceptive state
        hidden_size:the hidden layer units size of mlp for proprioceptive state encoder and projection head

        """
        super(localTransformer,self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.spatial_size=spatial_size
        self.input_channels=input_channels
        self.visual_channels=output_channnels
        self.output_size=action_size
        self.batch_size=batch_size
        #propriceptive state encoder
        self.prop_encoder=MlpModel(prop_size,hidden_size,output_size)
        self.linear=nn.Linear(output_size,output_size)
        #depth visual infomation encoder
        self.visual_encoder=Conv2dModel(input_channels,[output_channnels],[(16,16)],[(16,16)])
        #2 transformer layers
        self.translayer1=transformerEncoder(input_dim=output_channnels,
                                            dim_k=output_channnels,
                                            dim_v=output_channnels)
        self.translayer2=transformerEncoder(input_dim=output_channnels,
                                            dim_k=output_channnels,
                                            dim_v=output_channnels)
        #projection head
        self.projection_head=MlpModel(input_size=2*output_channnels,
                                        hidden_sizes=hidden_size,  
                                        output_size=action_size)
        
        self.critic_linear =nn.Sequential(init_(nn.Linear(action_size, 1)),nn.ReLU())
        

    def forward(self,props,visions):
        visions=visions.reshape((-1,self.input_channels,64,64))
        props=props.reshape((-1,93))
        if(not torch.is_tensor(visions) or not torch.is_tensor(props)):
            props= torch.from_numpy(props).to(torch.float32)
            visions=torch.from_numpy(visions).to(torch.float32)
        #print(visions.shape)
        E_prop=self.prop_encoder(props)
        E_vision=self.visual_encoder(visions)
        #print(E_prop.shape,E_vision.shape)
        #.T.reshape((self.spatial_size,self.spatial_size,self.visual_channels))
        
        # weight,bias=self.prop_encoder.linear_params()
        # w_prop=weight[-1]
        # b_prop=bias
        # t_prop=np.dot(w_prop,E_prop)+b_prop
        t_prop=self.linear(E_prop)
        batch_size=t_prop.shape[0]
        #print(t_prop.shape)
        T0=t_prop
        for i in range(self.spatial_size):
            for j in range(self.spatial_size):
                T0=torch.cat((T0,E_vision[:,:,i,j]))
        T0=T0.view(batch_size,self.spatial_size**2+1,-1)
        T1=self.translayer1(T0)
        T2=self.translayer2(T1)
        #T2 is a tensor with shape of (N^2+1)*C
        vision_output=T2[:,1:,:]
        prop_feature=T2[:,0,:]
        vision_feature=torch.sum(vision_output,dim=1)/self.spatial_size**2
        features=torch.cat((prop_feature,vision_feature)).view(-1)
        if(batch_size>1):
            features=features.view(batch_size,-1)
        #print('features size:',features.size())

        action=self.projection_head(features)

        value=self.critic_linear(action)
        return value,action

        