import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from myRL.distributions import Bernoulli, Categorical, DiagGaussian
from myRL.utils import init
from env.model.network import localTransformer


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        #init base by default if params not given
        if base is None:
            base=localTransformer()

        self.base = base
        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(base.output_size, num_outputs)
       
    # def forward(self, inputs, rnn_hxs, masks):
    #     raise NotImplementedError

    def act(self, input1,input2, deterministic=False):
        value, actor_features= self.base(input1,input2)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self,input1,input2):
        value, _= self.base(input1,input2)
        return value

    def evaluate_actions(self,input1,input2, action):
        value, actor_features= self.base(input1,input2)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


