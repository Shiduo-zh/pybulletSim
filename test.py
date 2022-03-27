from distutils.command.clean import clean
import numpy as np
import torch 
import torch.nn as nn
from env.terrain.thin_obstacle import thin_obstacle
import pybullet as p
from collections import deque
import time
import glob
import time
import os

from utils.arguments import get_args
from env.envs import UniTreeEnv
from myRL.policy import Policy
from myRL.trajectory import Trajectory
from myRL.algo.ppo import *
from myRL.utils import *
from env.model.network import localTransformer
from torch.utils.data.distributed import DistributedSampler

 

if __name__=='__main__':
    import tqdm
    env=UniTreeEnv(
        obs_type= ["vision",'joints','inertial'],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "POSITION_CONTROL",
            pb_control_kwargs= dict(force= 20),
            simulate_timestep= 1./500,
            default_base_transform=np.array([0,0,0.43,0,0,1,1])
        ),
        connection_mode= p.GUI,
        env='thin_obstacle',
        bullet_debug= True,
    )
    action_list=np.load('action_list.npy')
    param=np.load()
    for i in tqdm.tqdm(range(10000)):
        # env.debug_step()
        # env.pb_client.stepSimulation()
        # image = env.render(camera_name= "front")
        obs,reward,done,info = env.step(action_list[i])
        #time.sleep(1./500)
