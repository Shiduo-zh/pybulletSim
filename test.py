from distutils.command.clean import clean
import numpy as np
import torch 
import torch.nn as nn
from env.terrain.thin_obstacle import ThinObstacle
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
import matplotlib.pyplot as plt

 

if __name__=='__main__':
    import tqdm
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    env=UniTreeEnv(
        obs_type= ["vision",'joints','inertial'],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "POSITION_CONTROL",
            pb_control_kwargs= dict(force= 5),
            simulate_timestep= 1./50,
            default_base_transform=np.array([0,0,0.35,0,0,1,1])
        ),
        connection_mode= p.GUI,
        surrounding_tpye='thin_obstacle',
        bullet_debug= True,
    )
    obs_all=env._get_obs()
    vel=obs_all['inertial'][3:6]

    obs_prop,obs_visual=env.reset()
    obs_prop=np.float32(obs_prop)
    obs_visual=np.float32(obs_visual)

    action_list=np.load('action_list.npy')
    
    model=torch.load('./trained_models/thin_obstacle.pt')
    model.to(device)
    model=model.eval()
    zero_action=np.zeros(12)
    _,action,_=model.act(torch.from_numpy(obs_prop).to(device),torch.from_numpy(obs_visual).to(device))
    step=0
    total_reward=0
    
    for i in range(10000):
        # env.debug_step()
        # env.pb_client.stepSimulation()
        # image = env.render(camera_name= "front")
        #obs,reward,done,infos=env.step(action,None)
        env.step_simulation_from_action(np.clip(action_list[i%600],env.action_space.low,env.action_space.high))
        # _,action,_=model.act(obs['prop'].float().to(device),obs['vision'].float().to(device))

        #width,height,image=env.snapshot()
        #plt.imshow(image)
        #plt.show()
        # step+=1
        # total_reward+=reward
        # if(done==True):
        #     print('step',step)
        #     print('reward',total_reward)
        #     break
        time.sleep(1./50)
        
