from distutils.command.clean import clean
from re import T
import numpy as np
from pandas import wide_to_long
import torch 
import torch.nn as nn
from exptools.logging.console import colorize
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
from utils.log import *

from env.terrain.thin_obstacle import *
from env.terrain.wide_obstacle import *
from env.terrain.moving_obstacle import *
from env.terrain.office import *
from env.terrain.unflatten import *
from env.model.mlp import *
from env.model.conv2d import *
from env.model.transformer import *

class PreUnitreeEnv(UniTreeEnv):
    def __init__(self,
            obs_type= ["vision",'joints','inertial'],
            robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "POSITION_CONTROL",
            pb_control_kwargs= dict(force= 20),
            simulate_timestep= 1./500,
            default_base_transform=np.array([0,0,0.42,0,0,1,1]),
        ),
        connection_mode= p.DIRECT,
        surrounding_tpye='thin',
        bullet_debug= True,):
        super().__init__(obs_type= ["vision",'joints','inertial'],
            robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "POSITION_CONTROL",
            pb_control_kwargs= dict(force= 20),
            simulate_timestep= 1./500,
            default_base_transform=np.array([0,0,0.42,0,0,1,1]),
        ),
        connection_mode= p.DIRECT,
        surrounding_tpye='thin',
        bullet_debug= True,)
    
    def compute_reward(self, delta_time, bonus_flag, next_obs, energy_coef=0.001, forward_coef=1, alive_coef=0.1, start_pos=..., end_pos=...):
        current_vel=self._get_obs()['inertial'][3:6]
        R_forward=current_vel[1]
        return R_forward

def main():
    args=get_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark=True
    
    #torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    #torch.distributed.init_process_group(backend='nccl')

    log_dir=os.path.expanduser(args.log_dir)
    evaluate_dir=log_dir+'_eval'
    cleanup_log_dir(log_dir)
    cleanup_log_dir(evaluate_dir)
    
    #simulation settings
    mode=p.GUI if args.connection_mode=='visual' else p.DIRECT
    timestep=1./500
   
    envs=PreUnitreeEnv(
        obs_type= ["vision",'joints','inertial'],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "POSITION_CONTROL",
            pb_control_kwargs= dict(force= 20),
            simulate_timestep= timestep,
            default_base_transform=np.array([0,0,0.42,0,0,1,1]),
        ),
        connection_mode= mode,
        surrounding_tpye=args.env_type,
        bullet_debug= True,
    )
    
    surrounding_type=args.env_type
    if("thin" in surrounding_type):
        surrounding=ThinObstacle()
    elif("wide" in surrounding_type):
        surrounding=WideObstacle()
    elif('moving' in surrounding_type):
        surrounding=MovingObstacle()
    elif('office' in surrounding_type):
        surrounding=office()
    elif('mountain' in surrounding_type):
        surrounding=unflatten_terrain()
    
    surrounding.init_env()
    surrounding.create_obstacle()
    obs_prop,obs_visual=envs.reset()
    
    
    net=localTransformer()
    net.to(device)

    policy=Policy(
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        base=net,
    )
    
    policy.to(device)

    algo=PPO(
        actor_critic=policy,
        clip_param=args.clip,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        lr=args.policy_lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )
    

    trajectory=Trajectory(
        num_steps=args.horizen,
        prop_shape=envs.observation_space['prop'].shape,
        visual_shape=(4,)+envs.observation_space['vision'].shape,
        action_space=envs.action_space

    )
    
    obs_prop=torch.from_numpy(obs_prop).to(device)
    obs_visual=torch.from_numpy(obs_visual).to(device)
    #print(obs_visual.shape)
    
    trajectory.obs_prop[0].copy_(obs_prop)
    trajectory.obs_visual[0].copy_(obs_visual)
    trajectory.to(device)

    episode_rewards=deque(maxlen=10)
    total_steps=0
    
    num_env_step=args.num_env_steps
    num_steps=args.horizen
    #num_progresses=args.num_prosesses
    num_episode=int(num_env_step)//num_steps
    #num_updates=num_episode//num_progresses

    for episode in range(num_episode):
        #for a episode
        surrounding.reload_obstacle()
        obs_prop,obs_visual=envs.reset()
        
        obs_prop=torch.from_numpy(obs_prop).to(device)
        obs_visual=torch.from_numpy(obs_visual).to(device)
        trajectory.obs_prop[0].copy_(obs_prop)
        trajectory.obs_visual[0].copy_(obs_visual)
        trajectory.to(device)

        episode_reward=0
        snapshots=np.array([])
        for step in range(num_steps):
            with torch.no_grad():
                value,action,action_log_prob=policy.act(
                    trajectory.obs_prop[step],
                    trajectory.obs_visual[step]
                )
            obs,reward,done,infos=envs.step(action.cpu().numpy(),surrounding)

            masks=torch.FloatTensor([1.0] if not done else [0.0]).to(device)
            bad_masks=torch.FloatTensor([1.0]).to(device)
            trajectory.insert(obs['prop'],obs['vision'],
                            action,action_log_prob,value,torch.tensor(reward),masks,bad_masks)
            
            episode_reward+=reward
            episode_steps=infos['episode_steps']
            # if((episode+1)%args.log_interval==0):
            # if(step>=0 and step<100):
            width,height,image=envs.snapshot()
            image=image.reshape(1,*image.shape)
            snapshots=np.append(snapshots,image)
            snapshots=snapshots.reshape((-1,480,480,4))
            # if(step>190 and step<=210):
            #         width,height,image=envs.snapshot()
            #         image=image.reshape(1,*image.shape)
            #         snapshots=np.append(snapshots,image)
            #         snapshots=snapshots.reshape((-1,480,480,4))
            
            if(done):
                break
        
        res='episode:'+str(episode)+'  steps:'+str(episode_steps)+' reward:'+str(episode_reward)
        systime='['+str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+']'
        log_gif(args.log_dir,args.run_id,episode,snapshots.astype(np.uint8))

        print(colorize(systime,color='yellow'),colorize(res,color='green'))
        episode_rewards.append(episode_reward)
        total_steps+=episode_steps
        with torch.no_grad():
            next_value=policy.get_value(trajectory.obs_prop[-1],trajectory.obs_visual[-1])
            

        trajectory.compute_returns(next_value,args.discount)
        
        value_loss,action_loss,dist_entropy=algo.update(trajectory)

        trajectory.after_update()

        if(((episode+1) % args.save_interval ==0 
            or episode==num_episode-1)
            and args.save_dir !=''):
                save_path=os.path.join(args.save_dir,args.run_id)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass
                torch.save(policy,os.path.join(save_path,args.env_type + '.pt'))
            
        if((episode+1) %args.log_interval==0):
            episode_infos=dict()

            text="{} log update, total step{},\n mean and median reward{:.1f} and {:.1f},\n max/min reward{:.1f} and {:.1f},entropy:{},value_loss{},action_loss{}"\
                    .format(episode+1,total_steps,
                    np.mean(episode_rewards),np.median(episode_rewards),
                    np.max(episode_rewards),np.min(episode_rewards),
                    dist_entropy,value_loss,action_loss
                    )
            
            print(colorize(text,color='green'))
            # episode_infos['episode']=episode
            # episode_infos['images']=snapshots
            # episode_infos['reward']=text
            # log(args.log_dir,args.run_id,'episode'+str(episode),**episode_infos)
            
                


if __name__=="__main__":
    main()