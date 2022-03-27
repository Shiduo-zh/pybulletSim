from distutils.command.clean import clean
import numpy as np
import torch 
import torch.nn as nn
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
   
    envs=UniTreeEnv(
        obs_type= ["vision",'joints','inertial'],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "DELTA_POSITION_CONTROL",
            pb_control_kwargs= dict(force= 20),
            simulate_timestep= timestep,
            default_base_transform=np.array([0,0,0.42,0,0,1,1]),
        ),
        connection_mode= mode,
        env=args.env_type,
        bullet_debug= True,
    )
    

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
        envs.reset()
        episode_reward=0
        for step in range(num_steps):
            with torch.no_grad():
                value,action,action_log_prob=policy.act(
                    trajectory.obs_prop[step],
                    trajectory.obs_visual[step]
                )
            obs,reward,done,infos=envs.step(action.cpu().numpy())

            masks=torch.FloatTensor([1.0] if not done else [0.0]).to(device)
            bad_masks=torch.FloatTensor([1.0]).to(device)
            trajectory.insert(obs['prop'],obs['vision'],
                            action,action_log_prob,value,torch.tensor(reward),masks,bad_masks)
            
            episode_reward+=reward
            episode_steps=infos['episode_steps']
            
            if(done):
                break
        
        print('episode:',episode,'  steps:',episode_steps)   
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
                save_path=os.path.join(args.save_dir)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass
                torch.save(policy,os.path.join(save_path,args.env_type + '.pt'))
            
        if((episode+1) %args.log_interval==0):
                print(
                    "{} log update, total step{},\n mean and median reward{:.1f} and {:.1f},\n min/max reward{:.1f} and {:.1f},entropy:{},value_loss{},action_loss{}"
                    .format(episode+1,total_steps,
                    np.mean(episode_rewards),np.median(episode_rewards),
                    np.max(episode_rewards),np.min(episode_rewards),
                    dist_entropy,value_loss,action_loss
                    )
                )

if __name__=="__main__":
    main()