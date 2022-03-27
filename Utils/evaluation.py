import numpy as np 
import torch 
from env.envs import *

def evaluate(policy,env_type,seed,max_step,eval_episode_num,device):
   
    eval_env = UniTreeEnv(
        obs_type= ["vision",'joints','inertial'],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "DELTA_POSITION_CONTROL",
            pb_control_kwargs= dict(force= 20),
            simulate_timestep= 1./500,
        ),
        connection_mode= p.DIRECT,
        env=env_type,
        bullet_debug= True,
    )
    env.load_obstacle()
    obs_prop,obs_visual=env.reset()

    eval_episode_rewards=list()
    while len(eval_episode_rewards)<eval_episode_num:
        episode_reward=0
        for step in range(max_step):
            with torch.no_grad():
                _,action,_=policy.act(obs_prop,obs_visual)
                
                obs,reward,done,infos=eval_env.step(action)
                episode_reward+=reward

                if(done):
                    break
        eval_episode_rewards.append(episode_reward)

    eval_env.close()

