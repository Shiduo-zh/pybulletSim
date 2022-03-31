import numpy as np
import pybullet as p
import pybullet_data
import sys
sys.path.append('../../')

from terrain.obstacle_creator import creator
from terrain.wide_obstacle import wide_obstacle
from envir.envs import UniTreeEnv
import tqdm, time


env = UniTreeEnv(
        obs_type= ["vision","joints","inertial"],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "DELTA_POSITION_CONTROL",
            pb_control_kwargs= dict(force= 20),
            simulate_timestep= 1./500,
        ),
        connection_mode= p.GUI,
        env="thin_obstacle",
        bullet_debug= True,
    )
obs = env.reset()
env.load_obstacle()

for _ in tqdm.tqdm(range(10000)):
        # env.debug_step()
        # env.pb_client.stepSimulation()
        # image = env.render(camera_name= "front")
        
        #sample是做一个随机采样
        obs = env.step(env.action_space.sample())[0]
        time.sleep(1./500)


