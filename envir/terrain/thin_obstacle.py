from re import A
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces, Env
from utils import line2float

from terrain import terrain
import sys
sys.path.append("../")

from envir.agent import UniTreeEnv
from time import sleep

class thin_obstacle():
    def __init__(self):
        self.controller=terrain()
        self.obstacle_id=np.array([])
        self.bonus_id=np.array([])
        self.bonus_num=0
    
    def create_obstacle(self):
        
        for i in range(6):
            self.obstacle_id=np.append(self.obstacle_id,self.controller.create_box([0.5,0.5,2],[-5,5*(i+1),2],10000))
            self.obstacle_id=np.append(self.obstacle_id,self.controller.create_box([0.5,0.5,2],[5,5*(i+1)+3,2],10000,[1,0,0,1]))
        
        for i in range(8):
            self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[0,3*(i+1),0.3],0))
    
    def get_obstacle_id(self):
        print(self.obstacle_id)
        return self.obstacle_id
    
    def get_bonus(self,id):
        if(id!=-1):
            if (id in self.bonus_id):
                self.bonus_num+=1
        print('current bonus_num is',self.bonus_num)

if __name__ == "__main__":
    import tqdm, time,os
    
    env = UniTreeEnv(
        obs_type= ["vision","joints","inertial"],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "DELTA_POSITION_CONTROL",
            pb_control_kwargs= dict(force= 20),
            simulate_timestep= 1./500,
        ),
        connection_mode= p.GUI,
        bullet_debug= True,
    )
    obs = env.reset()

    creator=thin_obstacle()
    creator.create_obstacle()

    # with open('scibotpark//unitree//mocap.txt') as f:
    #     for line in f:
    #         line=line2float(line)
    #         obs=env.step(line[2:])
    #         time.sleep(1./500)
    #     f.close()
    
    for _ in tqdm.tqdm(range(10000)):
        # env.debug_step()
        # env.pb_client.stepSimulation()
        # image = env.render(camera_name= "front")
        # sample() is to sample randomly
        obs,prop,visual= env.step(env.action_space.sample())
        # print(obs)
        time.sleep(1./500)