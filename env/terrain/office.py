#from envir.envs import UniTreeEnv
import pybullet as p
import pybullet_data as pd
import numpy as np
from env.terrain.obstacle_creator import Creator
import random

class office():
    def __init__(self,pb_client,mass=2,orientation=[0,0,0,0]):
        self.controller=Creator(pb_client)
        self.mass=mass
        self.orientation=orientation
        self.desk_id=np.array([])
        self.chair_id=np.array([])
        self.pb_client=pb_client
    
    def init_walls(self):
        for i in range(9):
            self.controller.create_box([0.2,5,1],[-12,i*10,1],10000,[1,1,1,1])
            self.controller.create_box([0.2,5,1],[12,i*10,1],10000,[1,1,1,1])
        for i in range(12):
            self.controller.create_box([1,0.2,1],[2*i-11,85.2,1],10000,[1,1,1,1])

    def init_desks(self):
        for i in range(8):
            self.desk_id=np.append(self.desk_id,self.controller.create_desk(self.mass,[-5+0.1*random.randrange(-20,20,1),6*(i+1)+0.1*random.randrange(-10,10,1),0.4],self.orientation))
        for i in range(8):
            self.desk_id=np.append(self.desk_id,self.controller.create_desk(self.mass,[0.1*random.randrange(-20,20,1),6*(i+1)+0.1*random.randrange(-10,10,1),0.4],self.orientation))
        for i in range(8):
            self.desk_id=np.append(self.desk_id,self.controller.create_desk(self.mass,[5+0.1*random.randrange(-20,20,1),6*(i+1)+0.1*random.randrange(-10,10,1),0.4],self.orientation))

    def get_desks_id(self):
        return self.desk_id
    
    def init_chairs(self):
        for i in range(6):
            self.chair_id=np.append(self.chair_id,self.controller.create_chair(self.mass,[-3+0.1*random.randrange(-30,10,1),3+6*(i+1)+0.1*random.randrange(-10,10,1),0.2],[0,0,0,1],[0.6,0.4,0,1]))
        for i in range(6): 
            self.chair_id=np.append(self.chair_id,self.controller.create_chair(self.mass,[0+0.1*random.randrange(-10,10,1),3+6*(i+1)+0.1*random.randrange(-10,10,1),0.2],[0,0,0,1],[0.6,0.4,0,1]))
        for i in range(6):    
            self.chair_id=np.append(self.chair_id,self.controller.create_chair(self.mass,[3+0.1*random.randrange(-10,30,1),3+6*(i+1)+0.1*random.randrange(-10,10,1),0.2],[0,0,0,1],[0.6,0.4,0,1]))
           
        

