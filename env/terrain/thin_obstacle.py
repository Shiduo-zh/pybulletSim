from re import A
import pybullet as p
import pybullet_data
import numpy as np
import random
from gym import spaces, Env
from env.terrain.utils import line2float

from env.terrain.obstacle import Obstacle
from env.terrain.obstacle_creator import Creator


from time import sleep

class ThinObstacle(Obstacle):
    def __init__(self,pb_client):
        self.controller=Creator(pb_client)
        self.obstacle_id=np.array([])
        self.bonus_id=np.array([])
        self.bonus_num=0
        super().__init__(pb_client)

    def create_obstacle(self):
        
        for i in range(15):
            self.obstacle_id=np.append(self.obstacle_id,self.controller.create_box([0.5,0.5,2],[-8+0.1*random.randrange(-25,25,5),5*(i+1)+0.1*random.randrange(-20,20,2),2],10000))
            self.obstacle_id=np.append(self.obstacle_id,self.controller.create_box([0.5,0.5,2],[-3+0.1*random.randrange(-25,25,5),5*(i+1)+0.1*random.randrange(-20,20,2),2],10000))
            self.obstacle_id=np.append(self.obstacle_id,self.controller.create_box([0.5,0.5,2],[2+0.1*random.randrange(-25,25,5),5*(i+1)+0.1*random.randrange(-20,20,2),2],10000))
            self.obstacle_id=np.append(self.obstacle_id,self.controller.create_box([0.5,0.5,2],[7+0.1*random.randrange(-25,25,5),5*(i+1)+0.1*random.randrange(-20,20,2),2],10000))

        
        for i in range(13):
            self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[0.1*random.randrange(-70,70,2),5*(i+1)+0.1*random.randrange(20,25,1),0.3],0))
            self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[0.1*random.randrange(-70,70,2),5*(i+1)+0.1*random.randrange(25,30,1),0.3],0))
    
    def get_obstacle_id(self):
        print(self.obstacle_id)
        return self.obstacle_id
    
    def add_bonus(self,id):
        if(id!=-1):
            if (id in self.bonus_id):
                self.bonus_num+=1
                p.removeBody(id)
                print('get bonus!current bonus num is',self.bonus_num)
                return 1
            else:
                #print('no bonus!current bonus num is',self.bonus_num)
                return 0
        
        else:
            return 0
    
    def test(self):
         self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[0,1,0.3],0))
         self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[0,2,0.3],0))
         self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[0,3,0.3],0))


