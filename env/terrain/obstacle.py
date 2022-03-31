import numpy as np
import pybullet as p
import pybullet_data
from env.terrain.obstacle_creator import Creator

class Obstacle():
    def __init__(self,pb_client):
        self.controller=Creator(pb_client)
        self.obstacle_id=np.array([])
        self.bonus_id=np.array([])
        self.bonus_num=0
        self.pb_client=pb_client
    
    def init_env(self):
        for i in range(9):
            self.controller.create_box([0.2,5,2],[-12,i*10,2],10000)
            self.controller.create_box([0.2,5,2],[12,i*10,2],10000)
        for i in range(12):
            self.controller.create_box([1,0.2,2],[2*i-11,85.2,2],10000,[0,0,0,1])
    
    def create_obstacle(self):
        pass

    def reload_obstacle(self):
        for id in self.obstacle_id:
            self.pb_client.removeBody(int(id))
        for id in self.bonus_id:
            self.pb_client.removeBody(int(id))
        self.obstacle_id=np.array([])
        self.bonus_id=np.array([])
        self.create_obstacle()

    def add_bonus(self,id):
        if(id!=-1):
            if (id in self.bonus_id):
                self.bonus_num+=1
                self.pb_client.removeBody(id)
                #print('current bonus_num is',self.bonus_num)
                np.delete(self.bonus_id,np.argwhere(self.bonus_id==id),aixs=0)
                return 1 
            else:
                #collision with obstacless
                return 0

    def get_obstacle_id(self):
        print(self.obstacle_id)
        return self.obstacle_id
    
    def get_bonus_num(self):
        return self.bonus_num
