import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import random
from env.terrain.obstacle import Obstacle
from env.terrain.obstacle_creator import Creator
from time import sleep

class MovingObstacle(Obstacle):
    def __init__(self,pb_client):
        self.controller=Creator(pb_client)
        #obstacles move in x aixs
        self.obstacle_id_x=np.array([])
        #obstacles move in y aixs
        self.obstacle_id_y=np.array([])

        self.basic_xs=np.array([])
        self.basic_ys=np.array([])
        super().__init__(pb_client)        

        
    def change_position_x(self,id,low_x,high_x,x=0,direction='left'):
        current_state=self.pb_client.getBasePositionAndOrientation(int(id))
        pos=current_state[0]
        orn=current_state[1]
        dir=direction
        if(direction=='right'):
            if(pos[0]>=high_x):
                dir='left'
                newpos=[pos[0]-x,pos[1],pos[2]]
            else:
                newpos=[pos[0]+x,pos[1],pos[2]]

        elif(direction=='left'):
            if(pos[0]<=low_x):
                dir='right'
                newpos=[pos[0]+x,pos[1],pos[2]]
            else:
                newpos=[pos[0]-x,pos[1],pos[2]]
        
        p.resetBasePositionAndOrientation(int(id),newpos,orn)
        return dir

    def change_position_y(self,id,low_y,high_y,y=0,direction='forward'):
        current_state=self.pb_client.getBasePositionAndOrientation(int(id))
        pos=current_state[0]
        orn=current_state[1]
        if(direction=='forward'):
            if(pos[1]>=high_y):
                direction='backward'
                newpos=[pos[0],pos[1]-y,pos[2]]
            else:
                newpos=[pos[0],pos[1]+y,pos[2]]

        elif(direction=='backward'):
            if(pos[1]<=low_y):
                direction='forward'
                newpos=[pos[0],pos[1]+y,pos[2]]
            else:
                newpos=[pos[0],pos[1]-y,pos[2]]
        
        self.pb_client.resetBasePositionAndOrientation(int(id),newpos,orn)
        return direction

    def create_obstacle(self):
        shape=[0.5,0.5,2]
        #init obstacle moving in x-aixs
        for j in range(3):
            for i in range(7):
                basic_x=7*(j-1)+0.1*random.randrange(-20,20,1)
                self.basic_xs=np.append(self.basic_xs,basic_x)
                self.obstacle_id_x=np.append(self.obstacle_id_x,self.controller.create_box( shape,
                                                                                            [basic_x,8*(i+1)+0.1*random.randrange(-20,20,1),2],
                                                                                            10000))
        for j in range(5):
        #init obstacles moving in y-aixs
            for i in range(3):
                basic_y=4+8*j+0.1*random.randrange(-20,20,1)
                self.basic_ys=np.append(self.basic_ys,basic_y)
                self.obstacle_id_y=np.append(self.obstacle_id_y,self.controller.create_box( shape,
                                                                                            [6*(i-1)+0.1*random.randrange(-20,20,1),basic_y,2],
                                                                                            10000))
        

if __name__=='__main__':
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(1)
    p.setAdditionalSearchPath(pd.getDataPath())
    _ = p.loadURDF("plane.urdf", useMaximalCoordinates=True)

    operator=moving_obstacle()
    operator.init_env()
    operator.create_obstacle()
    
    #dict initial
    dir_x=dict()
    dir_y=dict()
    for id in operator.obstacle_id_x:
        newdir=random.randint(0,1)
        if(newdir==1):
            newdir='left'
        else:
            newdir='right'
        dir_x[str(id)]=newdir
    for id in operator.obstacle_id_y:
        newdir=random.randint(0,1)
        if(newdir==1):
            newdir='forward'
        else:
            newdir='backward'
        dir_y[str(id)]=newdir
    #obstacle moving
    while True:
        for (id,basic_x) in zip(operator.obstacle_id_x,operator.basic_xs):
            dirx=operator.change_position_x(id,basic_x-1,basic_x+1,0.1,dir_x[str(id)])
            dir_x[str(id)]=dirx
        for (id,basic_y) in zip(operator.obstacle_id_y,operator.basic_ys):
            diry=operator.change_position_y(id,basic_y-1,basic_y+1,0.1,dir_y[str(id)])
            dir_y[str(id)]=diry
        time.sleep(1/20)
     
