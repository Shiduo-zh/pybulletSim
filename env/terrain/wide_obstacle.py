from calendar import c
import pybullet as p
import pybullet_data
import numpy as np
import random
from env.terrain.obstacle_creator import Creator
from env.terrain.obstacle import Obstacle
from time import sleep


class WideObstacle(Obstacle):
    def __init__(self,pb_client):
        self.controller=Creator()
        self.obstacle_id=np.array([])
        self.bonus_id=np.array([])
        self.bonus_num=0
        super().__init__(pb_client)

    def create_obstacle(self):
        
        for i in range(12):
            self.obstacle_id=np.append(self.obstacle_id,self.controller.create_box([2,0.5,0.5],[-7+0.1*random.randrange(-20,20,1),6*(i+1)+0.1*random.randrange(-20,20,1),0.5],10000))
            self.obstacle_id=np.append(self.obstacle_id,self.controller.create_box([2,0.5,0.5],[-1+0.1*random.randrange(-20,20,1),6*(i+1)+0.1*random.randrange(-20,20,1),0.5],10000))
            self.obstacle_id=np.append(self.obstacle_id,self.controller.create_box([2,0.5,0.5],[6+0.1*random.randrange(-20,20,1),6*(i+1)+0.1*random.randrange(-20,20,1),0.5],10000))
        
        for i in range(13):
            self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[0.1*random.randrange(-70,70,2),5*(i+1)+0.1*random.randrange(20,25,1),0.3],0))
            self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[0.1*random.randrange(-70,70,2),5*(i+1)+0.1*random.randrange(25,30,1),0.3],0))



if __name__ == "__main__":
    use_gui = True
    if use_gui:
        serve_id = p.connect(p.GUI)
    else:
        serve_id = p.connect(p.DIRECT)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
   
    _ = p.loadURDF("plane.urdf", useMaximalCoordinates=True)
    robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.5], useMaximalCoordinates=True)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)

    operator=WideObstacle()
    operator.init_env()
    operator.create_obstacle()

    for i in range(p.getNumJoints(robot_id)):
        if "wheel" in p.getJointInfo(robot_id, i)[1].decode("utf-8"):           # 如果是轮子的关节，则为马达配置参数，否则禁用马达
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=30,
                force=100
            )
        else:
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )


    while True:
        p.stepSimulation()
        P_min, P_max = p.getAABB(robot_id)
        id_tuple = p.getOverlappingObjects(P_min, P_max)
        if len(id_tuple) > 1:
            for ID, _ in id_tuple:
                if ID == robot_id:
                    continue
                else:
                    print(f"hit happen! hit object is {p.getBodyInfo(ID)}")
                    operator.add_bonus(ID)
                    continue
        sleep(1 / 240)

    p.disconnect(serve_id)