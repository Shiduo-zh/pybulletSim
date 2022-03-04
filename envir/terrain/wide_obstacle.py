import pybullet as p
import pybullet_data
import numpy as np
from envir.terrain.terrain import terrain
from time import sleep


class wide_obstacle():
    def __init__(self):
        self.controller=terrain()
    
    def create_obstacle(self):
        
        for i in range(6):
            self.controller.create_box([2,0.5,0.5],[0,5*i,0.5],10000)
            self.controller.create_box([2,0.5,0.5],[10,5*i+3,0.5],10000,[1,0,0,1])
        
        for i in range(8):
            self.controller.create_ball(0.3,[5,3*i,0.3],20)

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
    robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, -10, 0], useMaximalCoordinates=True)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)

    creator=wide_obstacle()
    creator.create_obstacle()
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
        sleep(1 / 240)

    p.disconnect(serve_id)