
import pybullet as p
import pybullet_data
from time import sleep
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('../../')

from terrain.thin_obstacle import thin_obstacle


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

op=thin_obstacle()
op.test()

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)

for i in range(p.getNumJoints(robot_id)):
    if "wheel" in p.getJointInfo(robot_id, i)[1].decode("utf-8"):   
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
                op.add_bonus(ID)
    sleep(1 / 240)

p.disconnect(serve_id)