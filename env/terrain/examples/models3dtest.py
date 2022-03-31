import pybullet as p
import pybullet_data as pd
import numpy as np
from time import sleep

use_gui = True
if use_gui:
    serve_id = p.connect(p.GUI)
else:
    serve_id = p.connect(p.DIRECT)

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

p.setAdditionalSearchPath(pd.getDataPath())
_ = p.loadURDF("plane.urdf", useMaximalCoordinates=True)



p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)

visual_shape_id=p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName='..//3dmodels//room.obj',
            rgbaColor=[1,1,1,1,],
            meshScale=[0.002,0.002,0.002]
        )
collision_id=p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName='..//3dmodels//room.obj',
            meshScale=[0.002,0.002,0.002]
        )
chair_id=p.createMultiBody(
            baseMass=10,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0,0,0],
            baseOrientation=[1,0,0,1]
        )

while True:
    p.stepSimulation()
    sleep(1 / 240)