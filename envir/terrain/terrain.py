import numpy as np
import pybullet as p
import pybullet_data

class terrain():
    def __init__(self):
        pass
    
    def create_box(self,box_size,position,mass,color=[0,0,1,1]):
        #visual model
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=box_size,
            rgbaColor=color
        )
        
        #collision model
        collison_box_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=box_size
        )
        
        #muti model
        box_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collison_box_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )

        return box_id

    
    def create_cylinder(self,height,radius,position,mass,color=[1,0,0,1]):
        visual_shape_id=p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            length=height,
            radius=radius,
            rgbaColor=color
        )

        collision_id=p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=height
        )

        cylinder_id=p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )

        return cylinder_id
    
    def create_ball(self,radius,position,mass,color=[0,1,0,1]):
        visual_shape_id=p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        # collision_id=p.createCollisionShape(
        #     shapeType=p.GEOM_SPHERE,
        #     radius=radius,
        # )
        ball_id=p.createMultiBody(
            baseMass=mass,
            # baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )

        return ball_id