import numpy as np
import pybullet as p
import pybullet_data

class Creator():
    def __init__(self,pb_client):
        self.pb_client=pb_client
    
    def create_box(self,box_size,position,mass,color=[0,0,0,1]):
        #visual model
        visual_shape_id = self.pb_client.createVisualShape(
            shapeType=self.pb_client.GEOM_BOX,
            halfExtents=box_size,
            rgbaColor=color
        )
        
        #collision model
        collison_box_id = self.pb_client.createCollisionShape(
            shapeType=self.pb_client.GEOM_BOX,
            halfExtents=box_size
        )
        
        #muti model
        box_id = self.pb_client.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collison_box_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )

        return box_id

    
    def create_cylinder(self,height,radius,position,mass,color=[1,0,0,1]):
        visual_shape_id=self.pb_client.createVisualShape(
            shapeType=self.pb_client.GEOM_CYLINDER,
            length=height,
            radius=radius,
            rgbaColor=color
        )

        collision_id=self.pb_client.createCollisionShape(
            shapeType=self.pb_client.GEOM_CYLINDER,
            radius=radius,
            length=height
        )

        cylinder_id=self.pb_client.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )

        return cylinder_id
    
    def create_ball(self,radius,position,mass,color=[0.8,0,0,1]):
        visual_shape_id=self.pb_client.createVisualShape(
            shapeType=self.pb_client.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        # collision_id=self.pb_client.createCollisionShape(
        #     shapeType=self.pb_client.GEOM_SPHERE,
        #     radius=radius,
        # )
        ball_id=self.pb_client.createMultiBody(
            baseMass=mass,
            # baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )

        return ball_id
    
    def create_desk(self,mass,position,orientation=[0,0,0,1],color=[0.3,0.5,0.6,1]):
        visual_shape_id=self.pb_client.createVisualShape(
            shapeType=self.pb_client.GEOM_MESH,
            fileName='3dmodels//desk.obj',
            rgbaColor=color
        )
        collision_id=self.pb_client.createCollisionShape(
            shapeType=self.pb_client.GEOM_MESH,
            fileName='3dmodels//desk.obj',
        )

        desk_id=self.pb_client.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=orientation
        )

        return desk_id
    
    def create_chair(self,mass,position,orientation,color=[0.6,0.3,0.2]):
        visual_shape_id=self.pb_client.createVisualShape(
            shapeType=self.pb_client.GEOM_MESH,
            fileName='3dmodels//chair.obj',
            rgbaColor=color,
            meshScale=[0.01,0.01,0.01]
        )
        collision_id=self.pb_client.createCollisionShape(
            shapeType=self.pb_client.GEOM_MESH,
            fileName='3dmodels//chair.obj',
            meshScale=[0.01,0.01,0.01]
        )

        chair_id=self.pb_client.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=orientation
        )

        return chair_id
