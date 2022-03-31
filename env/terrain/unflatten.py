import pybullet as p
import pybullet_data as pd
import random
import numpy as np
from env.terrain.obstacle_creator import Creator

class unflatten_terrain():
    def __init__(self,rows,cols,file_img,pb_client):
        #scale of heightmap
        self.heightRows=rows
        self.heightCols=cols
        self.heightData=np.array([0]*self.heightCols*self.heightRows).reshape((self.heightCols,self.heightRows))
        self.terrain=file_img
        self.pb_client=pb_client
        self.controller=Creator(pb_client)
        self.final_id=None

    #form random heightfield
    def random_height(self):
        for i in range(self.heightRows):
            for j in range(self.heightCols):
                self.heightData[i][j]=2*random.random()
        self.heightData=list(self.heightData.reshape((self.heightCols*self.heightRows),1))
    
    #save the proper map  data
    def save(self):
        with open('terrain//heightmaps//heightdata.txt','wb') as f:
            f.write(self.heightData)
        f.close()
    
    def form_heightmap(self):
        terrainShape = self.pb_client.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.1,.1,24],fileName = self.terrain)
        textureId = self.pb_client.loadTexture("..//heightmaps/Textures.png")
        terrain  = self.pb_client.createMultiBody(0, terrainShape)
        self.pb_client.changeVisualShape(terrain, -1, textureUniqueId = textureId)
    
    def set_final(self):
        self.final_id=self.controller.create_ball(2,[10,3,-2],0,[1,0,0,1])
    
    def set_start(self):
        self.controller.create_ball(1,[0,-25,-4],0,[0,1,1,1])
    
if __name__=='__main__':
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
       
        terrain='..//heightmaps//terrain2.png'
        op=unflatten_terrain(1024,1024,file_img=terrain)
        op.form_heightmap()
        op.set_final()
        op.set_start()

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

      
        while True:
            p.stepSimulation()
           
            sleep(1 / 240)

    