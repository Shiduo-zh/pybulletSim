from matplotlib.pyplot import axis
from matplotlib.transforms import Transform
import numpy as np
import math
from scipy.linalg import *
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client
import torch

import gym
from gym import spaces, Env

from scibotpark.unitree.unitree_robot import UniTreeRobot
from scibotpark.pybullet.base import PybulletEnv
from env.terrain.thin_obstacle import *
from env.terrain.wide_obstacle import *
from env.terrain.moving_obstacle import *
from env.terrain.office import *
from env.terrain.unflatten import *
from env.model.mlp import *
from env.model.conv2d import *
from env.model.transformer import *


class UniTreeEnv(PybulletEnv,Env):
    def __init__(self, 
        obs_type=["vision"],
        render_kwargs=dict(),
        robot_kwargs=dict(),
        connection_mode=p.DIRECT, 
        bullet_debug=False,
        surrounding_tpye='thin_obstacle',
        **kwargs,
        ):

        #mode configure,may include visual-info,joint-info or IMU-info.
        self.obs_type=obs_type
        
        #render args init and update
        self.render_kwargs=dict(
            resolution=(64,64),
            camera_name="front",
            modal="depth",
        )
        self.render_kwargs.update(render_kwargs)

        self.visual_resolution=self.render_kwargs["resolution"]
        #agent args init and update
        self.robot_kwargs=dict(
            robot_type="a1",
            default_base_transform=np.array([0,0,0.42,0,0,1,1]),
            pb_control_mode="POSITION_CONTROL",
            pb_control_kwargs=dict(force=20),
        )
        self.robot_kwargs.update(robot_kwargs)
        self.bullet_debug = bullet_debug
        self.surrounding_type=surrounding_tpye

        #last up to 3 times prop take by agent
        self.last_obs=np.zeros(0)

        #last angular observation 
        self.last_angle=np.zeros(12)
        
        #angular change
        self.angular_delta=np.zeros(12)
               
        #last position for moving reward
        self.last_pos=self.robot_kwargs['default_base_transform'][0:3]

        #alive time for alive reward
        self.step_num=0
        
        self.bonus_num=0
   
        self.images=np.zeros(4*64*64).reshape((4,64*64))
        #connection to engine
        pb_client = bullet_client.BulletClient(connection_mode= connection_mode, options= "")

        if("thin" in self.surrounding_type):
            self.surrounding=ThinObstacle(pb_client)
        elif("wide" in self.surrounding_type):
            self.surrounding=WideObstacle(pb_client)
        elif('moving' in self.surrounding_type):
            self.surrounding=MovingObstacle(pb_client)
        elif('office' in self.surrounding_type):
            self.surrounding=office(pb_client)
        elif('mountain' in self.surrounding_type):
            self.surrounding=unflatten_terrain(pb_client)

        super().__init__(pb_client= pb_client, **kwargs)

    def _build_robot(self):
        self._robot=UniTreeRobot(
            bullet_debug=self.bullet_debug,
            pb_client=self.pb_client,
            **self.robot_kwargs,
        )
    
    def _build_surroundings(self):
        self.surrounding.init_env()
        self.surrounding.create_obstacle()
        return super()._build_surroundings()
    
    # def load_obstacle(self):
    #     self.operator.init_env()
    #     self.operator.create_obstacle()

    @property
    def action_space(self):
        "Box(low,high,dtype)"
        limits = self.robot.get_cmd_limits()
        return spaces.Box(
            limits[0],
            limits[1],
            dtype= np.float32,
        )

  
    @property
    def observation_space(self):
        obs_space = dict()
        obs_space['prop']=spaces.Box(
            -np.inf,
            np.inf,
            shape=(93,),
            dtype=np.float32
        )
        if "vision" in self.obs_type:
            obs_space["vision"] = spaces.Box(
                0, 1,
                shape= (*self.visual_resolution,),
                dtype= np.float32,
            )
        if "joints" in self.obs_type:
            joint_limits = self.robot.get_joint_limits("position")
            obs_space["joints"] = spaces.Box(
                joint_limits[0],
                joint_limits[1],
                dtype= np.float32,
            )
        if "inertial" in self.obs_type:
            obs_space["inertial"] = spaces.Box(
                -np.inf,
                np.inf,
                shape= (12,), # linear / angular, velocity / position
                dtype= np.float32,
            )
        if len(self.obs_type) == 1:
            return obs_space[self.obs_type[0]]
        else:
            return spaces.Dict(obs_space)

    #Joint angle mensioned in paper is obs["joints"], which is a 12-dimensional vec
    #IMU information is inertial_data["rotation"]+inertial_data["angular_velocity"], also can be written as obs["intertial"][6:10]
    #Base displacement is inertial_data["position"],also as obs["inertial"][0:3]
    def _get_obs(self):
        "get visual information by fuc render"
        "get inner joints observations including joint position, linear velocity,rotation and angular velocity"
        obs = dict()
        if "vision" in self.obs_type:
            obs["vision"] = self.render(**self.render_kwargs)
        if "joints" in self.obs_type:
            obs["joints"] = self.robot.get_joint_states("position")
            self.last_angle=obs["joints"]#record joints angular this time as last obs
        
        #IMU infomation: return a 4-dimensional vector records orientations and angular velocities
        if "inertial" in self.obs_type:
            inertial_data = self._robot.get_inertial_data()
            obs["inertial"] = np.concatenate([
                #3-dimension
                inertial_data["position"],
                #3
                inertial_data["linear_velocity"],
                #3
                inertial_data["rotation"],
                #3
                inertial_data["angular_velocity"],
            ])
        if len(self.obs_type) == 1:
            return obs[self.obs_type[0]]
        else:
            return obs
    
    #PS：to get last_obs, before calling this func, add last_obs=self.last_obs and treat it as args
    def angular_change(self,last_obs):
        obs=self._get_obs()
        return obs["joints"]-last_obs
    
    #93-dimension proprioceptive state
    def prop_state(self):
        last_angle=self.last_angle
        obs=self._get_obs()
        self.angular_delta=self.angular_change(last_angle)
        current_prop=np.concatenate([
            obs["joints"],
            obs["inertial"][6:10],
            obs["inertial"][0:3],
            self.angular_delta,
        ])
        if(len(self.last_obs)<62):
            if(len(self.last_obs)==0):
                prop=np.concatenate([np.zeros(62),current_prop])
            elif(len(self.last_obs)==31):
                prop=np.concatenate([np.zeros(31),self.last_obs,current_prop])
            self.last_obs=np.append(self.last_obs,current_prop,axis=0)
            

        elif(len(self.last_obs)==62):
            prop=np.append(self.last_obs.reshape(1,62),current_prop)
            #update last_obs
            self.last_obs=np.append(self.last_obs,current_prop,axis=0)
            self.last_obs=self.last_obs[31:]
            #print(self.last_obs.shape())
        
        return prop
        
       
    def terminal(self):
        # for id in self.robot.valid_joint_ids:
        obs=self._get_obs()
        pos=obs["inertial"][2]
        feet_id=[5,9,13,17]
        for id in self.robot.thigh_joints_id:
            info=self.pb_client.getContactPoints(bodyA=self.robot.body_id,linkIndexA=id)
            if(len(info)>0):
                posA=info[0][5]
                if(posA[2]<0.01):
                    return True
        #base body info
        info=self.pb_client.getContactPoints(bodyA=self.robot.body_id,linkIndexA=1)
        if(len(info)>0):
            posA=info[0][5]
            if(posA[2]<0.01):
                return True
        
        # for foot in feet_id:
        #     foot_state=p.getLinkState(self.robot.body_id,foot)
        #     if(foot_state[0][2]>pos):
        #         return True
            # if(leg_state[2]>pos):
            #     return True
        
        return False

    
    def compute_reward(self,delta_time,bonus_flag,next_obs,energy_coef=0.005,forward_coef=1,alive_coef=0.1,start_pos=[-1,-1,-1],end_pos=[-1,-1,-1]):
        """
        compute total reward in one step
           params:delta_time:time span taken by a step
                  bonus_flag:whether get a bonus in this step
                  **_coef:coefficient for all kinds of reward
           return: immediate reward in this step 
        """
        #forward reward
        current_pos=self._get_obs()['inertial'][0:3]
        current_vel=self._get_obs()['inertial'][3:6]
        if('mountain' in self.surrounding_type):
           #in unflatten terrain,reward is the velocity/distance along the direction forward to the goal
            vec_dir=end_pos-start_pos
            vec_cur=current_pos-self.last_pos
            R_forward=(np.multiply(vec_dir*vec_cur)\
                    /np.linalg.norm(vec_dir,ord=2))/delta_time
                    #1/500 is delta time
        else:
            #the velocity/distance along y-aixs
            #R_forward=(current_pos[1]-self.last_pos[1])/delta_time
            R_forward=current_vel[1]
        #energy reward
        torque=self.robot.get_joint_states("torque").reshape((12,1))
        R_energy=-np.linalg.norm(torque, ord=None, axis=None, keepdims=False)
        #alive reward(1 or self.step_num?)
        R_alive=1
        #update last_pos
        self.last_pos=current_pos
        return forward_coef*R_forward+alive_coef*R_alive+energy_coef*R_energy+bonus_flag
        
    #return conllision obstacle id. if obstacle is bonus, bonus reward will increase. other obs will lead to punishment
    def collision_dect(self):
        p_min,p_max=self.pb_client.getAABB(self.robot.body_id)
        id_tuple = self.pb_client.getOverlappingObjects(p_min, p_max)
        if(len(id_tuple)>1):
            for id in id_tuple:
                if id==self.robot.body_id:
                    continue
                else:
                    collision_id=id
                    return collision_id
        else:
            return -1



    def reset(self):
        self.surrounding.reload_obstacle()
        basePos=self.robot_kwargs['default_base_transform'][0:3]
        baseOri=(0.0, 0.0, 0.7071067811865475, 0.7071067811865476)
        self.pb_client.resetBasePositionAndOrientation(self.robot.body_id,basePos,baseOri)
        self.robot.reset_joint_states()
        # self.operator.reload_obstacle()
        self.step_num=0
        self.last_obs=np.zeros(0)
        self.last_angle=np.zeros(12)
        self.angular_delta=np.zeros(12)
        self.last_pos=self.robot_kwargs['default_base_transform'][0:3]
        self.step_num=0
        self.bonus_num=0
   
        self.images=np.zeros(4*64*64).reshape((4,64*64))
        obs_prop=self.prop_state()
        obs_all=self._get_obs()
        vision=obs_all['vision'].reshape((*self.visual_resolution,))
        self.update_image(vision)
        obs_vision=self.images.reshape(4,*self.visual_resolution)

        return obs_prop,obs_vision
    
    def update_image(self,new_image):
        new_image=new_image.reshape((1,64*64))
        self.images=np.concatenate((self.images,new_image),axis=0)[1:5]

    
    def step(self, action):
        pos=self._get_obs()['inertial'][0:3]
        action =np.clip(action, self.action_space.low, self.action_space.high)
        self.step_simulation_from_action(action)  
        prop=self.prop_state()
        #visual=self.render()
        collision_id=self.collision_dect()
        #bonus_flag=self.operator.add_bonus(collision_id)
        bonus_flag=self.surrounding.add_bonus(collision_id)
        self.step_num+=1
        self.bonus_num+=bonus_flag
        obs_all = self._get_obs()
        
        reward = self.compute_reward(1./500,bonus_flag,obs_all)
        done = self.terminal()#judge by whether the thigh links collision with ground
        
        infos={}
        infos['vision']=obs_all['vision'].reshape((*self.visual_resolution,))
        infos['episode_steps']=self.step_num
        self.update_image(infos['vision'])
        
        obs={}
        obs['prop']=torch.from_numpy(prop)
        obs['vision']=torch.from_numpy(self.images.reshape((4,*self.visual_resolution)))
        
        return obs,reward,done,infos

        
    def seed(self, seed=None):
        pass

    def render(self, **render_kwargs):
        if "mode" in render_kwargs and render_kwargs["mode"] == "rgb_array":
            kwargs = self.render_kwargs.copy()
            kwargs["resolution"] = (render_kwargs["width"], render_kwargs["height"])
            return self._robot.get_onboard_camera_image(**kwargs)
        else:
            return self._robot.get_onboard_camera_image(**render_kwargs)
   
    def snapshot(self):
        cameraPos=self.last_pos+np.array([0,-3,1])
        targetPos=cameraPos+np.array([0,1,-0.4])
        vecUpPos=np.array([0,0.4,1])
        viewMatrix=self.pb_client.computeViewMatrix(cameraPos,targetPos,vecUpPos)
        projectionMatrix = self.pb_client.computeProjectionMatrixFOV(fov=50.0,               
                                                        aspect=1.0,
                                                        nearVal=0.01,            
                                                        farVal=20)
        width, height, rgbImg, depthImg, segImg=self.pb_client.getCameraImage(480,480,viewMatrix=viewMatrix,projectionMatrix=projectionMatrix)
        return width,height,rgbImg

    def close(self):
        if self._physics_client_id >= 0:
            self.pb_client.disconnect()
        self._physics_client_id = -1


if __name__ == "__main__":
    import tqdm, time

    env = UniTreeEnv(
        obs_type= ["vision",'joints','inertial'],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "DELTA_POSITION_CONTROL",
            pb_control_kwargs= dict(force= 20),
            simulate_timestep= 1./500,
        ),
        connection_mode= p.GUI,
        env="thin_obstacle",
        bullet_debug= True,
    )
    
    env.load_obstacle()
    obs = env.reset()

    for i in tqdm.tqdm(range(10000)):
        # env.debug_step()
        # env.pb_client.stepSimulation()
        # image = env.render(camera_name= "front")
        obs,reward,done,info = env.step(env.action_space.sample())
        #time.sleep(1./500)
        if((i+1)%50==0):
            env.reset()
    
    
