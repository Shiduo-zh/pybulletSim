import numpy as np
import math
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client
import torch

import gym
from gym import spaces, Env

from scibotpark.unitree.unitree_robot import UniTreeRobot
from scibotpark.pybullet.base import PybulletEnv

from Utils.camera import *

class UniTreeEnv(PybulletEnv,Env):
    def __init__(self, 
        obs_type=["vision"],
        render_kwargs=dict(),
        robot_kwargs=dict(),
        connection_mode=p.DIRECT, 
        bullet_debug=False,
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
            default_base_transform=np.array([0,0,0.45,0,0,1,1]),
            pb_control_mode="POSITION_CONTROL",
            pb_control_kwargs=dict(force=20),
        )
        self.robot_kwargs.update(robot_kwargs)
        self.bullet_debug = bullet_debug

        #last up to 3 times prop take by agent
        self.last_obs=np.zeros(31)

        #last angular observation 
        self.last_angle=np.zeros(12)
        
        #angular change
        self.angular_delta=np.zeros(12)

        #last two times action taken by agent
        self.last_action=np.array([])
               
        #last position for moving reward
        self.last_pos=np.array([])

        #bonus num for bonus reward
        self.bonus_num=0

        #alive time for alive reward
        self.step_num=0

        #connection to engine
        pb_client = bullet_client.BulletClient(connection_mode= connection_mode, options= "")
        super().__init__(pb_client= pb_client, **kwargs)

    def _build_robot(self):
        self._robot=UniTreeRobot(
            bullet_debug=self.bullet_debug,
            pb_client=self.pb_client,
            **self.robot_kwargs,
        )

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
        if "vision" in self.obs_type:
            obs_space["vision"] = spaces.Box(
                0, 1,
                shape= (*self.visual_resolution, 3),
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
                #1
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
            self.last_obs=np.append(self.last_obs,current_prop,axis=0)
            if(len(self.last_obs)==31):
                prop=np.concatenate([np.zeros(31),self.last_obs,current_prop])
            elif(len(self.last_obs)==62):
                prop=np.concatenate([self.last_obs,current_prop])
            

        elif(len(self.last_obs)==62):
            prop=np.append(self.last_obs.reshape(1,62),current_prop)
            #update last_obs
            self.last_obs=np.append(self.last_obs,current_prop,axis=0)
            self.last_obs=self.last_obs[31:]
            #print(self.last_obs.shape())
        
        return prop
        
       
            

    def compute_reward(self,next_obs,**kwargs):
        pass
        #energy reward
        # a2=self.alpha_energy
        # torque=self.robot.get_joint_states("torque")
        # R_energy=-math.sqrt(np.dot(torque,torque.T))
        
    #return conllision obstacle id. if obstacle is bonus, bonus reward will increase. other obs will lead to punishment
    def collision_dect(self):
        p_min,p_max=p.getAABB(self.robot.body_id)
        id_tuple = p.getOverlappingObjects(p_min, p_max)
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
        pass
    
    #todo:further override
    def step(self, action):
        # pass
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.step_simulation_from_action(action)
        
        prop=self.prop_state()

        visual=self.render()
        
        obs = self._get_obs()
        # reward = self.compute_reward(obs)
        # done = False
        # info = {}
        collision_id=self.collision_dect()
        return obs, prop,visual
        #return obs, reward, done, info, collision_id
        #here is action update
        #here is agent-state fetch
        #here is reward computation
        
    def seed(self, seed=None):
        pass

    def render(self, **render_kwargs):
        if "mode" in render_kwargs and render_kwargs["mode"] == "rgb_array":
            kwargs = self.render_kwargs.copy()
            kwargs["resolution"] = (render_kwargs["width"], render_kwargs["height"])
            return self._robot.get_onboard_camera_image(**kwargs)
        else:
            return self._robot.get_onboard_camera_image(**render_kwargs)
   
    #get depth image information,may be deleted aftersoon
    def getDepImg(self):
        "get depth image information from local agent"
        "returns: a dict including width,height,viewMatrix,projectionMatrix"
        image_data=self._robot.get_onboard_camera_image(camera_name="front",modal="depth")
        return image_data
    
    #about to override
    def getLinearProp(self,out_feature:int=128,in_feature:int=93):
        "get linear output of prop"
        "returns: C-dimension prop"
        position=self.robot.get_joint_states("position")
        velocity=self.robot.get_joint_states("velocity")
        torque=self.robot.get_joint_states("torque")
        prop=position
        linear_model=torch.nn.Linear(in_feature,out_feature)
        t_prop=linear_model(prop)
        return t_prop

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1
    
    
