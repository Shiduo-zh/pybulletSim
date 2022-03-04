import numpy as np
import pybullet as p
import pybullet_data


def setCameraPicAndGetPic(robot_id : int, width : int = 224, height : int = 224, physicsClientId : int = 0):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
    tx_vec = np.array([matrix[0], matrix[3], matrix[6]])              # 变换后的x轴
    tz_vec = np.array([matrix[2], matrix[5], matrix[8]])              # 变换后的z轴

    basePos = np.array(basePos)
    # 摄像头的位置
    cameraPos = basePos + 1 * tx_vec + 0.5 * 1 * tz_vec
    targetPos = cameraPos + 1 * tx_vec

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
        physicsClientId=physicsClientId
    )

    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=50.0,               # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.01,            # 摄像头焦距下限
        farVal=20,               # 摄像头能看上限
        physicsClientId=physicsClientId
    )

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId
    )
    
    return width, height, rgbImg, depthImg, segImg