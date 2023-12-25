import time

from robot import Robot
import img
from predict import GraspPredict

import robot_utils as utils
import pybullet as p
import pybullet_data as pd
import numpy as np
import math
import cv2
import torch

# const
flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
model_path = "model/plan2/epoch_92_iou_0.72_statedict.pt"
# GUI
cid = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0, 0, -9.8)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraPitch=-20, cameraYaw=90, cameraTargetPosition=[0, 0, 0])

# time
timeStep = 1./60.
p.setTimeStep(timeStep)

# load robot and objects
p.setAdditionalSearchPath(pd.getDataPath())
panda = Robot(p)

# p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1], flags=flags)
p.loadURDF("table\\table.urdf", [0.8, 0, -0.625], [0, 0, 0, 1], flags=flags)
obj = []
# p.loadURDF("sphere_small.urdf", np.array([0.5, -0.2, 0.1]), flags=flags)
# p.loadURDF("lego\\lego.urdf", np.array([0.5, 0.0, 0.1]), flags=flags)
# p.loadURDF("jenga/jenga.urdf", np.array([0.5, 0.0, 0.2]),
#            baseOrientation=p.getQuaternionFromEuler([0, 0, -math.pi/4]), flags=flags)
p.loadURDF("duck_vhacd.urdf", np.array([0.5, 0.0, 0.1]),
           baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, 0]), flags=flags)
# p.loadURDF("cube_small.urdf", np.array([0.5, 0.0, 0.1]), flags=flags)
# load GGCNN
predict = GraspPredict(model_path)

t = 0
ENABLE_CAM = 1
GRASP = 0
RESET = 0
stand_by = 0
READY = 0
grab = 0
i = 0
# p.enableJointForceTorqueSensor(panda.panda, 10, 1)
while True:
    # get images
    if ENABLE_CAM:
        pics = panda.setCameraPicAndGetPic(300, 300, cid)
        # from RGBA to RGB
        rgb_img = pics[2]
        rgb_img = img.RGBA_RGB(rgb_img)
        # depth image
        depth = pics[3]

        # grasp predict
        with torch.no_grad():
            predict.depth_process(depth)
            q_img, ang_img, width_img = predict.net_output()
            q_max, angle, width, [r, c] = predict.grasp(q_img, ang_img, width_img)

        [x, y] = [c, r]
        # show grasp in rgb_img
        rgb_img = img.graspRGB(rgb_img, angle, width, x, y)
        # show grasp center in q_img
        q_img = img.graspQ(q_img, x, y)
        # angle image
        ang_img = img.graspAng(ang_img * 180 / math.pi)
        # width image
        width_img = img.graspW(width_img)
        # depth_img
        # depth_img = img.depth_grey(depth)

        stack1 = np.hstack([rgb_img, q_img])
        stack2 = np.hstack([ang_img, width_img])
        grasp_img = np.vstack([stack1, stack2])
        cv2.imshow(model_path, grasp_img)
        # cv2.imshow("depth", depth_img)
        cv2.waitKey(1)

    if GRASP:
        if not stand_by:
            grasp_pos, width_l = utils.frame_transform([r, c], depth, width)
            finger = width_l / 2 / 3 * (0.04 / 0.04)
            if finger > 0.04:
                finger = 0.04
            stand_by = panda.step(grasp_pos + np.array([0, 0, 0.1]), angle)
        elif READY:
            grab = panda.step(grasp_pos + np.array([0, 0, -0.]))
            i = i + grab
        if i > 50:
            READY = 0
            panda.gripper(finger)
            i = i + 1
        if i > 100:
            grab = 0
            panda.step([0.4, 0, 0.5], lift=1)

    if RESET:
        panda.step([0.4, 0.0, 0.5], 0, lift=1)
        panda.gripper(0.04)

    keys = p.getKeyboardEvents()
    if len(keys) > 0:
        for k, v in keys.items():
            if v & p.KEY_WAS_TRIGGERED:
                if k == ord('1'):
                    ENABLE_CAM = 0
                    GRASP = 1
                    RESET = 0
                    print("Cam disabled, start grasping")
                if k == ord('2'):
                    ENABLE_CAM = 0
                    GRASP = 0
                    RESET = 1
                    READY = 0
                    stand_by = 0
                    grab = 0
                    i = 0
                    print("Robot reset")
                if k == ord('3'):
                    ENABLE_CAM = 1
                    GRASP = 0
                    RESET = 0
                    print("Cam start")
                if k == ord('4'):
                    READY = 1
                    print("grab")
            if v & p.KEY_WAS_RELEASED:
                pass

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    t += timeStep

    time.sleep(timeStep)
    p.stepSimulation()


