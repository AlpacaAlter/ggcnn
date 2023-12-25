import math
import numpy as np


pandaNumDofs = 7
pandaEndEffectorIndex = 11
ll = [-7] * pandaNumDofs
# upper limits for null space (todo: set them to proper range)
ul = [7] * pandaNumDofs
# joint ranges for null space (todo: set them to proper range)
jr = [7] * pandaNumDofs

initPositions = (-0.160, -0.494, 0.121, -2.115, 0.057, 1.623, 0.728, 0.029, 0.030)
# initPositions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
rp = initPositions


def speed_limit(x):
    if x > 0.005:
        return np.array(0.005)
    elif x < -0.005:
        return np.array(-0.005)
    elif np.abs(x) <= 0.001:
        return np.array(0)
    else:
        return np.array(x)


class Robot(object):
    def __init__(self, bullet_client):
        self.bullet_client = bullet_client
        self.jointPoses = [0, 0, 0, 0, 0, 0, 0]
        self.pos = [0.4, 0, 0.5]
        self.finger = 0.04
        self.euler = [math.pi / 1., 0., 0.]
        self.grasp_pos = [0.0, 0.0, 0.0]
        self.angle = 0

        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.panda = self.bullet_client.loadURDF("\\franka_panda\\panda.urdf",
                                                 [0, 0, 0], useFixedBase=True, flags=flags)

        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        self.reset()

        self.move(self.pos, self.euler)
        self.gripper(0.04)

    def step(self, pos, angle=None, lift=0):
        if angle is None:
            angle = self.angle
        self.angle = angle
        if pos[2] < 0.015:
            pos[2] = 0.015
        dX = speed_limit(pos[0] - self.pos[0])
        dY = speed_limit(pos[1] - self.pos[1])
        dZ = speed_limit(pos[2] - self.pos[2])
        dA = speed_limit(angle - self.euler[2])
        ok = 0
        if lift:
            # z first
            self.pos, self.euler = self.move([self.pos[0], self.pos[1], self.pos[2] + dZ])
            if dZ == 0:
                self.pos, self.euler = self.move([self.pos[0] + dX, self.pos[1] + dY, self.pos[2]],
                                                 self.euler + np.array([0, 0, dA]))
        else:
            # x, y first
            self.pos, self.euler = self.move([self.pos[0] + dX, self.pos[1] + dY, self.pos[2]])
            if dX * dY == 0:
                self.pos, self.euler = self.move([self.pos[0], self.pos[1], self.pos[2] + dZ],
                                                 self.euler + np.array([0, 0, dA]))
        d = np.sum([np.abs(dX), np.abs(dY), np.abs(dZ), np.abs(dA)])
        if d == 0.0:
            ok = 1
        return ok

    def move(self, targetpos, euler=None):
        if euler is None:
            euler = self.euler
        orn = self.bullet_client.getQuaternionFromEuler(euler)
        self.jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, targetpos, orn, ll, ul,
                                                                        jr, rp, maxNumIterations=100)
        for i in range(7):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.jointPoses[i], force=5 * 240.)
        return targetpos, euler

    def gripper(self, finger):
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     finger, force=10)

    def reset(self):
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)

            jointName = info[1]
            jointType = info[2]

            if jointType == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(self.panda, j, initPositions[index])
                index = index + 1
            if jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda, j, initPositions[index])
                index = index + 1

            self.gripper(0.04)

    def setCameraPicAndGetPic(self, width: int = 224, height: int = 224, physicsClientId: int = 0):
        baseOrientation = self.bullet_client.getQuaternionFromEuler([math.pi / 1., 0., 0.])
        matrix = self.bullet_client.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
        tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # 变换后的x轴
        ty_vec = np.array([matrix[1], matrix[4], matrix[7]])  # 变换后的y轴
        tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # 变换后的z轴

        cameraPos = [0.5, 0, 0.5] + 0.05 * tx_vec - 0.05 * tz_vec
        # targetPos = cameraPos - 0.01 * tx_vec + 0.04 * tz_vec
        targetPos = cameraPos + 0.04 * tz_vec
        viewMatrix = self.bullet_client.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=tx_vec,
            physicsClientId=physicsClientId
        )
        projectionMatrix = self.bullet_client.computeProjectionMatrixFOV(
            fov=50.0,  # 摄像头的视线夹角
            aspect=1.0,
            nearVal=0.01,  # 摄像头焦距下限
            farVal=20,  # 摄像头能看上限
            physicsClientId=physicsClientId
        )
        width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(
            width=width, height=height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=physicsClientId
        )

        return width, height, rgbImg, depthImg, segImg

    def accurateCalculateInverseKinematics(self, endEffectorId, targetPos, orn, threshold, maxIter):
        closeEnough = False
        iter = 0
        dist2 = 1e30
        while not closeEnough and iter < maxIter:
            jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, endEffectorId, targetPos, orn)
            for i in range(pandaNumDofs):
                self.bullet_client.resetJointState(self.panda, i, jointPoses[i])
            ls = self.bullet_client.getLinkState(self.panda, endEffectorId)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < threshold)
            iter = iter + 1

        return jointPoses

