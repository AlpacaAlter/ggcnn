import math
import numpy as np


def frame_transform(pos, depth, width):
    pos_cam, cam_angle, width = img_to_cam(pos, depth, width, [-0.00, 0, 0.04])
    grasp_pos = cam_to_robot(pos_cam, cam_angle)

    return grasp_pos, width


def img_to_cam(pos_img, depth, width, cam_target):
    # pixel to length
    cam_height = 0.55
    fov = 50 * math.pi / 180
    cam_ang = cam_target[0] / cam_target[2]
    dl = cam_height * (np.tan(fov / 2) + np.tan(fov / 2)) / 300 * np.cos(cam_ang)
    x = (150 - pos_img[0]) * dl
    y = (pos_img[1] - 150) * dl
    # normalise depth
    p_depth = depth[pos_img[0]][pos_img[1]]
    p_depth = 1 / ((1 - p_depth) * 100)
    if p_depth < 0:
        p_depth = 0
    z = p_depth

    width = width * dl * z / cam_height

    pos_cam = np.array([x, y, z])

    return pos_cam, cam_ang, width


def cam_to_robot(pos_cam, cam_ang):
    # rotate
    x_theta = (-1)*math.pi
    y_theta = (-1)*np.arctan(cam_ang)
    x_rotate = np.array([[1, 0, 0],
                        [0, np.cos(x_theta), (-1)*np.sin(x_theta)],
                        [0, np.sin(x_theta), np.cos(x_theta)]])
    y_rotate = np.array([[np.cos(y_theta), 0, np.sin(y_theta)],
                        [0, 1, 0],
                        [(-1)*np.sin(y_theta), 0, np.cos(y_theta)]])
    pos1 = pos_cam @ y_rotate
    pos2 = pos1 @ x_rotate
    pos = pos2 + np.array([0.55, 0, 0.55])

    return pos

