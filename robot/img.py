import numpy as np
import cv2


def depth_grey(pic):
    depth_img = pic - np.min(pic)
    depth_img = (depth_img / np.max(depth_img) * 255).astype(np.uint8)

    return depth_img


def RGBA_RGB(rgb_img):
    planes = cv2.split(rgb_img)
    rgb_img = cv2.merge([planes[2], planes[1], planes[0]])

    return rgb_img


def graspQ(q_img, x, y):
    # q >= 0
    q_img = np.where(q_img < 0, 0, q_img)
    q_img = np.where(q_img == 1, 1, q_img)
    # normalise q_img [0, 255]
    q_img = (q_img * 255).astype(np.uint8)
    # b g channel
    zeros = np.zeros([300, 300]).astype(np.uint8)
    # as red channel
    q_img = cv2.merge([zeros, zeros, q_img])
    # Green point as center point
    # array[r][c], cv2 [x, y], transpose needed
    q_img = cv2.circle(q_img, [x, y], 0, color=(0, 255, 0), thickness=3)

    return q_img


def graspAng(ang_img):
    # ang_img > 0 red
    _gb = (np.where(ang_img < 0, 0, ang_img) / 90 * 255).astype(np.uint8)
    # ang_img < 0 blue
    _gr = (np.where(ang_img > 0, 0, ang_img) / -90 * 255).astype(np.uint8)
    # white
    white = np.ones([300, 300]).astype(np.uint8) * 255
    # white bottom: red = [b: 255 - x, g: 255 - x, r: 255]
    b = white - _gb
    g = white - _gb - _gr
    r = white - _gr
    # generate rgb pic
    ang_img = cv2.merge([b, g, r])

    return ang_img


def graspW(width_img):
    # 0 <= width <= 150
    _gb = (np.where(width_img < 0, 0, width_img) / 150 * 255).astype(np.uint8)
    # white
    white = np.ones([300, 300]).astype(np.uint8) * 255
    b = white - _gb
    g = white - _gb
    r = white
    width_img = cv2.merge([b, g, r])

    return width_img


def graspRGB(rgb_img, angle, width, x, y):
    # draw the grasp on RGB image
    dy = int(np.floor(width / 2 * np.sin(angle)) / 3)
    dx = int(np.floor(width / 2 * np.cos(angle)) / 3)
    # a point is represented by (x, y) in cv2, transpose needed
    start = [x - dx, y + dy]
    end = [x + dx, y - dy]

    # rgb image
    rgb_img = cv2.line(rgb_img, start, end, color=(0, 255, 0), thickness=2)  # Green line
    rgb_img = cv2.circle(rgb_img, [x, y], 0, color=(0, 0, 255), thickness=2)  # Red point as center point

    return rgb_img



