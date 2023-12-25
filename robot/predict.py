import torch
import numpy as np
from models.ggcnn import GGCNN
from utils.dataset_processing import image
from models import common


class GraspPredict:
    def __init__(self, path):
        self.model = GGCNN()
        self.model.load_state_dict(torch.load(path))
        self.depth = np.zeros([300, 300])

    def depth_process(self, img):   # processing Depth image
        depth = image.DepthImage(img)
        depth.inpaint()
        depth.normalise()
        self.depth = depth.img

    def net_output(self):
        x = torch.from_numpy(np.expand_dims(self.depth, 0).astype(np.float32))
        pos, cos, sin, width = self.model(x)
        q_img, ang_img, width_img = common.post_process_output(pos, cos, sin, width)
        return q_img, ang_img, width_img

    def grasp(self, q, angle, width):
        # grasp point, angle, width
        q_max = np.max(q)
        idx = np.argmax(q)
        r, c = int(idx // 300), int(idx % 300)
        angle = angle[r][c]
        width = width[r][c]

        return q_max, angle, width, [r, c]



