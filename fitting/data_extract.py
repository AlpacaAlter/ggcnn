import glob
import os
from utils.dataset_processing.grasp import GraspRectangles
from utils.dataset_processing.grasp import GraspRectangle
import matplotlib.pyplot as plt

file_path = "..\\Cornell dataset"

graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
graspf.sort()
l = len(graspf)

if l == 0:
    raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(1, 1, 1)
ax = plt.axes(projection='3d')
my_cmap = plt.get_cmap('hsv')
i = 0
ii = []
width = []
angle = []
for pos_f in graspf:

    grs = GraspRectangles.load_from_cornell_file(pos_f)
    for rectangle in grs.to_array():
        gr = GraspRectangle(rectangle)
        # width = gr.width
        # angle = gr.angle
        width.append(gr.width)
        angle.append(gr.angle)
        # print(width, angle)
        ii.append(i)
    i = i + 1
    # if i == 100:
    #     break

ax.set_xlabel('item')
ax.set_ylabel('width')
ax.set_zlabel('angle')
p = ax.scatter3D(ii, width, angle, c=angle, cmap=my_cmap)
fig.colorbar(p, ax=ax)
plt.show()
plt.close()
