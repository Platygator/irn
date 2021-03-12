"""
Created by Jan Schiffeler on 05.03.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import numpy as np
import torch as tc
import cv2

# name = '1613059845760941028'
name = '02_01_00625'

ir_label = f"result/ir_label/{name}.png"
sem_seg = f"result/sem_seg/{name}.png"

cam_npy = f"result/cam/{name}.npy"
ins_seg_npy = f"result/ins_seg/{name}.npy"

original = cv2.imread(f"boulderDataSet/images/{name}.png")
# original = cv2.resize(original, (500, 375))

ir_label = cv2.imread(ir_label)
sem_seg = cv2.imread(sem_seg, 0)

cam_npy = np.load(cam_npy, allow_pickle=True).item()
ins_seg_npy = np.load(ins_seg_npy, allow_pickle=True).item()

original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# create semantic
semantics = cv2.bitwise_and(original, original, mask=sem_seg)
gray_area = np.stack((original_gray,)*3, axis=-1)
gray_area = cv2.bitwise_or(gray_area, gray_area, mask=cv2.bitwise_not(sem_seg // 15 * 255))
semantics += gray_area // 2

# create instance
instances = np.zeros_like(original)
inst_masks = [k.astype('uint8') for k in ins_seg_npy['mask']]
ran_col = [np.rint(np.random.random(3) * 255).astype('uint8') for k in inst_masks]
for n, mask in enumerate(inst_masks):
    instances[:, :, 0] += mask * ran_col[n][0]
    instances[:, :, 1] += mask * ran_col[n][1]
    instances[:, :, 2] += mask * ran_col[n][2]

instances = cv2.addWeighted(instances, 0.5, original, 0.5, 0)
cv2.imwrite("semantic.png", semantics)
cv2.imwrite("instances.png", instances)

cams = cam_npy["cam"].numpy()
cams_high_res = cam_npy['high_res']

cv2.imwrite("cam.png", cams_high_res[0, :, :] * 255)

edge = np.load("edge.npy")
dp = np.load("dp.npy")
print("Done")


