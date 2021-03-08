"""
Created by Jan Schiffeler on 08.03.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import torch
import numpy as np
import cv2
import glob
import os.path as osp

for img_path in glob.glob("fake_cam/*.png"):
    img = cv2.imread(img_path, 0)
    keys = torch.tensor([0])
    cam = cv2.resize(img, (125, 94))
    cam = torch.from_numpy(cam[np.newaxis, :, :].astype("float32"))
    cam_npy = {'keys': keys, 'cam': cam, 'high_res': cv2.resize(img, (500, 375))[np.newaxis, :, :]}
    np.save(f"result/cam/{osp.basename(img_path)[:-4]}.npy", cam_npy)
