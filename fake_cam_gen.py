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

for img_path in glob.glob("boulderDataSet/images/*.png"):
    img = cv2.imread(img_path)

    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    rgb_planes = cv2.split(img)
    cl_img = []
    for p in rgb_planes:
        cl_img.append(cl.apply(p))

    equalized_colour = cv2.merge(cl_img)

    edges_img = cv2.pyrDown(cv2.pyrDown(cv2.cvtColor(equalized_colour, cv2.COLOR_BGR2GRAY)))
    edges = cv2.Canny(edges_img, 100, 200)
    edges = cv2.GaussianBlur(edges, (11, 11), 0)
    edges = cv2.resize(edges, (500, 375)).astype("float32") / 255

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(equalized_colour)

    activation_map = cv2.addWeighted(saliencyMap, 0.8, edges, 0.2, 0.0)
    cam = cv2.resize(activation_map, (125, 94))
    cam = torch.from_numpy(cam[np.newaxis, :, :])

    cam_npy = {'keys': torch.tensor([0]), 'cam': cam, 'high_res': cv2.resize(activation_map, (500, 375))[np.newaxis, :, :]}
    np.save(f"result/cam/{osp.basename(img_path)[:-4]}.npy", cam_npy)
    print(img_path)
