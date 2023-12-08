#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:03:27 2021

@author: md703
"""


from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
import scipy.io as sio
import cv2
import json
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% important part to modify

# save file ?
saveFile = True
fileName = "phantom"

### voxel size and phantom size [mm]
# model
voxelLength = 0.25  # mm
def convertUnit(length, voxelSize=voxelLength):
        numGrid = length / voxelSize
        return numGrid
modelX = convertUnit(66)  # phantom x [mm]
modelY = convertUnit(66)  # phantom y [mm]
modelZ = convertUnit(25)  # phantom z [mm]

# %% part that should not be modified
# source
srcHolderX = convertUnit(28)
srcHolderY = convertUnit(28)
srcHolderZ = convertUnit(6)
irraWinRadius = convertUnit(2.5)
# detecotr
detHolderX = convertUnit(17)
detHolderY = convertUnit(14)
detHolderZ = convertUnit(6)
prismX = convertUnit(17)
prismY = convertUnit(5)
prismZ = convertUnit(5)
# 0.3675
fiberR = convertUnit(0.3675)

### start to construct
# model and air (in the beginning)
vol = np.ones((int(modelX), int(modelY), int(modelZ)))
# source
vol[int(modelX//2-srcHolderX):int(modelX//2), 
    int(modelY//2-srcHolderY//2):int(modelY//2+srcHolderY//2),
    :int(srcHolderZ)] = 2  # holder
count = 0
srcCenterX = modelX//2-srcHolderX//2
for x in range(int(srcCenterX)-int(np.ceil(irraWinRadius)), int(srcCenterX)+int(np.ceil(irraWinRadius))):
    for y in range(int(modelY//2)-int(np.ceil(irraWinRadius)), int(modelY//2)+int(np.ceil(irraWinRadius))):
        isDist1 = np.sqrt((srcCenterX-x)**2 + (modelY//2-y)**2) < np.ceil(irraWinRadius)
        isDist2 = np.sqrt((srcCenterX-(x+1))**2 + (modelY//2-y)**2) < np.ceil(irraWinRadius)
        isDist3 = np.sqrt((srcCenterX-x)**2 + (modelY//2-(y+1))**2) < np.ceil(irraWinRadius)
        isDist4 = np.sqrt((srcCenterX-(x+1))**2 + (modelY//2-(y+1))**2) < np.ceil(irraWinRadius)
        if isDist1 or isDist2 or isDist3 or isDist4:
            count += 1
            vol[x][y] = 1  # air
# detector
vol[int(modelX//2):int(modelX//2+detHolderX), 
    int(modelY//2-detHolderY//2):int(modelY//2+detHolderY//2),
    :int(detHolderZ)] = 2  # first holder
# vol[int(modelX//2-srcHolderX//2-detHolderX):int(modelX//2-srcHolderX//2), 
#     int(modelY//2-detHolderY//2):int(modelY//2+detHolderY//2),
#     :int(detHolderZ)] = 2  # second holder
vol[int(modelX//2):int(modelX//2+detHolderX), 
    int(modelY//2-prismY//2):int(modelY//2+prismY//2),
    int(detHolderZ-prismZ):int(detHolderZ)] = 3  # first prism
# vol[int(modelX//2-srcHolderX//2-detHolderX):int(modelX//2-srcHolderX//2), 
#     int(modelY//2-prismY//2):int(modelY//2+prismY//2),
#     int(detHolderZ-prismZ):int(detHolderZ)] = 3  # second prism
vol[int(modelX//2):int(modelX//2+detHolderX), 
    int(modelY//2-prismY//2):int(modelY//2+prismY//2),
    :int(detHolderZ-prismZ)] = 0  # first fiber
# vol[int(modelX//2-srcHolderX//2-detHolderX):int(modelX//2-srcHolderX//2), 
#     int(modelY//2-prismY//2):int(modelY//2+prismY//2),
#     :int(detHolderZ-prismZ)] = 0  # second fiber
# phantom body
vol[:, :, int(detHolderZ):] = 4
# %% Show
plt.imshow(vol[int(modelX//2), :, :])
plt.show()
plt.imshow(vol[int(modelX//2), :, :].T)
plt.show()
tmp = vol[:, :, 0]
plt.imshow(vol[:, :, 0])
plt.colorbar()
plt.show()
plt.imshow(vol[:, int(modelY//2), :].T)
plt.colorbar()
plt.show()
# save file
if saveFile:
    vol = vol.astype(np.uint8)
    np.save(file=fileName, arr=vol)





