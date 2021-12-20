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


# %% parameters setting
subject = "Eric"
date = "20211023"
state = "IJVSmall"
tissueSet = ["skin", "fat", "ijv", "cca"]
with open("blood_vessel_segmentation_line.json") as f:
    paramSet = json.load(f)[subject][date][state]
skinStartEdge = paramSet["skinStartEdge"]
bloodThold = paramSet["bloodThold"]
length10mmEdge = paramSet["length10mmEdge"]
voxelLength = 0.25  # [mm]
gridNumIn10mm = int(10/voxelLength)


# %% main
# plot original image
image = cv2.imread("{}_{}_{}.jpg".format(subject, date, state), cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap="gray")
plt.colorbar()
plt.title("original")
plt.show()

# plot tissue area only
plt.imshow(image[160:1500], cmap="gray")
plt.colorbar()
plt.title("tissue (but no trim to skin surface)")
plt.show()

# trim image to start from skin surface
image = image[skinStartEdge:1500]
plt.imshow(image, cmap="gray")
plt.plot([0, 0], length10mmEdge, "r.")
plt.colorbar()
plt.title("tissue (trim to skin surface)")
plt.show()


# %% catch and plot real target tissue area
legalRowSet = {}
legalColSet = {}
for idx, tissue in enumerate(tissueSet):
    # highlight
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    if tissue == "skin" or tissue == "fat":
        plt.axhline(y=paramSet[tissue]["x"], c="r", linewidth=1)
    if tissue == "ijv":
        for vector, edge in zip(paramSet[tissue]["v"], np.array(paramSet[tissue]["x"])):
            y = -vector[2]/vector[3]*edge - vector[1]/vector[3]
            plt.plot(y, edge, c="r")
    if tissue == "cca":
        angles = np.linspace(0, 2*np.pi, num=100)
        x = paramSet[tissue]["v"][2] * np.cos(angles) + paramSet[tissue]["v"][0]
        y = paramSet[tissue]["v"][2] * np.sin(angles) + paramSet[tissue]["v"][1]
        plt.plot(x, y, c="r")
    plt.title("({}) potential {} area".format(state, tissue))
    plt.show()
    
    if tissue == "ijv" or tissue == "cca":
        # highlight and catch target tissue
        plt.imshow(image, cmap="gray")
        plt.colorbar()
        if tissue == "ijv":
            for vector, edge in zip(paramSet[tissue]["v"], np.array(paramSet[tissue]["x"])):
                y = -vector[2]/vector[3]*edge - vector[1]/vector[3]
                plt.plot(y, edge, c="r")
        if tissue == "cca":
            angles = np.linspace(0, 2*np.pi, num=100)
            x = paramSet[tissue]["v"][2] * np.cos(angles) + paramSet[tissue]["v"][0]
            y = paramSet[tissue]["v"][2] * np.sin(angles) + paramSet[tissue]["v"][1]
            plt.plot(x, y, c="r")
        # mesh coordinate
        coordinates = np.meshgrid(np.arange(image.shape[1]), np.flip(np.arange(image.shape[1])))
        coordinates = np.insert(coordinates, 0, 1, axis=0)  # add dummy coordinate
        # sketch potential region
        targetMatch = np.ones((image.shape[1], image.shape[1]), dtype=bool)
        # match line
        if tissue == "ijv":
            for vector in np.array(paramSet[tissue]["v"]):
                if vector[0] == -1:
                    targetMatch = targetMatch & ((vector.reshape(-1, 1, 1)[1:] * coordinates).sum(axis=0) < 0)
                if vector[0] == 1:
                    targetMatch = targetMatch & ((vector.reshape(-1, 1, 1)[1:] * coordinates).sum(axis=0) > 0)
            # match blood
            targetMatch = targetMatch & (np.rot90(image[:image.shape[1], :image.shape[1]]) < bloodThold)
        if tissue == "cca":
            targetMatch = targetMatch & ((coordinates[1]-paramSet[tissue]["v"][1])**2 + (coordinates[2]-paramSet[tissue]["v"][0])**2 <= paramSet[tissue]["v"][2]**2)
        # plot image (before scale)
        legalRow, legalCol = np.where(targetMatch == True)
        legalRow = targetMatch.shape[0]-legalRow
        plt.imshow(image, cmap="gray")
        plt.plot(legalRow, legalCol, "r.", markersize=3)
        plt.title("catch {} (before scale)".format(tissue))
        plt.show()
        
        # plot image (after scale)
        scalePercentage = gridNumIn10mm / (length10mmEdge[1]-length10mmEdge[0])
        newImage = cv2.resize(image, (int(np.round(image.shape[1]*scalePercentage)), int(np.round(image.shape[0]*scalePercentage))), interpolation=cv2.INTER_AREA)
        targetMatch = targetMatch.astype(float)
        targetMatch = cv2.resize(targetMatch, (int(np.round(targetMatch.shape[1]*scalePercentage)), int(np.round(targetMatch.shape[0]*scalePercentage))), interpolation=cv2.INTER_AREA)
        targetMatch = targetMatch.astype(bool)
        legalRow, legalCol = np.where(targetMatch == True)
        legalRow = targetMatch.shape[0]-legalRow
        plt.imshow(newImage, cmap="gray")
        plt.colorbar()
        plt.title("image after scale")
        plt.show()
        plt.imshow(newImage, cmap="gray")
        plt.colorbar()
        plt.plot(legalRow, legalCol, "r.", markersize=3)
        plt.title("catch {} (after scale)".format(tissue))
        plt.show()
        
        legalRowSet[tissue] = legalRow
        legalColSet[tissue] = legalCol


# %% make volume
def convertUnit(length, voxelSize=voxelLength):
        numGrid = length / voxelSize
        return numGrid

### related size [mm]
# model
modelX = convertUnit(132)
modelY = convertUnit(38)
modelZ = convertUnit(32)
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
vol[int(modelX//2-srcHolderX//2):int(modelX//2+srcHolderX//2), 
    int(modelY//2-srcHolderY//2):int(modelY//2+srcHolderY//2),
    :int(srcHolderZ)] = 2  # holder
for x in range(int(modelX//2)-int(np.ceil(irraWinRadius)), int(modelX//2)+int(np.ceil(irraWinRadius))):
    for y in range(int(modelY//2)-int(np.ceil(irraWinRadius)), int(modelY//2)+int(np.ceil(irraWinRadius))):
        isDist1 = np.sqrt((modelX//2-x)**2 + (modelY//2-y)**2) < np.ceil(irraWinRadius)
        isDist2 = np.sqrt((modelX//2-(x+1))**2 + (modelY//2-y)**2) < np.ceil(irraWinRadius)
        isDist3 = np.sqrt((modelX//2-x)**2 + (modelY//2-(y+1))**2) < np.ceil(irraWinRadius)
        isDist4 = np.sqrt((modelX//2-(x+1))**2 + (modelY//2-(y+1))**2) < np.ceil(irraWinRadius)
        if isDist1 or isDist2 or isDist3 or isDist4:
            vol[x][y] = 1  # air
# detector
vol[int(modelX//2+srcHolderX//2):int(modelX//2+srcHolderX//2+detHolderX), 
    int(modelY//2-detHolderY//2):int(modelY//2+detHolderY//2),
    :int(detHolderZ)] = 2  # first holder
vol[int(modelX//2-srcHolderX//2-detHolderX):int(modelX//2-srcHolderX//2), 
    int(modelY//2-detHolderY//2):int(modelY//2+detHolderY//2),
    :int(detHolderZ)] = 2  # second holder
vol[int(modelX//2+srcHolderX//2):int(modelX//2+srcHolderX//2+detHolderX), 
    int(modelY//2-prismY//2):int(modelY//2+prismY//2),
    int(detHolderZ-prismZ):int(detHolderZ)] = 3  # first prism
vol[int(modelX//2-srcHolderX//2-detHolderX):int(modelX//2-srcHolderX//2), 
    int(modelY//2-prismY//2):int(modelY//2+prismY//2),
    int(detHolderZ-prismZ):int(detHolderZ)] = 3  # second prism
vol[int(modelX//2+srcHolderX//2):int(modelX//2+srcHolderX//2+detHolderX), 
    int(modelY//2-prismY//2):int(modelY//2+prismY//2),
    :int(detHolderZ-prismZ)] = 0  # first fiber
vol[int(modelX//2-srcHolderX//2-detHolderX):int(modelX//2-srcHolderX//2), 
    int(modelY//2-prismY//2):int(modelY//2+prismY//2),
    :int(detHolderZ-prismZ)] = 0  # second fiber
# fiberCenterX = int(modelX//2+srcHolderX//2+detHolderX//2)
# fiberCenterY = int(modelY//2)
# for x in range(fiberCenterX-int(np.ceil(fiberR)), fiberCenterX+int(np.ceil(fiberR))):
#     for y in range(fiberCenterY-int(np.ceil(fiberR)), fiberCenterY+int(np.ceil(fiberR))):
#         isDist1 = np.sqrt((fiberCenterX-x)**2 + (fiberCenterY-y)**2) < np.ceil(fiberR)
#         isDist2 = np.sqrt((fiberCenterX-(x+1))**2 + (fiberCenterY-y)**2) < np.ceil(fiberR)
#         isDist3 = np.sqrt((fiberCenterX-x)**2 + (fiberCenterY-(y+1))**2) < np.ceil(fiberR)
#         isDist4 = np.sqrt((fiberCenterX-(x+1))**2 + (fiberCenterY-(y+1))**2) < np.ceil(fiberR)
#         if isDist1 or isDist2 or isDist3 or isDist4:
#             vol[x, y, :int(detHolderZ-prismZ)] = 0  # fiber
# muscle
vol[:, :, int(detHolderZ):] = 6
# fat
fatDepth = int(paramSet["fat"]["x"]*scalePercentage)
vol[:, :, int(detHolderZ):int(detHolderZ)+fatDepth] = 5
# skin
skinDepth = int(paramSet["skin"]["x"]*scalePercentage)
vol[:, :, int(detHolderZ):int(detHolderZ)+skinDepth] = 4
# %% Can add perturbed region
# ijv
shiftNumber = np.round(modelY//2 - np.mean(legalRowSet["ijv"]), 0).astype(int)
vol[:, np.array(legalRowSet["ijv"])+shiftNumber, np.array(legalColSet["ijv"])+int(detHolderZ)] = 7 if state == "IJVLarge" else 8    
# cca
# if state == "IJVSmall":
#     vol[:, np.array(legalRowSet["cca"])+shiftNumber, np.array(legalColSet["cca"])+int(detHolderZ)] = 9
if state == "IJVLarge":
    vol[:, np.array(legalRowSet["cca"])+shiftNumber, np.array(legalColSet["cca"])+int(detHolderZ)] = 9
vol[:, 78, 62] = 7  # fill the hole
plt.imshow(vol[int(modelX//2), :, :])
plt.show()
plt.imshow(vol[int(modelX//2), :, :].T)
plt.show()
plt.imshow(vol[:, :, 0])
plt.show()
plt.imshow(vol[:, int(modelY//2), :].T)
plt.show()
# save file
vol = vol.astype(np.uint8)
np.save(file="perturbed_large_to_small", arr=vol)
# sio.savemat('testIJV.mat', {'testIJV': vol})
# sio.savemat('testIJV.mat', {'testIJV': vol[:, :, :int(detHolderZ)]})





