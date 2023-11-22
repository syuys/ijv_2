#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:03:27 2021

@author: md703
"""


import numpy as np
import json
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% parameters and function setting
# subject = "Eric"
# date = "20211023"
# state = "IJVSmall"  # it's ok for both Large & Small
folder = "ultrasound_image_processing"
# with open(os.path.join(folder, "blood_vessel_segmentation_line.json")) as f:
#     paramSet = json.load(f)[subject][date][state]

projectID = "20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99"
changeType = "pulse"  # depth, minor
times = "150%"  # +0
unit = "mm"  # std
ccafixed = False
# sessionID = f"ijv_dis_{changeType}_{times}_{unit}"
sessionID = "ijv_dis_depth_+0_std"  # doesn't matter for dis or col,  ijv_dis_depth_+0_std  ijv_dis_minor_-3.5_mm
voxelLength = 0.25  # [mm]
isSaveVolume = False
# fileName = f"tinyHolder_contrast_sds_5to45_cca_sAng_{sessionID.split('_')[-1]}"
# fileName = f"tinyHolder_contrast_sds_5to45_{sessionID.split('_')[2]}_upperNeck"
fileName = f"tinyHolder_contrast_sds_5to45_ijv_{changeType}_{times}"

with open(os.path.join(projectID, sessionID, "model_parameters.json")) as f:
    modelParam = json.load(f)

# length10mmEdge = paramSet["length10mmEdge"]

# gridNumIn10mm = int(10/voxelLength)
# scalePercentage = gridNumIn10mm / (length10mmEdge[1]-length10mmEdge[0])

def convertUnit(length, voxelSize=voxelLength):
    numGrid = int(np.around(length / voxelSize))
    return numGrid

def isInEllipse(x, y, major, minor):  # if circle, major=minor
    state = x**2/major**2 + y**2/minor**2 < 1
    return state

def getXofIntersection(theta, major, minor):
    return np.sqrt(major**2 * minor**2 / (major**2 * np.tan(theta)**2 + minor**2))


# %% read data

### related size [mm]
modelSize = modelParam["ModelSize"]
hardwareParam = modelParam["HardwareParam"]
geoParam = modelParam["GeoParam"]
# model
modelX = convertUnit(modelSize["XSize"])
modelY = convertUnit(modelSize["YSize"])
modelZ = convertUnit(modelSize["ZSize"])
print(f"Total Number of volume voxels = {modelX*modelY*modelZ}.")
# source
srcHolderX = convertUnit(hardwareParam["Source"]["Holder"]["XSize"])
srcHolderY = convertUnit(hardwareParam["Source"]["Holder"]["YSize"])
srcHolderZ = convertUnit(hardwareParam["Source"]["Holder"]["ZSize"])
irraWinRadius = convertUnit(hardwareParam["Source"]["Holder"]["IrraWinRadius"])
# detecotr
detHolderX = convertUnit(hardwareParam["Detector"]["Holder"]["XSize"])  # sds limit - srcHolderX/2 + prismLength/2
detHolderY = convertUnit(hardwareParam["Detector"]["Holder"]["YSize"])
detHolderZ = convertUnit(hardwareParam["Detector"]["Holder"]["ZSize"])
prismY = convertUnit(hardwareParam["Detector"]["Prism"]["legSize"])  # prismX need not be set
prismZ = convertUnit(hardwareParam["Detector"]["Prism"]["legSize"])

# ijv
ijvDepth = convertUnit(geoParam["IJV"]["Depth"])  # mm to grid
ijvMajorAxisNormalInMM = geoParam["IJV"]["MajorAxisNormal"]  # mm  # 7.9795
ijvMinorAxisNormalInMM = geoParam["IJV"]["MinorAxisNormal"]  # mm  # 3.76175
ijvMajorAxisChangePct = geoParam["IJV"]["MajorAxisChangePct"]  # 0.081
ijvMinorAxisChangePct = geoParam["IJV"]["MinorAxisChangePct"]  # 0.127
ijvMajorAxisLarge = convertUnit(ijvMajorAxisNormalInMM*(1+ijvMajorAxisChangePct))  # mm to grid
ijvMinorAxisLarge = convertUnit(ijvMinorAxisNormalInMM*(1+ijvMinorAxisChangePct))  # mm to grid
ijvMajorAxisSmall = convertUnit(ijvMajorAxisNormalInMM*(1-ijvMajorAxisChangePct))  # mm to grid
ijvMinorAxisSmall = convertUnit(ijvMinorAxisNormalInMM*(1-ijvMinorAxisChangePct))  # mm to grid

# cca
ccaRadiusInMM = geoParam["CCA"]["Radius"]  # mm
ccaRadius = convertUnit(ccaRadiusInMM)  # mm to grid
sDistInMM = geoParam["CCA"]["sDist"]  # mm
sAng = np.deg2rad(geoParam["CCA"]["sAng"])  # deg to rad
if sAng < np.pi/2:
    xIntersecInMM = -getXofIntersection(sAng, ijvMajorAxisNormalInMM, ijvMinorAxisNormalInMM)  # mm
elif sAng > np.pi/2:
    xIntersecInMM = getXofIntersection(sAng, ijvMajorAxisNormalInMM, ijvMinorAxisNormalInMM)  # mm
else:
    raise Exception("sAng = 90 degrees !!")
yIntersecInMM = xIntersecInMM * np.tan(sAng)  # mm
if ccafixed == False:
    cca2ijvInMM = np.sqrt(xIntersecInMM**2 + yIntersecInMM**2) + sDistInMM + ccaRadiusInMM  # center to center, in mm
else:
    cca2ijvInMM = 19  # 14.5
print(f"Distance of IJV center to CCA center: {round(cca2ijvInMM, 2)} mm")
ccaShiftY = -convertUnit(cca2ijvInMM * np.cos(sAng))  # mm to grid
ccaShiftZ = convertUnit(cca2ijvInMM * np.sin(sAng))  # mm to grid


# %% make volume

### start to construct

# model and air (in the beginning)
vol = np.ones((modelX, modelY, modelZ))

# source
vol[modelX//2-srcHolderX//2:modelX//2+srcHolderX//2, 
    modelY//2-srcHolderY//2:modelY//2+srcHolderY//2,
    :srcHolderZ] = 2  # holder
for x in range(modelX//2-irraWinRadius, modelX//2+irraWinRadius):
    for y in range(modelY//2-irraWinRadius, modelY//2+irraWinRadius):
        isDist1 = np.sqrt((modelX//2-x)**2 + (modelY//2-y)**2) < irraWinRadius
        isDist2 = np.sqrt((modelX//2-(x+1))**2 + (modelY//2-y)**2) < irraWinRadius
        isDist3 = np.sqrt((modelX//2-x)**2 + (modelY//2-(y+1))**2) < irraWinRadius
        isDist4 = np.sqrt((modelX//2-(x+1))**2 + (modelY//2-(y+1))**2) < irraWinRadius
        if isDist1 or isDist2 or isDist3 or isDist4:
            vol[x][y] = 1  # air

# detector
vol[modelX//2+srcHolderX//2:modelX//2+srcHolderX//2+detHolderX, 
    modelY//2-detHolderY//2:modelY//2+detHolderY//2,
    :detHolderZ] = 2  # first holder
vol[modelX//2-srcHolderX//2-detHolderX:modelX//2-srcHolderX//2, 
    modelY//2-detHolderY//2:modelY//2+detHolderY//2,
    :detHolderZ] = 2  # second holder
vol[modelX//2+srcHolderX//2:modelX//2+srcHolderX//2+detHolderX, 
    modelY//2-prismY//2:modelY//2+prismY//2,
    detHolderZ-prismZ:detHolderZ] = 3  # first prism
vol[modelX//2-srcHolderX//2-detHolderX:modelX//2-srcHolderX//2, 
    modelY//2-prismY//2:modelY//2+prismY//2,
    detHolderZ-prismZ:detHolderZ] = 3  # second prism
vol[modelX//2+srcHolderX//2:modelX//2+srcHolderX//2+detHolderX, 
    modelY//2-prismY//2:modelY//2+prismY//2,
    :detHolderZ-prismZ] = 0  # first fiber
vol[modelX//2-srcHolderX//2-detHolderX:modelX//2-srcHolderX//2, 
    modelY//2-prismY//2:modelY//2+prismY//2,
    :detHolderZ-prismZ] = 0  # second fiber

# muscle
vol[:, :, detHolderZ:] = 6

# skin
# skinDepth = int(paramSet["skin"]["x"]*scalePercentage)
skinDepth = convertUnit(geoParam["Skin"]["Thickness"])
vol[:, :, detHolderZ:detHolderZ+skinDepth] = 4

# fat
# fatDepth = int(paramSet["fat"]["x"]*scalePercentage)
fatDepth = convertUnit(geoParam["Fat"]["Thickness"])
vol[:, :, detHolderZ+skinDepth:detHolderZ+skinDepth+fatDepth] = 5

# ijv - perturbed area (large minus small)
for y in range(-ijvMajorAxisLarge, ijvMajorAxisLarge):
    for z in range(-ijvMinorAxisLarge, ijvMinorAxisLarge):
        isDist1 = isInEllipse(y, z, ijvMajorAxisLarge, ijvMinorAxisLarge)
        isDist2 = isInEllipse(y+1, z, ijvMajorAxisLarge, ijvMinorAxisLarge)
        isDist3 = isInEllipse(y, z+1, ijvMajorAxisLarge, ijvMinorAxisLarge)
        isDist4 = isInEllipse(y+1, z+1, ijvMajorAxisLarge, ijvMinorAxisLarge)
        if isDist1 or isDist2 or isDist3 or isDist4:
            vol[:, y+modelY//2, z+detHolderZ+ijvDepth] = 7

# ijv - area when collapsed
for y in range(-ijvMajorAxisSmall, ijvMajorAxisSmall):
    for z in range(-ijvMinorAxisSmall, ijvMinorAxisSmall):
        isDist1 = isInEllipse(y, z, ijvMajorAxisSmall, ijvMinorAxisSmall)
        isDist2 = isInEllipse(y+1, z, ijvMajorAxisSmall, ijvMinorAxisSmall)
        isDist3 = isInEllipse(y, z+1, ijvMajorAxisSmall, ijvMinorAxisSmall)
        isDist4 = isInEllipse(y+1, z+1, ijvMajorAxisSmall, ijvMinorAxisSmall)
        if isDist1 or isDist2 or isDist3 or isDist4:
            vol[:, y+modelY//2, z+detHolderZ+ijvDepth] = 8

# cca
for y in range(-ccaRadius, ccaRadius):
    for z in range(-ccaRadius, ccaRadius):
        isDist1 = isInEllipse(y, z, ccaRadius, ccaRadius)
        isDist2 = isInEllipse(y+1, z, ccaRadius, ccaRadius)
        isDist3 = isInEllipse(y, z+1, ccaRadius, ccaRadius)
        isDist4 = isInEllipse(y+1, z+1, ccaRadius, ccaRadius)
        if isDist1 or isDist2 or isDist3 or isDist4:
            vol[:, y+modelY//2+ccaShiftY, z+detHolderZ+ijvDepth+ccaShiftZ] = 9  # ijvDepth, 58


tissue = ["Fiber", "Air", "PLA", "Prism", "Skin", "Fat", "Muscle", "Perturbed region", "IJV", "CCA"]
legendfontsize = 13

# %% front view - all and only tissue
# plt.imshow(vol[modelX//2, 160:320, 24:200].T)
plt.imshow(vol[modelX//2, modelY//2-85:modelY//2+85, 24:160].T)  # 190 for upperedge, 165 for cca
plt.axis("off")
# plt.colorbar()
plt.show()

# tmp = vol[modelX//2, modelY//2-85:modelY//2+85, 24:170].T
# values = np.unique(tmp.ravel())
# fig, ax = plt.subplots(1, 1)
# im = plt.imshow(tmp)
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[idx], label=tissue[int(v)] ) for idx, v in enumerate(values) ]
# ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), edgecolor="black", fontsize=legendfontsize*1.164, loc=2, borderaxespad=0. )
# ax.text(0.01, 0.02, "(a)", fontsize=legendfontsize*2, horizontalalignment='left',
#          verticalalignment='bottom', transform=ax.transAxes)
# ax.axis("off")
# plt.show()

# %% top view
fig, ax = plt.subplots(1, 1)
tmp = vol[modelX//2-290:modelX//2+290, modelY//2-100:modelY//2+100, 0].T
# tmp = vol[:, :, 0].T
values = np.unique(tmp.ravel())
im = ax.imshow(tmp)
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[idx], label=tissue[int(v)] ) for idx, v in enumerate(values) ]
ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), edgecolor="black", 
          # fontsize=legendfontsize,
          fontsize="medium",
          loc=2, borderaxespad=0. )
ax.axis("off")
ax.set_xticks(np.linspace(-0.5, tmp.T.shape[0]-0.5, num=5), 
           np.linspace(-tmp.T.shape[0]*voxelLength/2, tmp.T.shape[0]*voxelLength/2, num=5))
ax.set_yticks(np.linspace(-0.5, tmp.T.shape[1]-0.5, num=5), 
           np.linspace(-tmp.T.shape[1]*voxelLength/2, tmp.T.shape[1]*voxelLength/2, num=5))
# ax.set_xlabel("X [mm]")
# ax.set_ylabel("Y [mm]")
# ax.set_title("Top view (whole)")
plt.show()

# top view (zoom in)
# fig, ax = plt.subplots(1, 1)
# # top = vol[modelX//2-290:modelX//2+290, modelY//2-100:modelY//2+100, 0].T
# top = vol[modelX//2-10:modelX//2+10, modelY//2-10:modelY//2+10, 0].T
# values = np.unique(top.ravel())
# im = ax.imshow(top)
# plt.plot(9.5, 9.5, marker="x", color="white")
# # ax.axhline(11.5, color ='white')
# # ax.axvline(11.5, color ='white')
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[idx], label=tissue[int(v)] ) for idx, v in enumerate(values) ]
# ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), edgecolor="black", fontsize=legendfontsize, loc=2, borderaxespad=0. )
# # ax.axis("off")
# ax.set_xticks(np.linspace(-0.5, top.T.shape[0]-0.5, num=5), 
#            np.linspace(-top.T.shape[0]*voxelLength/2, top.T.shape[0]*voxelLength/2, num=5))
# ax.set_yticks(np.linspace(-0.5, top.T.shape[1]-0.5, num=5), 
#            np.linspace(-top.T.shape[1]*voxelLength/2, top.T.shape[1]*voxelLength/2, num=5))
# ax.set_xlabel("X [mm]")
# ax.set_ylabel("Y [mm]")
# ax.set_title("Top view (near source)")
# plt.show()

# %% side view
fig, ax = plt.subplots(1, 1)
tmp = vol[modelX//2-290:modelX//2+290, modelY//2, :29].T
values = np.unique(tmp.ravel())
im = ax.imshow(tmp)
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[idx], label=tissue[int(v)] ) for idx, v in enumerate(values) ]
ax.legend(handles=patches, bbox_to_anchor=(1.02, 4.42), edgecolor="black", fontsize="medium", loc=2, borderaxespad=0. )
ax.axis("off")
plt.show()

# top view - instrument
# tmp = vol[modelX//2-290:modelX//2+290, modelY//2-85:modelY//2+85, 0].T
# values = np.unique(tmp.ravel())
# fig, ax = plt.subplots(1, 1)
# im = ax.imshow(tmp)
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[idx], label=tissue[int(v)] ) for idx, v in enumerate(values) ]
# ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), edgecolor="black", fontsize=legendfontsize, loc=2, borderaxespad=0. )
# ax.text(0.01, 0.02, "(a)", fontsize=legendfontsize*1.7, horizontalalignment='left',
#          verticalalignment='bottom', transform=ax.transAxes)
# ax.axis("off")
# plt.show()

# side view - only tissue
# plt.imshow(vol[:, modelY//2, :].T)
# plt.colorbar()
# plt.show()
# tmp = vol[modelX//2-130:modelX//2+130, modelY//2, 24:170].T
# values = np.unique(tmp.ravel())
# fig, ax = plt.subplots(1, 1)
# im = ax.imshow(tmp)
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[idx], label=tissue[int(v)] ) for idx, v in enumerate(values) ]
# ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), edgecolor="black", fontsize=legendfontsize, loc=2, borderaxespad=0. )
# ax.text(0.01, 0.02, "(b)", fontsize=legendfontsize*1.7, horizontalalignment='left',
#          verticalalignment='bottom', transform=ax.transAxes)
# ax.axis("off")
# plt.show()

# side view - instrument
# tmp = vol[modelX//2-290:modelX//2+290, modelY//2, :29].T
# values = np.unique(tmp.ravel())
# fig, ax = plt.subplots(1, 1)
# im = ax.imshow(tmp)
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[idx], label=tissue[int(v)] ) for idx, v in enumerate(values) ]
# ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), edgecolor="black", fontsize=legendfontsize/2, loc=2, borderaxespad=0. )
# ax.text(0.01, 0.02, "(b)", fontsize=legendfontsize, horizontalalignment='left',
#          verticalalignment='bottom', transform=ax.transAxes)
# ax.axis("off")
# plt.show()

# save file
if isSaveVolume:
    print("File will be saved !!")
    path = os.path.join(folder, projectID)
    if not os.path.isdir(path):
        os.mkdir(path)
    vol = vol.astype(np.uint8)
    np.save(file=os.path.join(path, f"perturbed_{fileName}"), arr=vol)