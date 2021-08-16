#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:24:45 2021

@author: md703
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
import jdata as jd
import os
from glob import glob
import json
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

def plotIntstDistrb(sessionID):
    # read position of source and radius of irradiated window
    with open(os.path.join("output", sessionID, "json_output", "input.json")) as f:
        mcxInput = json.load(f)
    srcPos = np.round(mcxInput["Optode"]["Source"]["Pos"]).astype(int)  # srcPos can be converted to integer although the z value may be 19.9999
    winRadius = int(mcxInput["Optode"]["Source"]["Param2"][2]) # winRadius can be converted to integer
    # glob all flux output and read
    fluxOutputPathSet = glob(os.path.join("output", sessionID, "mcx_output", "*.jnii"))
    data = np.empty((len(fluxOutputPathSet), 
                     mcxInput["Domain"]["Dim"][0],
                     mcxInput["Domain"]["Dim"][1],
                     mcxInput["Domain"]["Dim"][2]))
    for idx, fluxOutputPath in enumerate(fluxOutputPathSet):
        fluxOutput = jd.load(fluxOutputPath)
        header = fluxOutput["NIFTIHeader"]
        data[idx] = fluxOutput["NIFTIData"]
        # print info    
        print("Session name: {} \nDescription: {} \nDim: {} \n\n".format(header["Name"], 
                                                                         header["Description"], 
                                                                         header["Dim"]))
    # read voxel size (voxel size is the same for all header based on this sessionID)
    voxelSize = header["VoxelSize"][0]    
    # process and plot
    data = data.sum(axis=0)
    zDistrb = data.sum(axis=(0, 1))
    # plot distribution along depth
    plt.plot(zDistrb, marker=".")
    plt.xlabel("Depth [grid]")
    plt.ylabel("Energy density")
    plt.title("Distribution of intensity along Z axis")
    plt.show()
    
    # retrieve the distribution in the first skin layer
    xyDistrb = data[:, :, srcPos[2]]
    xyDistrb = xyDistrb / xyDistrb.max()  # normalization for this surface
    # retrieve the distribution in the first skin layer near source
    xyDistrbFocusCenter = xyDistrb[srcPos[0]-2*winRadius:srcPos[0]+2*winRadius,
                                   srcPos[1]-2*winRadius:srcPos[1]+2*winRadius]
    # plot distribution in the first skin layer
    plt.imshow(xyDistrb.T, cmap="jet")
    plt.colorbar()
    plt.xticks(np.linspace(-0.5, xyDistrb.shape[0]-0.5, num=5), 
               np.linspace(-xyDistrb.shape[0]*voxelSize/2, xyDistrb.shape[0]*voxelSize/2, num=5))
    plt.yticks(np.linspace(-0.5, xyDistrb.shape[1]-0.5, num=5), 
               np.linspace(-xyDistrb.shape[1]*voxelSize/2, xyDistrb.shape[1]*voxelSize/2, num=5))
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Distribution of normalized intensity in the first skin layer")
    plt.show()
    # plot distribution in the first skin layer near source
    plt.imshow(xyDistrbFocusCenter.T, cmap="jet")
    plt.colorbar()
    plt.xticks(np.linspace(-0.5, xyDistrbFocusCenter.shape[0]-0.5, num=5), 
               np.linspace(-xyDistrbFocusCenter.shape[0]*voxelSize/2, xyDistrbFocusCenter.shape[0]*voxelSize/2, num=5))
    plt.yticks(np.linspace(-0.5, xyDistrbFocusCenter.shape[1]-0.5, num=5), 
               np.linspace(-xyDistrbFocusCenter.shape[1]*voxelSize/2, xyDistrbFocusCenter.shape[1]*voxelSize/2, num=5))
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Distribution of normalized intensity in the first skin layer near source")
    plt.show()


def analyzeReflectance(sessionID, showCvVariation=False):    
    # parameters
    na = 0.22
    nAir = 1
    nPrism = 1.51
    with open(os.path.join("output", sessionID, "json_output", "input_745.json")) as f:
        mcxInput = json.load(f)
    detNum = len(mcxInput["Optode"]["Detector"])
    
    detOutputPathSet = glob(os.path.join("output", sessionID, "mcx_output", "*.jdat"))
    # sort (to make calculation of cv is consistent in each time)
    detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
    
    # for convenience of compressing and calculating cv, remove some output
    mod = len(detOutputPathSet) % 10
    if mod != 0:
        del detOutputPathSet[-mod:]
    
    # analyze detected photon
    reflectance = np.empty((len(detOutputPathSet), detNum))
    for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
        # read detected data
        detOutput = jd.load(detOutputPath)
        info = detOutput["MCXData"]["Info"]
        photonData = detOutput["MCXData"]["PhotonData"]
        
        # retrieve valid detector ID and valid ppath
        critAng = np.arcsin(na/nPrism)
        afterRefractAng = np.arccos(abs(photonData["v"][:, 2]))
        beforeRefractAng = np.arcsin(nAir*np.sin(afterRefractAng)/nPrism)
        validPhotonBool = beforeRefractAng <= critAng
        validDetID = photonData["detid"][validPhotonBool]
        validDetID = validDetID - 1  # make detid start from 0
        validPPath = photonData["ppath"][validPhotonBool]
        
        # calculate reflectance
        mua = np.array([0.25,  # skin
                        0.1,   # fat
                        0.05,  # muscle
                        0.4,   # IJV
                        0.3    # CCA
                        ])
        validPPath = validPPath[:, 4:]  # retreive the pathlength of skin, fat, muscle, ijv, cca
        
        for detectorIdx in range(info["DetNum"]):
            usedValidPPath = validPPath[validDetID[:, 0]==detectorIdx]
            # I = I0 * exp(-mua*L)
            reflectance[detOutputIdx][detectorIdx] = np.exp(-np.matmul(usedValidPPath, mua)).sum() / info["TotalPhoton"]
            
    # if showCvVariation is set "true", plot cv variation curve.
    if showCvVariation:
        analyzeNum = int(np.floor(np.log10(reflectance.shape[0])))
        photonNum = []
        cv = []
        for i in range(analyzeNum):
            groupingNum = 10 ** i
            sample = reflectance[:groupingNum*10].reshape(groupingNum, 10, detNum)
            sample = sample.mean(axis=0)
            sampleMean = sample.mean(axis=0)
            sampleStd = sample.std(axis=0, ddof=1)
            sampleCV = sampleStd / sampleMean
            photonNum.append(info["TotalPhoton"] * groupingNum)
            cv.append(sampleCV)
        cv = np.array(cv)
        for detectorIdx in range(cv.shape[1]):
            print("Photon number:", photonNum)
            print("sds_{} cv variation: {}".format(detectorIdx, cv[:, detectorIdx]), end="\n\n")
            plt.plot(photonNum, cv[:, detectorIdx], marker="o", label="sds_{}".format(detectorIdx))
        plt.xscale("log")
        plt.yscale("log")
        # plt.yticks(cv, ["{:.0%}".format(y) for y in cv])
        plt.legend()
        plt.xlabel("Photon number")
        plt.ylabel("Estimated coefficient of variation")
        plt.title("Estimated coefficient of variation against photon number")
        plt.show()
    
    groupingNum = int(reflectance.shape[0] / 10)  # 10 is the cv calculation base
    # grouping reflectance
    reflectance = reflectance.reshape(groupingNum, 10, detNum)  # 10 is the cv calculation base
    # compress reflectance and calculate mean of grouping
    reflectance = reflectance.mean(axis=0)
    # calculate real mean and cv for 10 times
    reflectanceMean = reflectance.mean(axis=0)
    reflectanceCV = reflectance.std(axis=0, ddof=1) / reflectanceMean
    
    return reflectance, reflectanceMean, reflectanceCV, info["TotalPhoton"], groupingNum


# %%
if __name__ == "__main__":
    analyzeReflectance("extended_prism", showCvVariation=True)
    
    
    
    