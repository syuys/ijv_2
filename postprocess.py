#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:24:45 2021

@author: md703
"""

# from IPython import get_ipython
# get_ipython().magic('clear')
# get_ipython().magic('reset -f')
import numpy as np
from scipy import stats
from scipy.signal import convolve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.close("all")
import jdata as jd
import os
from glob import glob
import json
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

def plotIntstDistrb(projectID, sessionID):
    # read position of source and radius of irradiated window
    with open(os.path.join("/home/md703/syu/ijv_2_output", projectID, sessionID, "json_output", f"input_{sessionID}.json")) as f:
        mcxInput = json.load(f)
    # print(mcxInput["Optode"]["Source"]["Pos"])
    srcPos = np.round(mcxInput["Optode"]["Source"]["Pos"]).astype(int)  # srcPos can be converted to integer although the z value may be 19.9999
    winRadius = int(mcxInput["Optode"]["Source"]["Param2"][2]) # winRadius can be converted to integer
    # glob all flux output and read
    fluxOutputPathSet = glob(os.path.join("/home/md703/syu/ijv_2_output", projectID, sessionID, "mcx_output", "*.jnii"))
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
    print(f"data shape: {data.shape}")
    data = data.mean(axis=0)  # average w.r.t all output
    zDistrb = data.sum(axis=(0, 1))
    # plot distribution along depth
    plt.plot(zDistrb[:len(zDistrb)//6], marker=".")
    plt.xlabel("Depth [grid]")
    plt.ylabel("Energy density [-]")
    plt.title("Along Z axis")
    # plt.title("Distribution of intensity along Z axis")
    plt.show()
    
    # retrieve the distribution in the first skin layer
    xyDistrb = data[:, :, srcPos[2]]
    xyDistrb = xyDistrb / xyDistrb.max()  # normalization for this surface
    # retrieve the distribution in the first skin layer near source
    xyDistrbFocusCenter = xyDistrb[srcPos[0]-int(1.25*winRadius):srcPos[0]+int(1.25*winRadius),
                                   srcPos[1]-int(1.25*winRadius):srcPos[1]+int(1.25*winRadius)]
    
    # plot distribution in the first skin layer
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(xyDistrb.T, cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xticks(np.linspace(-0.5, xyDistrb.shape[0]-0.5, num=5), 
               np.linspace(-xyDistrb.shape[0]*voxelSize/2, xyDistrb.shape[0]*voxelSize/2, num=5))
    ax.set_yticks(np.linspace(-0.5, xyDistrb.shape[1]-0.5, num=5), 
               np.linspace(-xyDistrb.shape[1]*voxelSize/2, xyDistrb.shape[1]*voxelSize/2, num=5))
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_title("Top view (whole)")
    # plt.title("Distribution of normalized intensity in the first skin layer")
    plt.show()
    
    # plot distribution in the first skin layer near source
    print(f"shape of near source: {xyDistrbFocusCenter.shape}")
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
    
    # plot distribution in the first skin layer near source (add circle edge)
    origin = xyDistrbFocusCenter.shape[0]//2 - 0.5
    r = 8
    theta = np.arange(0, 2*np.pi, 2*np.pi/1000)
    x = r * np.cos(theta) + origin
    y = r * np.sin(theta) + origin
    
    print(f"shape of near source: {xyDistrbFocusCenter.shape}")
    plt.imshow(xyDistrbFocusCenter.T, cmap="jet")
    plt.colorbar()
    plt.plot(origin, origin, marker="x", color="white")
    plt.plot(x, y, color="white")
    plt.xticks(np.linspace(-0.5, xyDistrbFocusCenter.shape[0]-0.5, num=5), 
               np.linspace(-xyDistrbFocusCenter.shape[0]*voxelSize/2, xyDistrbFocusCenter.shape[0]*voxelSize/2, num=5))
    plt.yticks(np.linspace(-0.5, xyDistrbFocusCenter.shape[1]-0.5, num=5), 
               np.linspace(-xyDistrbFocusCenter.shape[1]*voxelSize/2, xyDistrbFocusCenter.shape[1]*voxelSize/2, num=5))
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Top view (near source)")
    # plt.title("Distribution of normalized intensity in the first skin layer near source")
    plt.show()
    
    # plot intensity trend in the first skin layer near source
    plt.plot(xyDistrbFocusCenter.T[xyDistrbFocusCenter.shape[0]//2-1], marker=".", linestyle="-")
    plt.xticks(np.linspace(-0.5, xyDistrbFocusCenter.shape[0]-0.5, num=5), 
               np.linspace(-xyDistrbFocusCenter.shape[0]*voxelSize/2, xyDistrbFocusCenter.shape[0]*voxelSize/2, num=5))
    plt.xlabel("X [mm]")
    plt.ylabel("Normalized intensity [-]")
    plt.title("Distribution of normalized intensity in the first skin layer near source")
    plt.show()
    
    return xyDistrbFocusCenter.T
    


def updateReflectance(sessionID, muaPathSet, detectorNA):
    # new reflectance
    raw, maR, maRM, maRCV, photon, groupingNum = analyzeReflectance(sessionID, muaPathSet=muaPathSet, detectorNA=detectorNA, updateResultFile=False)
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)
    # for tracing every cv
    cvSession = []
    # conduct core-function
    for muaIdx, muaPath in enumerate(muaPathSet):
        muaType = muaPath.split("/")[-1][:-5]
        # old reflectance
        with open(os.path.join(config["OutputPath"], sessionID, "post_analysis", f"{sessionID}_simulation_result_{muaType}.json")) as f:
            result = json.load(f)
        oldGroupingNum = result["GroupingNum"]
        print(f"\n{muaType}: weighted-num (old, new) = ({oldGroupingNum}, {groupingNum})\n")
        # update reflectance
        cvMua = []
        for detectorIdx, (sds, samplevalues) in enumerate(result["MovingAverageGroupingSampleValues"].items()):
            # weighted-average  
            # print(f"samplevalues: {samplevalues.shape},  maR: {maR.shape}")
            samplevalues = (np.array(samplevalues)*oldGroupingNum + maR[:, detectorIdx, muaIdx]*groupingNum) / (oldGroupingNum + groupingNum)            
            # update
            result["MovingAverageGroupingSampleValues"][sds] = samplevalues.tolist()
            result["MovingAverageGroupingSampleStd"][sds] = samplevalues.std(ddof=1)
            result["MovingAverageGroupingSampleMean"][sds] = samplevalues.mean()
            cv = samplevalues.std(ddof=1) / samplevalues.mean()
            result["MovingAverageGroupingSampleCV"][sds] = cv
            cvMua.append(cv)
        cvSession.append(cvMua)
        # update other infos
        result["AnalyzedSampleNum"] = result["AnalyzedSampleNum"] + raw.shape[0]
        result["GroupingNum"] = oldGroupingNum + groupingNum
        result["PhotonNum"]["GroupingSample"] = "{:.4e}".format(config["PhotonNum"]*result["GroupingNum"])
        # save
        with open(os.path.join(config["OutputPath"], sessionID, "post_analysis", f"{sessionID}_simulation_result_{muaType}.json"), "w") as f:
            json.dump(result, f, indent=4)
    
    return np.array(cvSession)


def analyzeReflectance(sessionID, muaPathSet, detectorNA, updateResultFile=True, showCvVariation=False):    
    # read files
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)  # about detector na, & photon number
    with open(os.path.join(sessionID, "model_parameters.json")) as f:
        modelParameters = json.load(f)  # about index of materials & fiber number
    fiberSet = modelParameters["HardwareParam"]["Detector"]["Fiber"]
    
    # 20221208 add. if need to recalculate reflectance from jdata in 5TB disk, uncomment the following 2 lines of code, and type in preferred project name
    # projectName = "20230715_contrast_invivo_geo_simulation"
    # config["OutputPath"] = f"/media/md703/Expansion/syu/ijv_2_output/{projectName}"
    
    detOutputPathSet = glob(os.path.join(config["OutputPath"], sessionID, "mcx_output", "*.jdat"))  # about paths of detected photon data
    
    # main
    # sort (to make calculation of cv be consistent in each time)
    detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
    
    # for convenience of compressing and calculating cv, remove some output
    cvSampleNum = 10  # the cv calculation base number
    if len(detOutputPathSet) // cvSampleNum == 0:  # if number of detOutput < 10
        cvSampleNum = len(detOutputPathSet)
    else:  # if number of detOutput > 10
        mod = len(detOutputPathSet) % cvSampleNum
        if mod != 0:
            del detOutputPathSet[-mod:]
    
    # organize mua
    mua = []
    for muaPath in muaPathSet:
        with open(muaPath) as f:
            tmp = json.load(f)
        if config["Type"] == "ijv":
            mua.append([tmp["1: Air"],
                        tmp["2: PLA"],
                        tmp["3: Prism"],
                        tmp["4: Skin"],
                        tmp["5: Fat"],
                        tmp["6: Muscle"],
                        tmp["7: Muscle or IJV (Perturbed Region)"],
                        tmp["8: IJV"],
                        tmp["9: CCA"]
                        ])
            
        elif config["Type"] == "ijv_cca_both_pulse":
            mua.append([tmp["1: Air"],
                        tmp["2: PLA"],
                        tmp["3: Prism"],
                        tmp["4: Skin"],
                        tmp["5: Fat"],
                        tmp["6: Muscle"],
                        tmp["7: IJV"],
                        tmp["8: CCA"]
                        ])
        else:
            raise Exception("config type is not valid !!")
    
    mua = np.array(mua).T
    
    # get reflectance
    if config["Type"] == "ijv":
        reflectance = getReflectance(mua=mua,
                                     innerIndex=modelParameters["OptParam"]["Prism"]["n"], 
                                     outerIndex=modelParameters["OptParam"]["Fiber"]["n"], 
                                     detectorNA = detectorNA, 
                                     detectorNum=len(fiberSet)*3*2, 
                                     detOutputPathSet=detOutputPathSet,
                                     photonNum = config["PhotonNum"])
    if config["Type"] == "phantom":
        reflectance = getReflectance(mua=mua,
                                     innerIndex=modelParameters["OptParam"]["Prism"]["n"], 
                                     outerIndex=modelParameters["OptParam"]["Fiber"]["n"], 
                                     detectorNA=config["DetectorNA"], 
                                     detectorNum=len(fiberSet)*3, 
                                     detOutputPathSet=detOutputPathSet,
                                     photonNum = config["PhotonNum"])
    
    ## Calculate final CV
    finalGroupingNum = int(reflectance.shape[0] / cvSampleNum)
    # for idx, muaPath in enumerate(muaPathSet):        
    # grouping reflectance and compress (calculate mean of grouping). [cvSampleNum, detectorNum, muaNum]
    finalReflectance = reflectance.reshape(finalGroupingNum, cvSampleNum, reflectance.shape[1], reflectance.shape[-1]).mean(axis=0)
    # calculate real mean and cv for [cvSampleNum] times
    finalReflectanceStd = finalReflectance.std(axis=0, ddof=1)
    finalReflectanceMean = finalReflectance.mean(axis=0)
    finalReflectanceCV = finalReflectanceStd / finalReflectanceMean
    # arange detectors
    if config["Type"] == "ijv":
        # arange and fold detectors. [cvSampleNum, detectorNum, 3 (width), muaNum]
        movingAverageFinalReflectance = finalReflectance.reshape(finalReflectance.shape[0], 
                                                                 -1, 
                                                                 3, 
                                                                 2, # symmetric to source
                                                                 finalReflectance.shape[-1]).mean(axis=-2)  # mean w.r.t symmetry
        
    if config["Type"] == "phantom":
        movingAverageFinalReflectance = finalReflectance.reshape(finalReflectance.shape[0], -1, 3)
    # movingAverageFinalReflectance = movingAverageFinalReflectance.mean(-1)
    # do moving-average, sds number decrease 2. [cvSampleNum, detectorNumLength-2, muaNum]
    movingAverageFinalReflectance = movingAverage2D(movingAverageFinalReflectance, width=3).squeeze(axis=-2)
    # calculate statistics
    movingAverageFinalReflectanceStd = movingAverageFinalReflectance.std(axis=0, ddof=1)  # [detectorNumLength-2, muaNum]
    movingAverageFinalReflectanceMean = movingAverageFinalReflectance.mean(axis=0)  # [detectorNumLength-2, muaNum]
    movingAverageFinalReflectanceCV = movingAverageFinalReflectanceStd / movingAverageFinalReflectanceMean  # [detectorNumLength-2, muaNum]
    
    # save calculation result after grouping
    if updateResultFile:
        for muaIdx, muaPath in enumerate(muaPathSet):
            muaType = muaPath.split("/")[-1][:-5]
            resultfile = f"{sessionID}_simulation_result_{muaType}.json"
            if os.path.isfile(os.path.join(config["OutputPath"], sessionID, "post_analysis", resultfile)):
                with open(os.path.join(config["OutputPath"], sessionID, "post_analysis", resultfile)) as f:
                    result = json.load(f)
            else:
                rndpath = glob(os.path.join(config["OutputPath"], sessionID, "post_analysis", "*"))[0]
                with open(rndpath) as f:
                    result = json.load(f)
            result["AnalyzedSampleNum"] = reflectance.shape[0]
            result["GroupingNum"] = finalGroupingNum
            result["PhotonNum"]["GroupingSample"] = "{:.4e}".format(config["PhotonNum"]*finalGroupingNum)
            # result["GroupingSampleValues"] = {"sds_{}".format(detectorIdx): finalReflectance[:, detectorIdx].tolist() for detectorIdx in range(finalReflectance.shape[1])}
            # result["GroupingSampleStd"] = {"sds_{}".format(detectorIdx): finalReflectanceStd[detectorIdx] for detectorIdx in range(finalReflectanceStd.shape[0])}
            # result["GroupingSampleMean"] = {"sds_{}".format(detectorIdx): finalReflectanceMean[detectorIdx] for detectorIdx in range(finalReflectanceMean.shape[0])}
            # result["GroupingSampleCV"] = {"sds_{}".format(detectorIdx): finalReflectanceCV[detectorIdx] for detectorIdx in range(finalReflectanceCV.shape[0])}
            result["MovingAverageGroupingSampleValues"] = {"sds_{}".format(fiberSet[detectorIdx+1]["SDS"]): movingAverageFinalReflectance[:, detectorIdx, muaIdx].tolist() for detectorIdx in range(movingAverageFinalReflectance.shape[1])}
            result["MovingAverageGroupingSampleStd"] = {"sds_{}".format(fiberSet[detectorIdx+1]["SDS"]): movingAverageFinalReflectanceStd[detectorIdx, muaIdx] for detectorIdx in range(movingAverageFinalReflectanceStd.shape[0])}
            result["MovingAverageGroupingSampleMean"] = {"sds_{}".format(fiberSet[detectorIdx+1]["SDS"]): movingAverageFinalReflectanceMean[detectorIdx, muaIdx] for detectorIdx in range(movingAverageFinalReflectanceMean.shape[0])}
            result["MovingAverageGroupingSampleCV"] = {"sds_{}".format(fiberSet[detectorIdx+1]["SDS"]): movingAverageFinalReflectanceCV[detectorIdx, muaIdx] for detectorIdx in range(movingAverageFinalReflectanceCV.shape[0])}
            with open(os.path.join(config["OutputPath"], sessionID, "post_analysis", resultfile), "w") as f:
                json.dump(result, f, indent=4)
            if "media" in config["OutputPath"]:
                with open(os.path.join(f"/home/md703/syu/ijv_2_output/{projectName}", sessionID, "post_analysis", resultfile), "w") as f:
                    json.dump(result, f, indent=4)
    
    # plot cv variation curve.
    if showCvVariation:
        baseNum = 5
        analyzeNum = int(np.ceil(np.log(reflectance.shape[0]/cvSampleNum)/np.log(baseNum)))  # follow logarithm change of base rule
        photonNum = []
        cv = []
        for i in range(analyzeNum):
            groupingNum = baseNum ** i
            sample = reflectance[:groupingNum*cvSampleNum].reshape(groupingNum, cvSampleNum, len(fiberSet))
            sample = sample.mean(axis=0)
            sampleMean = sample.mean(axis=0)
            sampleStd = sample.std(axis=0, ddof=1)
            sampleCV = sampleStd / sampleMean
            photonNum.append(config["PhotonNum"] * groupingNum)
            cv.append(sampleCV)
        # add final(overall) cv
        photonNum.append(config["PhotonNum"] * finalGroupingNum)
        cv.append(finalReflectanceCV)
        # print(cv, end="\n\n\n")
        # plot
        cv = np.array(cv)
        for detectorIdx in range(cv.shape[1]):
            print("Photon number:", ["{:.4e}".format(prettyPhotonNum) for prettyPhotonNum in photonNum])
            print("sds_{} cv variation: {}".format(detectorIdx, cv[:, detectorIdx]), end="\n\n")
            plt.plot(photonNum, cv[:, detectorIdx], marker="o", label="sds {:.1f} mm".format(fiberSet[detectorIdx]["SDS"]))
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(photonNum, ["{:.2e}".format(x) for x in photonNum], rotation=-90)
        yticks = plt.yticks()[0][1:-1]
        plt.yticks(yticks, ["{:.2%}".format(ytick) for ytick in yticks])
        plt.legend()
        plt.xlabel("Photon number")
        plt.ylabel("Estimated coefficient of variation")
        plt.title("Estimated coefficient of variation against photon number")
        plt.show()
    
    return reflectance, movingAverageFinalReflectance, movingAverageFinalReflectanceMean, movingAverageFinalReflectanceCV, config["PhotonNum"], finalGroupingNum


def getReflectance(mua, innerIndex, outerIndex, detectorNA, detectorNum, detOutputPathSet, photonNum):    
    # analyze detected photon
    reflectance = np.empty((len(detOutputPathSet), detectorNum, mua.shape[1]))
    for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
        # # read detected data
        # detOutput = jd.load(detOutputPath)        
        # # trim divergent photon
        # detOutput = trimDivergentPhoton(detOutput, innerIndex, outerIndex, detectorNA)
        
        # read and trim detected data
        detOutput = trimJdata(detOutputPath, innerIndex, outerIndex, detectorNA)
        
        # retrieve detector ID and ppath
        photonData = detOutput["MCXData"]["PhotonData"]
        validDetID = photonData["detid"]
        validPPath = photonData["ppath"]
        # unit conversion for photon pathlength
        info = detOutput["MCXData"]["Info"]
        validPPath = validPPath * info["LengthUnit"]
        
        # calculate reflectance 
        for detectorIdx in range(info["DetNum"]):
            usedValidPPath = validPPath[validDetID[:, 0]==detectorIdx]
            # I = I0 * exp(-mua*L), each detectorIdx element contains different mua-reflectance
            reflectance[detOutputIdx, detectorIdx, :] = getSinglePhotonWeight(usedValidPPath, mua).sum(axis=0) / photonNum
        
        if len(detOutputPathSet) > 500:
            if detOutputIdx % 500 == 0:
                print(f"Ref cal progress ----> {np.around(detOutputIdx/len(detOutputPathSet)*100, 2)}%")
    
    return reflectance


def getSpectrum(mua, photonData, info):        
    # analyze detected photon
    detOutputNum = len(photonData["detid"])
    reflectance = np.empty((detOutputNum, info["DetNum"], mua.shape[1]))
    photonNum = photonData["totalSimPhoton"]
    
    for detOutputIdx in range(detOutputNum):
        # retrieve detector ID and ppath
        validDetID = photonData["detid"][detOutputIdx]
        validPPath = photonData["ppath"][detOutputIdx]
        # # unit conversion for photon pathlength
        # validPPath *= info["LengthUnit"]
        
        # calculate reflectance 
        for detectorIdx in range(info["DetNum"]):
            usedValidPPath = validPPath[validDetID[:, 0]==detectorIdx]
            # I = I0 * exp(-mua*L), each detectorIdx element contains different mua-reflectance
            # before get weight, usedValidPPath do unit conversion
            reflectance[detOutputIdx, detectorIdx, :] = getSinglePhotonWeight(usedValidPPath*info["LengthUnit"], mua).sum(axis=0)
    
    # sum all detOutputIdx and divided by photonNum
    reflectance = reflectance.sum(axis=0) / photonNum
    
    # arange and fold detectors. [detectorNumLength, detectorNumWidth, muaNum]
    reflectance = reflectance.reshape(-1, 
                                      3, 
                                      2, # symmetric to source
                                      reflectance.shape[-1]).mean(axis=-2)  # mean w.r.t symmetry
    print(reflectance.shape)
    reflectance = movingAverage2D(reflectance, width=3).squeeze(axis=-2)
    
    return reflectance


def getMeanPathlength(projectID, sessionID, mua):
    # read files
    with open(os.path.join(projectID, sessionID, "config.json")) as f:
        config = json.load(f)  # about detector na, & photon number    
    with open(os.path.join(projectID, sessionID, "model_parameters.json")) as f:
        modelParameters = json.load(f)  # about index of materials & fiber number
    # detectorNA=config["DetectorNA"]
    detectorNA = 0.22
    # detOutputPathSet = glob(os.path.join(config["OutputPath"], sessionID, "mcx_output", "*.jdat"))[:10]  # about paths of detected photon data
    detOutputPathSet = glob(os.path.join(f"/home/md703/syu/ijv_2_output/{projectID}", sessionID, "mcx_output", "*.jdat"))[:10]  # about paths of detected photon data
    print(len(detOutputPathSet))
    innerIndex=modelParameters["OptParam"]["Prism"]["n"]
    outerIndex=modelParameters["OptParam"]["Fiber"]["n"]
    detectorNum=len(modelParameters["HardwareParam"]["Detector"]["Fiber"])*3*2
    print(detectorNum)
    
    # analyze detected photon
    meanPathlength = np.empty((len(detOutputPathSet), detectorNum, len(mua)))
    for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
        # read detected data
        detOutput = jd.load(detOutputPath)
        info = detOutput["MCXData"]["Info"]
        photonData = detOutput["MCXData"]["PhotonData"]
        
        # unit conversion for photon pathlength
        photonData["ppath"] = photonData["ppath"] * info["LengthUnit"]
        
        # retrieve valid detector ID and valid ppath
        critAng = np.arcsin(detectorNA/innerIndex)
        afterRefractAng = np.arccos(abs(photonData["v"][:, -1]))
        beforeRefractAng = np.arcsin(outerIndex*np.sin(afterRefractAng)/innerIndex)
        validPhotonBool = beforeRefractAng <= critAng
        validDetID = photonData["detid"][validPhotonBool]
        validDetID = validDetID - 1  # make detid start from 0
        validPPath = photonData["ppath"][validPhotonBool]
        
        # calculate mean pathlength        
        for detectorIdx in range(info["DetNum"]):
            # raw pathlength
            usedValidPPath = validPPath[validDetID[:, 0]==detectorIdx]
            # sigma(wi*pi), for i=0, ..., n
            eachPhotonWeight = getSinglePhotonWeight(usedValidPPath, mua)
            if eachPhotonWeight.sum() == 0:
                meanPathlength[detOutputIdx][detectorIdx] = 0
                continue
            eachPhotonPercent = eachPhotonWeight / eachPhotonWeight.sum()
            eachPhotonPercent = eachPhotonPercent.reshape(-1, 1)
            meanPathlength[detOutputIdx][detectorIdx] = np.sum(eachPhotonPercent*usedValidPPath, axis=0)
            print(f"usedValidPPath shape: {usedValidPPath.shape}")
            # meanPathlength[detOutputIdx][detectorIdx] = np.sum(1/usedValidPPath.shape[0]*usedValidPPath, axis=0)
    
    # print(f"meanpathlength shape: {meanPathlength.shape}")
    cvSampleNum = 10
    meanPathlength = meanPathlength.reshape(-1, cvSampleNum, meanPathlength.shape[-2], meanPathlength.shape[-1]).mean(axis=0)
    movingAverageMeanPathlength = meanPathlength.reshape(meanPathlength.shape[0], -1, 3, 2, meanPathlength.shape[-1]).mean(axis=-2)
    movingAverageMeanPathlength = movingAverage2D(movingAverageMeanPathlength, width=3).reshape(movingAverageMeanPathlength.shape[0], -1, movingAverageMeanPathlength.shape[-1])
    
    return meanPathlength, movingAverageMeanPathlength


def getSinglePhotonWeight(ppath, mua):
    """

    Parameters
    ----------
    ppath : TYPE
        pathlength [mm], 2d array.
    mua : TYPE
        absorption coefficient [1/mm], 1d numpy array or list

    Returns
    -------
    weight : TYPE
        final weight of single(each) photon

    """
    # mua = np.array(mua)
    weight = np.exp(-np.matmul(ppath, mua))
    return weight


def movingAverage2D(arr, width):
    if arr.ndim == 3:
        kernel = np.ones((width, width, 1))
    
    # arr: [cvSampleNum, detectorNum (ex: 51), width (ex: 3), muaNum]
    elif arr.ndim == 4:
        kernel = np.ones((1, width, width, 1))
    
    else:
        raise Exception("arr shape is strange !")
    
    return convolve(arr, kernel, "valid") / width**2


def trimJdata(detOutputPath, innerIndex, outerIndex, detectorNA):
    # read detected data
    detOutput = jd.load(detOutputPath)
    
    # retrieve valid detector ID and valid ppath
    detOutput = trimDivergentPhoton(detOutput, innerIndex, outerIndex, detectorNA)
    
    # re-save
    jd.save(jd.encode(detOutput, {'compression':'zlib','base64':1}), detOutputPath)
    
    return detOutput


def trimDivergentPhoton(detOutput, innerIndex, outerIndex, detectorNA):
    """
    cut the photon which exit surface with too large angle,
    (retrieve valid detector ID and valid ppath)
    """ 
    critAng = np.arcsin(detectorNA/innerIndex)
    afterRefractAng = np.arccos(abs(detOutput["MCXData"]["PhotonData"]["v"][:, -1]))
    beforeRefractAng = np.arcsin(outerIndex*np.sin(afterRefractAng)/innerIndex)
    validPhotonBool = beforeRefractAng <= critAng
    detOutput["MCXData"]["PhotonData"]["detid"] = detOutput["MCXData"]["PhotonData"]["detid"][validPhotonBool]
    if detOutput["MCXData"]["PhotonData"]["detid"].min() == 1:
        detOutput["MCXData"]["PhotonData"]["detid"] -= 1  # make detid start from 0
    detOutput["MCXData"]["PhotonData"]["ppath"] = detOutput["MCXData"]["PhotonData"]["ppath"][validPhotonBool]
    detOutput["MCXData"]["PhotonData"]["v"] = detOutput["MCXData"]["PhotonData"]["v"][validPhotonBool]
    if detOutput["MCXData"]["PhotonData"]["v"].shape[1] == 3:
        detOutput["MCXData"]["PhotonData"]["v"] = np.delete(detOutput["MCXData"]["PhotonData"]["v"], [0, 1], 1)
    
    return detOutput


def testReflectanceMean(source1, sdsIdx1, source2, sdsIdx2):
    data1 = source1["GroupingSampleValues"]["sds_{}".format(sdsIdx1)]
    data2 = source2["GroupingSampleValues"]["sds_{}".format(sdsIdx2)]
    tStatistic1, pValue1 = stats.ttest_ind(data1, data2)
    print("Assume equal variance \nt-statistic: {} \np-value: {}".format(tStatistic1, pValue1), end="\n\n")
    tStatistic2, pValue2 = stats.ttest_ind(data1, data2, equal_var=False)
    print("Assume unequal variance \nt-statistic: {} \np-value: {}".format(tStatistic2, pValue2), end="\n\n")
    


# %%
if __name__ == "__main__":
    #### test
    # targetPath = os.path.join("/media/md703/Expansion/syu/ijv_2_output/20221129_contrast_investigate_op_sdsrange_3to40/ijv_col_mus_0%/mcx_output/", 
    #                           "*.jdat")
    # detOutputPathSet = glob(targetPath)
    detOutputPathSet = ["/home/md703/Desktop/c.jdat"]
    # detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
    
    mua = []
    with open("/media/md703/Expansion/syu_disk/ijv_2/20221129_contrast_investigate_op_sdsrange_3to40/ijv_col_mus_0%/mua_0%.json") as f:
        tmp = json.load(f)
    mua.append([tmp["1: Air"],
                tmp["2: PLA"],
                tmp["3: Prism"],
                tmp["4: Skin"],
                tmp["5: Fat"],
                tmp["6: Muscle"],
                tmp["7: Muscle or IJV (Perturbed Region)"],
                tmp["8: IJV"],
                tmp["9: CCA"]
                ])
    mua = np.array(mua).T
    
    reflectance = getReflectance(mua=mua,
                                 innerIndex=1.51, 
                                 outerIndex=1.457, 
                                 detectorNA = 0.59, 
                                 detectorNum=94*3*2, 
                                 detOutputPathSet=detOutputPathSet,
                                 photonNum = 1e8)
    print(reflectance[0, 0, 0])  # 0.00043221909362255466, 6.742888374202875e-06
    
    # #### do t test to infer whether the population means of two simulation are the same.
    # with open("extended_prism_simulation_result.json") as f:
    #     result1 = json.load(f)
    # with open("normal_prism_sds_20_simulation_result.json") as f:
    #     result2 = json.load(f)
    # testReflectanceMean(result1, 1, result2, 0)
    
    #### calculate mean pathlength
    sessionID = "mus_baseline"
    muaPath = "mua.json"
    with open(os.path.join(sessionID, muaPath)) as f:
        mua = json.load(f)
    muaUsed =[mua["1: Air"],
              mua["2: PLA"],
              mua["3: Prism"],
              mua["4: Skin"],
              mua["5: Fat"],
              mua["6: Muscle"],
              mua["7: Muscle or IJV (Perturbed Region)"],
              mua["8: IJV"],
              mua["9: CCA"]
              ]
    meanPathlength, movingAverageMeanPathlength = getMeanPathlength(sessionID, mua=muaUsed)
    
    
    
    
