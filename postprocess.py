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
    with open(os.path.join(sessionID, "output", "json_output", "input_900.json")) as f:
        mcxInput = json.load(f)
    srcPos = np.round(mcxInput["Optode"]["Source"]["Pos"]).astype(int)  # srcPos can be converted to integer although the z value may be 19.9999
    winRadius = int(mcxInput["Optode"]["Source"]["Param2"][2]) # winRadius can be converted to integer
    # glob all flux output and read
    fluxOutputPathSet = glob(os.path.join(sessionID, "output", "mcx_output", "*.jnii"))
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


def analyzeReflectance(sessionID, mua, showCvVariation=False):    
    # read files
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)  # about detector na, & photon number
    with open(os.path.join(sessionID, "model_parameters.json")) as f:
        modelParameters = json.load(f)  # about index of materials & fiber number
    fiberSet = modelParameters["HardwareParam"]["Detector"]["Fiber"]
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
    
    # get reflectance
    reflectance = getReflectance(mua=mua,
                                 innerIndex=modelParameters["OptParam"]["Prism"]["n"], 
                                 outerIndex=modelParameters["OptParam"]["Prism"]["n"], 
                                 detectorNA=config["DetectorNA"], 
                                 detectorNum=len(fiberSet)*3*2, 
                                 detOutputPathSet=detOutputPathSet,
                                 photonNum = config["PhotonNum"])
    
    # Calculate final CV
    finalGroupingNum = int(reflectance.shape[0] / cvSampleNum)
    # grouping reflectance and compress, calculate mean of grouping
    finalReflectance = reflectance.reshape(finalGroupingNum, cvSampleNum, reflectance.shape[1]).mean(axis=0)
    # calculate real mean and cv for [cvSampleNum] times
    finalReflectanceStd = finalReflectance.std(axis=0, ddof=1)
    finalReflectanceMean = finalReflectance.mean(axis=0)
    finalReflectanceCV = finalReflectanceStd / finalReflectanceMean
    # doing moving average
    movingAverageFinalReflectance = finalReflectance.reshape(finalReflectance.shape[0], -1, 3, 2).mean(axis=-1)
    movingAverageFinalReflectance = movingAverage2D(movingAverageFinalReflectance, width=3).reshape(movingAverageFinalReflectance.shape[0], -1)
    movingAverageFinalReflectanceStd = movingAverageFinalReflectance.std(axis=0, ddof=1)
    movingAverageFinalReflectanceMean = movingAverageFinalReflectance.mean(axis=0)
    movingAverageFinalReflectanceCV = movingAverageFinalReflectanceStd / movingAverageFinalReflectanceMean
    
    # save calculation result after grouping
    with open(os.path.join(config["OutputPath"], sessionID, "post_analysis", "{}_simulation_result.json".format(sessionID))) as f:
        result = json.load(f)
    result["AnalyzedSampleNum"] = reflectance.shape[0]
    result["GroupingNum"] = finalGroupingNum
    result["PhotonNum"]["GroupingSample"] = "{:.4e}".format(config["PhotonNum"]*finalGroupingNum)
    result["GroupingSampleValues"] = {"sds_{}".format(detectorIdx): finalReflectance[:, detectorIdx].tolist() for detectorIdx in range(finalReflectance.shape[1])}
    result["GroupingSampleStd"] = {"sds_{}".format(detectorIdx): finalReflectanceStd[detectorIdx] for detectorIdx in range(finalReflectanceStd.shape[0])}
    result["GroupingSampleMean"] = {"sds_{}".format(detectorIdx): finalReflectanceMean[detectorIdx] for detectorIdx in range(finalReflectanceMean.shape[0])}
    result["GroupingSampleCV"] = {"sds_{}".format(detectorIdx): finalReflectanceCV[detectorIdx] for detectorIdx in range(finalReflectanceCV.shape[0])}
    result["MovingAverageGroupingSampleValues"] = {"sds_{}".format(fiberSet[detectorIdx+1]["SDS"]): movingAverageFinalReflectance[:, detectorIdx].tolist() for detectorIdx in range(movingAverageFinalReflectance.shape[1])}
    result["MovingAverageGroupingSampleStd"] = {"sds_{}".format(fiberSet[detectorIdx+1]["SDS"]): movingAverageFinalReflectanceStd[detectorIdx] for detectorIdx in range(movingAverageFinalReflectanceStd.shape[0])}
    result["MovingAverageGroupingSampleMean"] = {"sds_{}".format(fiberSet[detectorIdx+1]["SDS"]): movingAverageFinalReflectanceMean[detectorIdx] for detectorIdx in range(movingAverageFinalReflectanceMean.shape[0])}
    result["MovingAverageGroupingSampleCV"] = {"sds_{}".format(fiberSet[detectorIdx+1]["SDS"]): movingAverageFinalReflectanceCV[detectorIdx] for detectorIdx in range(movingAverageFinalReflectanceCV.shape[0])}
    with open(os.path.join(config["OutputPath"], sessionID, "post_analysis", "{}_simulation_result.json".format(sessionID)), "w") as f:
        json.dump(result, f, indent=4)
    
    # if showCvVariation is set "true", plot cv variation curve.
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


def getReflectance(mua, innerIndex, outerIndex, detectorNA, detectorNum, detOutputPathSet, photonNum, photonDataVisible=False):    
    # analyze detected photon
    reflectance = np.empty((len(detOutputPathSet), detectorNum))
    for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
        # read detected data
        detOutput = jd.load(detOutputPath)
        info = detOutput["MCXData"]["Info"]
        if photonDataVisible:
            global photonData
        photonData = detOutput["MCXData"]["PhotonData"]
        
        # unit conversion for photon pathlength
        photonData["ppath"] = photonData["ppath"] * info["LengthUnit"]
        
        # retrieve valid detector ID and valid ppath
        critAng = np.arcsin(detectorNA/innerIndex)
        afterRefractAng = np.arccos(abs(photonData["v"][:, 2]))
        beforeRefractAng = np.arcsin(outerIndex*np.sin(afterRefractAng)/innerIndex)
        validPhotonBool = beforeRefractAng <= critAng
        validDetID = photonData["detid"][validPhotonBool]
        validDetID = validDetID - 1  # make detid start from 0
        validPPath = photonData["ppath"][validPhotonBool]
        
        # calculate reflectance        
        for detectorIdx in range(info["DetNum"]):
            usedValidPPath = validPPath[validDetID[:, 0]==detectorIdx]
            # I = I0 * exp(-mua*L)
            reflectance[detOutputIdx][detectorIdx] = getSinglePhotonWeight(usedValidPPath, mua).sum() / photonNum
    return reflectance


def getMeanPathlength(sessionID, mua):
    # read files
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)  # about detector na, & photon number    
    with open(os.path.join(sessionID, "model_parameters.json")) as f:
        modelParameters = json.load(f)  # about index of materials & fiber number
    detectorNA=config["DetectorNA"]
    detOutputPathSet = glob(os.path.join(config["OutputPath"], sessionID, "mcx_output", "*.jdat"))  # about paths of detected photon data
    innerIndex=modelParameters["OptParam"]["Prism"]["n"]
    outerIndex=modelParameters["OptParam"]["Prism"]["n"]
    detectorNum=len(modelParameters["HardwareParam"]["Detector"]["Fiber"])*3*2
    
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
        afterRefractAng = np.arccos(abs(photonData["v"][:, 2]))
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
    mua = np.array(mua)
    weight = np.exp(-np.matmul(ppath, mua))
    return weight


def movingAverage2D(arr, width):
    if arr.ndim == 3:
        kernel = np.ones((1, width, width))
    elif arr.ndim == 4:
        kernel = np.ones((1, width, width, 1))
    else:
        raise Exception("arr shape is strange !")
    return convolve(arr, kernel, "valid") / width**2


def testReflectanceMean(source1, sdsIdx1, source2, sdsIdx2):
    data1 = source1["GroupingSampleValues"]["sds_{}".format(sdsIdx1)]
    data2 = source2["GroupingSampleValues"]["sds_{}".format(sdsIdx2)]
    tStatistic1, pValue1 = stats.ttest_ind(data1, data2)
    print("Assume equal variance \nt-statistic: {} \np-value: {}".format(tStatistic1, pValue1), end="\n\n")
    tStatistic2, pValue2 = stats.ttest_ind(data1, data2, equal_var=False)
    print("Assume unequal variance \nt-statistic: {} \np-value: {}".format(tStatistic2, pValue2), end="\n\n")
    


# %%
if __name__ == "__main__":
    # analyze reflectance with specific session ID
    # raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = analyzeReflectance(sessionID="test_bc", wl=900, showCvVariation=False)
    
    # # do t test to infer whether the population means of two simulation are the same.
    # with open("extended_prism_simulation_result.json") as f:
    #     result1 = json.load(f)
    # with open("normal_prism_sds_20_simulation_result.json") as f:
    #     result2 = json.load(f)
    # testReflectanceMean(result1, 1, result2, 0)
    
    # calculate mean pathlength
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
    
    
    
    
