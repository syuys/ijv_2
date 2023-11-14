#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:47:40 2022

@author: md703
"""

import numpy as np
import json
import os
from glob import glob
import pandas as pd
from scipy.signal import convolve
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

outputPath_1 = "/media/md703/Expansion/syu/ijv_2_output/20230505_contrast_investigate_op_sdsrange_5to45_g99"
outputPath_2 = "/media/md703/Expansion/syu/ijv_2_output/20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99"
sessionIDSet = glob(os.path.join(outputPath_1, "*"))
sessionIDSet = [sessionID.split("/")[-1] for sessionID in sessionIDSet]
tmp = []
for sessionID in sessionIDSet:
    if len(sessionID.split("_")) > 4:
        tmp.append(sessionID)
sessionIDSet = tmp
tissueOrder = ["skin", "fat", "muscle", "blood"]
sessionIDSet.sort(key=lambda x: (tissueOrder.index(x.split("_")[-2]), 
                                 float(x.split("_")[-1][:-1]), 
                                 -ord(x.split("_")[1][0])))
sessionIDSet = np.array(sessionIDSet).reshape(-1, 2)
sdsObservedRangeSet = [
                        # [0, 53],
                        # [0, 34],
                        [0, 42],  # [0, 43],
                        # [0, 29],
                        # [5, 46]
                       ]
colorset = ["royalblue", "orange"]


# %% intialize
resultPath = glob(os.path.join(outputPath_1, sessionIDSet[0, 0], "post_analysis", "*"))[0]
with open(resultPath) as f:
    result = json.load(f)
sdsSet = []
for key in result["MovingAverageGroupingSampleCV"].keys():
    sdsSet.append(float(key[4:]))

flatSDS = []
contrast_mus_all = []
contrast_mua_all = []
win = 2
sensThold = 0.007

# %% PLOT CV and contrast (dis, col in the same plot)  --  mus
for sdsObservedRange in sdsObservedRangeSet:
    targetSdsSet = np.array(sdsSet[sdsObservedRange[0]: sdsObservedRange[1]])
    
    # cv & contrast
    reflectanceSet = {}    
    for musType in sessionIDSet:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        axtmp = ax[0].twinx()
        axtmp_1 = ax[-1].twinx()
        mus = "_".join(musType[0].split("_")[-2:])
        reflectanceSet[mus] = {}
        lns = []
        slopeSet = []
        for idx, sessionID in enumerate(musType):
            resultPathSet = glob(os.path.join(outputPath_1, sessionID, "post_analysis", "*"))
            resultPathSet.sort(key=lambda x: int(x.split("_")[-1].split("%")[0]), reverse=False)
            ijvType = "_".join(sessionID.split("_")[:2])
            reflectanceSet[mus][ijvType] = {}
            for resultPath in resultPathSet:
                with open(resultPath) as f:
                    result = json.load(f)
                photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
                resultPathStrSet = resultPath.split("_")
                resultStrIdx = resultPathStrSet.index("result")
                muaType = "_".join(resultPathStrSet[resultStrIdx+1:])
                muaType = muaType.split(".")[0]
                reflectanceSet[mus][ijvType][muaType] = np.array(list(result["MovingAverageGroupingSampleValues"].values()))
                reflectance = result["MovingAverageGroupingSampleMean"]
                cv = result["MovingAverageGroupingSampleCV"]
                lns += ax[0].plot(targetSdsSet, 
                                  list(reflectance.values())[sdsObservedRange[0]: sdsObservedRange[1]], 
                                  color=colorset[idx], 
                                  label=f"{sessionID.split('_')[1]} - ref")
                lns += axtmp.plot(targetSdsSet, 
                                  list(cv.values())[sdsObservedRange[0]: sdsObservedRange[1]], 
                                  linestyle=":", 
                                  color=colorset[idx], 
                                  label=f"{sessionID.split('_')[1]} - cv")
                
        # added these three lines
        titleSet = sessionID.split('_')[2:]
        titleSet.insert(1, "mus")
        title = '_'.join(titleSet)
        labels = [l.get_label() for l in lns]
        # ax[0].legend(lns, labels, loc="upper center")
        ax[0].legend(lns, labels, loc='upper center')
        axtmp.set_ylabel("CV [-]")
        axtmp.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))            
        ax[0].set_yscale("log")
        ax[0].set_xlabel("SDS [mm]")
        ax[0].set_ylabel("Reflectance mean [-]")
        # ax[0].legend()
        ax[0].set_title(f"{title} - reflectance info")
        
        # analyze cnr
        # rMax = np.array(list(reflectanceSet[mus]["ijv_col"].values()))
        # rMin = np.array(list(reflectanceSet[mus]["ijv_dis"].values()))
        # rMaxMean = rMax.mean(axis=-1)
        # rMinMean = rMin.mean(axis=-1)
        # contrast = rMaxMean / rMinMean
        rMax = np.array(list(reflectanceSet[mus]["ijv_col"].values()))
        rMin = np.array(list(reflectanceSet[mus]["ijv_dis"].values()))
        rMaxMean = rMax.mean(axis=-1)
        rMinMean = rMin.mean(axis=-1)
        contrast = rMax / rMin
        contrastMean = contrast.mean(axis=2)
        contrastStd = contrast.std(ddof=1, axis=2)
        contrastSnr = contrastMean / contrastStd
        contrastCv = contrastStd / contrastMean
        contrastCv /= np.sqrt(10)
        
        lns = []
        print(reflectanceSet[mus]["ijv_col"].keys())
        for idx, key in enumerate(reflectanceSet[mus]["ijv_col"].keys()):
            contrast_local = contrastMean[idx][sdsObservedRange[0]: sdsObservedRange[1]]
            contrast_mus_all.append((title, contrast_local))
            # ax[-1].fill_between(targetSdsSet, 
            #                     contrast_local-contrastStd[idx][sdsObservedRange[0]: sdsObservedRange[1]],
            #                     contrast_local+contrastStd[idx][sdsObservedRange[0]: sdsObservedRange[1]],
            #                     alpha=0.4)
            if ("skin" in key) & ("fat" in key) & ("muscle" in key) & ("ijv" in key) & ("cca" in key):
                labeltype = "all mua " + key.split("_")[-1]
            else:
                labeltype = key
            lns += ax[-1].plot(targetSdsSet, 
                               contrast_local, 
                               label=f"con - {labeltype}", marker=".", linestyle="-")
            lns += axtmp_1.plot(targetSdsSet, contrastCv[idx, sdsObservedRange[0]:sdsObservedRange[1]],
                                label=f"cv - {labeltype}", color=lns[-1].get_color(), linestyle="--")
            if len(sessionID.split("_")) > 4:
                print(sessionID)
                df = np.concatenate((targetSdsSet[:, None], contrast_local[:, None]), axis=1)
                df = pd.DataFrame(df, columns=["SDS [mm]", "Rmax/Rmin [-]"])
                # print(df)
                # df.to_csv(f"contrast_depth_+0_std_mus_{title.split('_')[2]}_mua_{labeltype.split(' ')[2]}.csv",
                #           index=False)
            # if (title == 'skin_mus_100%_fat_100%_muscle_100%_blood_100%') & (labeltype == "all mua 100%"):
                # maxsds = targetSdsSet[np.argmax(contrast_local[:26])]
                # df = pd.read_csv("/home/md703/syu/ijv_2/sim_contrast_trend.csv")
                # df["all mus mua 100% - OptSDS [mm]"] = [np.nan, np.nan, maxsds, np.nan]
                # print(df)
                # df.to_csv(os.path.join("/".join(os.getcwd().split("/")[:-1]), "sim_contrast_trend.csv"), 
                #           index=False)
                
        ax[-1].grid()
        ax[-1].set_xlabel("SDS [mm]")
        ax[-1].set_ylabel("Rmax / Rmin  [-]")
        labels = [l.get_label() for l in lns]
        ax[-1].legend(lns, labels, loc="upper left")
        axtmp_1.set_ylabel("CV [-]")
        axtmp_1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[-1].set_title(f"{title} - contrast")
        
        # show plot of this mus
        plt.tight_layout()
        plt.show()


# %% PLOT CV and contrast (dis, col in the same plot)  --  mua | for analyze tissue's response
musType = ["ijv_dis_depth_+0_std", "ijv_col_depth_+0_std"]
# muaIDSet = glob(os.path.join("/home/md703/syu/ijv_2/20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99/ijv_dis_depth_+0_std", "mua_skin*"))
muaIDSet = ['mua_skin_0%_fat_50%_muscle_50%_ijv_50%_cca_50%.json',
            'mua_skin_100%_fat_50%_muscle_50%_ijv_50%_cca_50%.json',
            'mua_skin_50%_fat_0%_muscle_50%_ijv_50%_cca_50%.json',
            'mua_skin_50%_fat_100%_muscle_50%_ijv_50%_cca_50%.json',
            'mua_skin_50%_fat_50%_muscle_0%_ijv_50%_cca_50%.json',
            'mua_skin_50%_fat_50%_muscle_100%_ijv_50%_cca_50%.json',
            'mua_skin_50%_fat_50%_muscle_50%_ijv_0%_cca_0%.json',
            'mua_skin_50%_fat_50%_muscle_50%_ijv_100%_cca_100%.json']
# muaIDSet.remove('mua_skin_50%_fat_50%_muscle_50%_ijv_0%_cca_50%.json')
# muaIDSet.remove('mua_skin_50%_fat_50%_muscle_50%_ijv_100%_cca_50%.json')
# muaIDSet.remove('mua_skin_50%_fat_50%_muscle_50%_ijv_50%_cca_0%.json')
# muaIDSet.remove('mua_skin_50%_fat_50%_muscle_50%_ijv_50%_cca_100%.json')


for sdsObservedRange in sdsObservedRangeSet:
    targetSdsSet = sdsSet[sdsObservedRange[0]: sdsObservedRange[1]]
    
    # cv & contrast
    reflectanceSet = {}    
    for muaID in muaIDSet:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        axtmp = ax[0].twinx()
        axtmp_1 = ax[-1].twinx()
        
        mua = np.array(muaID.split(".")[0].split("_")[1:]).reshape(-1, 2)
        muabool = (mua[:, 1] == "0%") | (mua[:, 1] == "100%")
        muatissue = mua[:, 0][muabool][0]
        if muatissue == "ijv": muatissue = "blood"            
        muapercent = mua[:, 1][muabool][0]
        mua = "_".join([muatissue, "mua", muapercent])
        
        reflectanceSet[mua] = {}
        lns = []
        for idx, sessionID in enumerate(musType):
            resultPathSet = glob(os.path.join(outputPath_2, sessionID, "post_analysis", 
                                              "_".join([sessionID, "simulation_result", muaID])))
            resultPathSet.sort(key=lambda x: int(x.split("_")[-1].split("%")[0]), reverse=True)
            ijvType = "_".join(sessionID.split("_")[:2])
            reflectanceSet[mua][ijvType] = {}
            for resultPath in resultPathSet:
                with open(resultPath) as f:
                    result = json.load(f)
                photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
                muaType = mua
                # muaType = muaType.split(".")[0]
                reflectanceSet[mua][ijvType][muaType] = np.array(list(result["MovingAverageGroupingSampleValues"].values()))
                reflectance = result["MovingAverageGroupingSampleMean"]
                cv = result["MovingAverageGroupingSampleCV"]
                lns += ax[0].plot(targetSdsSet, 
                                  list(reflectance.values())[sdsObservedRange[0]: sdsObservedRange[1]], 
                                  color=colorset[idx], 
                                  label=f"{sessionID.split('_')[1]} - ref")
                lns += axtmp.plot(targetSdsSet, 
                                  list(cv.values())[sdsObservedRange[0]: sdsObservedRange[1]], 
                                  linestyle=":", 
                                  color=colorset[idx], 
                                  label=f"{sessionID.split('_')[1]} - cv")
                
        # added these three lines
        labels = [l.get_label() for l in lns]
        # ax[0].legend(lns, labels, loc="upper center")
        ax[0].legend(lns, labels, loc='upper center')
        axtmp.set_ylabel("CV [-]")
        axtmp.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[0].set_yscale("log")
        ax[0].set_xlabel("SDS [mm]")
        ax[0].set_ylabel("Reflectance mean [-]")
        # ax[0].legend()
        ax[0].set_title(f"{mua} - reflectance info")
        
        # analyze cnr
        rMax = np.array(list(reflectanceSet[mua]["ijv_col"].values()))
        rMin = np.array(list(reflectanceSet[mua]["ijv_dis"].values()))
        rMaxMean = rMax.mean(axis=-1)
        rMinMean = rMin.mean(axis=-1)
        contrast = rMax / rMin
        contrastMean = contrast.mean(axis=2)
        contrastStd = contrast.std(ddof=1, axis=2)
        contrastSnr = contrastMean / contrastStd
        contrastCv = contrastStd / contrastMean
        
        lns = []
        for idx, key in enumerate(reflectanceSet[mua]["ijv_col"].keys()):
            contrast_local = contrastMean[idx][sdsObservedRange[0]: sdsObservedRange[1]]
            contrast_mua_all.append((mua, contrast_local))            
            
            ax[-1].fill_between(targetSdsSet, 
                                contrast_local-contrastStd[idx][sdsObservedRange[0]: sdsObservedRange[1]],
                                contrast_local+contrastStd[idx][sdsObservedRange[0]: sdsObservedRange[1]],
                                alpha=0.4)
            lns += ax[-1].plot(targetSdsSet, 
                               contrast_local, 
                               label="Rmax/Rmin", marker=".", linestyle="-")
            
            # sens = np.diff(contrast_local) / np.diff(targetSdsSet)
            # sens = convolve(abs(sens), np.ones((win))/win, mode='valid')
            # flatbool = (sens < sensThold) & (contrast_local[:-win] > contrast_local[:-win].mean())
            # flatidx = np.where(flatbool == True)[0]
            # if len(flatidx) != 0:
            #     flatbool[flatidx[0]:flatidx[-1]+win+1] = True
            #     flatsds = np.array(targetSdsSet)[:-win][flatbool]
            #     flatSDS.append((mua, flatsds))
            #     lns += ax[-1].plot(flatsds, contrast_local[:-win][flatbool], 
            #                        label="Flat area", color="red", marker=".", linestyle="-")
            
            lns += axtmp_1.plot(targetSdsSet, contrastCv[idx, sdsObservedRange[0]:sdsObservedRange[1]],
                                label="CV", color="orange", linestyle="--")
            
            
        ax[-1].grid()
        ax[-1].set_xlabel("SDS [mm]")
        ax[-1].set_ylabel("Rmax / Rmin  [-]")
        labels = [l.get_label() for l in lns]
        ax[-1].legend(lns, labels, loc="upper left")
        axtmp_1.set_ylabel("CV [-]")
        axtmp_1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[-1].set_title(f"{mua} - contrast")
        
        # show plot of this mus
        plt.tight_layout()
        plt.show()


# %% final analysis
sessionID = "ijv_dis_depth_+0_std"
with open(os.path.join(f"/media/md703/Expansion/syu/ijv_2_output/20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99/{sessionID}/post_analysis", 
                       f"{sessionID}_simulation_result_mua_skin_50%_fat_50%_muscle_50%_ijv_50%_cca_50%.json")) as f:
    result = json.load(f)
normalRmin = np.array(list(result["MovingAverageGroupingSampleValues"].values()))
sessionID = "ijv_col_depth_+0_std"
with open(os.path.join(f"/media/md703/Expansion/syu/ijv_2_output/20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99/{sessionID}/post_analysis", 
                       f"{sessionID}_simulation_result_mua_skin_50%_fat_50%_muscle_50%_ijv_50%_cca_50%.json")) as f:
    result = json.load(f)
normalRmax = np.array(list(result["MovingAverageGroupingSampleValues"].values()))
normalcon = np.mean(normalRmax / normalRmin, axis=1)[sdsObservedRange[0]: sdsObservedRange[1]]

# %% compare different tissue
rownum = 2
colnum = 2
fig, ax = plt.subplots(rownum, colnum, figsize=(6, 4))
for idx in range(len(contrast_mus_all)//2):
    # plt.figure(figsize=(4.5, 2.8))
    row = idx // rownum
    col = idx % colnum
    # mus
    ln = ax[row, col].plot(targetSdsSet, contrast_mus_all[idx*2][1]-1, linestyle="dashed", 
                  # marker=".", 
             # label=contrast_mus_all[idx*2][0].capitalize(),
             # label=contrast_mus_all[idx*2][0].split("_")[0].capitalize() + " $\mu_s$ " + contrast_mus_all[idx*2][0].split("_")[-1][:-1] + "\%",
             label="$\mu_s$: " + contrast_mus_all[idx*2][0].split("_")[-1][:-1] + "\%")
    ax[row, col].plot(targetSdsSet, contrast_mus_all[idx*2+1][1]-1, linestyle="dashdot", 
             # marker="*", 
             color=ln[0].get_color(), 
             # label=contrast_mus_all[idx*2+1][0].capitalize(),
             # label=contrast_mus_all[idx*2+1][0].split("_")[0].capitalize() + " $\mu_s$ " + contrast_mus_all[idx*2+1][0].split("_")[-1][:-1] + "\%",
             label="$\mu_s$: " + contrast_mus_all[idx*2+1][0].split("_")[-1][:-1] + "\%")
    # mua
    ln = ax[row, col].plot(targetSdsSet, contrast_mua_all[idx*2][1]-1, linestyle="dashed", 
                  # marker=".", 
                  # label=contrast_mua_all[idx*2][0].split("_")[0].capitalize() + " $\mu_a$ " + contrast_mua_all[idx*2][0].split("_")[-1][:-1] + "\%",
                  label="$\mu_a$: " + contrast_mua_all[idx*2][0].split("_")[-1][:-1] + "\%"
             # label=contrast_mua_all[idx*2][0].capitalize()
             )
    ax[row, col].plot(targetSdsSet, contrast_mua_all[idx*2+1][1]-1, linestyle="dashdot", 
             # marker="*", 
             color=ln[0].get_color(), 
             # label=contrast_mua_all[idx*2+1][0].split("_")[0].capitalize() + " $\mu_a$ " + contrast_mua_all[idx*2+1][0].split("_")[-1][:-1] + "\%",
             label="$\mu_a$: " + contrast_mua_all[idx*2+1][0].split("_")[-1][:-1] + "\%")
    # Average
    ax[row, col].plot(targetSdsSet, normalcon-1, linestyle="-", 
             color="black", label="Average")
    ax[row, col].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax[row, col].grid(visible=False)
    
     # text
    ax[row, col].text(0.15, 0.7, contrast_mua_all[idx*2+1][0].split("_")[0].capitalize(), 
                      color="navy",
                      fontsize="xx-large", horizontalalignment='left',
              verticalalignment='bottom', transform=ax[row, col].transAxes,
               weight=500,
              )
    
    # plt.legend(edgecolor="black", fontsize="small")
    if row == 1:
        ax[row, col].set_xlabel("SDS [mm]")
    if col == 0:
        ax[row, col].set_ylabel("Contrast [-]")
    # plt.title("Comparison of contrast")
handles, labels = ax[0, 0].get_legend_handles_labels()
plt.tight_layout()
fig.legend(handles, labels, edgecolor="black", 
           loc='upper center', bbox_to_anchor=(0.54, 1.08),
           ncol=5, fontsize=11)
plt.show()

# %% compare mus and mua
idxSet = [0, 1, 2]  # [0, 1, 2, 3]
colorSet = ["lightcoral", "tab:orange", "tab:brown", "red"]

fig, ax = plt.subplots(rownum, colnum, figsize=(6, 4))

# mus 0%
# plt.figure(figsize=(4.5, 2.8))
for idx in idxSet:
    ax[0, 0].plot(targetSdsSet, contrast_mus_all[idx*2][1]-1, 
             # label=contrast_mus_all[idx*2][0].split("_")[0].capitalize() + " $\mu_s$ " + contrast_mus_all[idx*2][0].split("_")[-1][:-1] + "\%",
             label=contrast_mus_all[idx*2][0].split("_")[0].capitalize(),
             # color=colorSet[idx],
             linestyle="--"
             )
ax[0, 0].plot(targetSdsSet, normalcon-1, linestyle="-", 
          color="black", label="Average")
ax[0, 0].text(0.15, 0.7, "$\mu_s$: 0\%", 
                  color="navy",
                  fontsize="xx-large", horizontalalignment='left',
          verticalalignment='bottom', transform=ax[0, 0].transAxes,
          )
ax[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax[0, 0].grid(visible=False)
# ax[0, 0].legend(edgecolor="black", fontsize="small")
# ax[0, 0].set_xlabel("SDS [mm]")
ax[0, 0].set_ylabel("Contrast [-]")
# plt.title("Comparison of each tissue's mus_0%")
# plt.show()

# mus 100%
# plt.figure(figsize=(4.5, 2.8))
for idx in idxSet:
    ax[1, 0].plot(targetSdsSet, contrast_mus_all[idx*2+1][1]-1, 
             # label=contrast_mus_all[idx*2+1][0].split("_")[0].capitalize() + " $\mu_s$ " + contrast_mus_all[idx*2+1][0].split("_")[-1][:-1] + "\%",
             label=contrast_mus_all[idx*2+1][0].split("_")[0].capitalize(),
             # color=colorSet[idx],
             linestyle="--"
             )
ax[1, 0].plot(targetSdsSet, normalcon-1, linestyle="-", 
          color="black", label="Average")
ax[1, 0].text(0.15, 0.7, "$\mu_s$: 100\%", 
                  color="navy",
                  fontsize="xx-large", horizontalalignment='left',
          verticalalignment='bottom', transform=ax[1, 0].transAxes,
          )
ax[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax[1, 0].grid(visible=False)
# ax[1, 0].legend(edgecolor="black", fontsize="small")
ax[1, 0].set_xlabel("SDS [mm]")
ax[1, 0].set_ylabel("Contrast [-]")
# plt.title("Comparison of each tissue's mus_100%")
# plt.show()

# mua 0%
# plt.figure(figsize=(4.5, 2.8))
for idx in idxSet:
    ax[0, 1].plot(targetSdsSet, contrast_mua_all[idx*2][1]-1, 
             # label=contrast_mua_all[idx*2][0].split("_")[0].capitalize() + " $\mu_a$ " + contrast_mua_all[idx*2][0].split("_")[-1][:-1] + "\%",
             label=contrast_mua_all[idx*2][0].split("_")[0].capitalize(),
             # color=colorSet[idx],
             linestyle="--"
             )
ax[0, 1].plot(targetSdsSet, normalcon-1, linestyle="-", 
          color="black", label="Average")
ax[0, 1].text(0.15, 0.7, "$\mu_a$: 0\%", 
                  color="navy",
                  fontsize="xx-large", horizontalalignment='left',
          verticalalignment='bottom', transform=ax[0, 1].transAxes,
          )
ax[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax[0, 1].grid(visible=False)
# ax[0, 1].legend(edgecolor="black", fontsize="small")
# ax[0, 1].set_xlabel("SDS [mm]")
# ax[0, 1].set_ylabel("Contrast [-]")
# plt.title(f"Comparison of each tissue's {contrast_mua_all[idx*2+1][0].split('_')[-2]}_0%")
# plt.show()

# mua 100%
# plt.figure(figsize=(4.5, 2.8))
for idx in idxSet:
    ax[1, 1].plot(targetSdsSet, contrast_mua_all[idx*2+1][1]-1, 
             # label=contrast_mua_all[idx*2+1][0].split("_")[0].capitalize() + " $\mu_a$ " + contrast_mua_all[idx*2+1][0].split("_")[-1][:-1] + "\%",
             label=contrast_mua_all[idx*2+1][0].split("_")[0].capitalize(),
             # color=colorSet[idx],
             linestyle="--"
             )
ax[1, 1].plot(targetSdsSet, normalcon-1, linestyle="-", 
          color="black", label="Average")
ax[1, 1].text(0.15, 0.7, "$\mu_a$: 100\%", 
                  color="navy",
                  fontsize="xx-large", horizontalalignment='left',
          verticalalignment='bottom', transform=ax[1, 1].transAxes,
          )
ax[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax[1, 1].grid(visible=False)
# ax[1, 1].legend(edgecolor="black", fontsize="small")
ax[1, 1].set_xlabel("SDS [mm]")
# ax[1, 1].set_ylabel("Contrast [-]")
# plt.title(f"Comparison of each tissue's {contrast_mua_all[idx*2+1][0].split('_')[-2]}_100%")
handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, edgecolor="black", 
           loc='upper center', bbox_to_anchor=(0.54, 1.08),
           ncol=5, fontsize=11)
plt.tight_layout()
plt.show()

# %% compare extreme mus mua
# del contrast_mus_all[4]
# del contrast_mus_all[1]
# mua_all = [0, 100]
# mus_all = [0, 100]
# for idx in range(len(mus_all)):
plt.figure(figsize=(4.5, 2.8))
ln1 = plt.plot(targetSdsSet, contrast_mus_all[0][1] - 1, linestyle="--", 
         label="All ($\mu_s, \mu_a$) = (0\%, 0\%)")
ln2 = plt.plot(targetSdsSet, contrast_mus_all[2][1] - 1, linestyle="--", 
         label="All ($\mu_s, \mu_a$) = (0\%, 100\%)")
plt.plot(targetSdsSet, contrast_mus_all[3][1] - 1, color=ln1[0].get_color(),
         label="All ($\mu_s, \mu_a$) = (100\%, 0\%)")
plt.plot(targetSdsSet, contrast_mus_all[5][1] - 1, color=ln2[0].get_color(), 
         label="All ($\mu_s, \mu_a$) = (100\%, 100\%)")
plt.plot(targetSdsSet, normalcon-1, linestyle="-", 
          color="black", label="Average")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.grid(visible=False)
plt.legend(edgecolor="black", fontsize="small")
plt.xlabel("SDS [mm]")
plt.ylabel("Contrast [-]")
plt.show()

    

# intersect sds
# optype, flatsds = zip(*flatSDS)
# sds_repeat = np.intersect1d(flatsds[0], flatsds[1])
# for sds in flatsds[2:]:
#     sds_repeat = np.intersect1d(sds_repeat, sds)
# for idx, flat in enumerate(flatSDS):
#     plt.hlines(idx, flat[1][0], flat[1][-1])
# plt.fill_between(sds_repeat, len(optype)-1, alpha=0.3)
# plt.xlabel("SDS [mm]")
# plt.yticks(np.arange(len(optype)), optype)
# plt.gca().invert_yaxis()
# plt.title("Contrast analysis w.r.t Optical Properties")
# plt.show()


