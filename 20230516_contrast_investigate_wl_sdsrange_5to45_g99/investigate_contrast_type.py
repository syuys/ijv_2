#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:47:40 2022

@author: md703
"""

import numpy as np
import pandas as pd
from scipy import stats
import utils
import json
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams["figure.dpi"] = 600

outputPath = "/media/md703/Expansion/syu/ijv_2_output/20230516_contrast_investigate_wl_sdsrange_5to45_g99"
sessionIDSet = glob(os.path.join(outputPath, "*"))
sessionIDSet = [sessionID.split("/")[-1] for sessionID in sessionIDSet]
sessionIDSet.sort(key=lambda x: (float(x[-5:-2]), -ord(x.split("_")[1][0])))
sessionIDSet = np.array(sessionIDSet).reshape(5, 2)
sdsObservedRangeSet = [
                        [0, 43],
                        # [0, 34],
                        # [0, 41]
                       ]
colorset = ["royalblue", "orange"]


# %% PLOT CV and contrast (dis, col in the same plot)
resultPath = glob(os.path.join(outputPath, sessionIDSet[0, 0], "post_analysis", "*"))[0]
with open(resultPath) as f:
    result = json.load(f)
cv = result["MovingAverageGroupingSampleCV"]
sdsSet = []
for key in cv.keys():
    sdsSet.append(float(key[4:]))


maxSDS = []
for sdsObservedRange in sdsObservedRangeSet: 
    targetSdsSet = sdsSet[sdsObservedRange[0]: sdsObservedRange[1]]
    
    # cv & contrast
    reflectanceSet = {}    
    for musType in sessionIDSet:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        axtmp_0 = ax[0].twinx()
        # axtmp_1 = ax[-1].twinx()
        mus = "_".join(musType[0].split("_")[-2:])
        reflectanceSet[mus] = {}
        lns = []
        for idx, sessionID in enumerate(musType):
            resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*ijv_0.*"))
            resultPathSet.sort(key=lambda x: float(x.split("_")[-2]))
            ijvType = "_".join(sessionID.split("_")[:2])
            reflectanceSet[mus][ijvType] = {}
            for resultPath in resultPathSet:
                with open(resultPath) as f:
                    result = json.load(f)
                photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
                muaType = np.around(float(resultPath.split("_")[-2])*100, 0)
                muaType = str(int(muaType)) + "%"
                reflectanceSet[mus][ijvType][muaType] = np.array(list(result["MovingAverageGroupingSampleValues"].values()))
                reflectance = result["MovingAverageGroupingSampleMean"]
                cv = result["MovingAverageGroupingSampleCV"]
                lns += ax[0].plot(targetSdsSet, 
                                  list(reflectance.values())[sdsObservedRange[0]: sdsObservedRange[1]], 
                                  color=colorset[idx], 
                                  label=f"{sessionID.split('_')[1]} - SijvO2 {muaType}")
                lns += axtmp_0.plot(targetSdsSet, 
                                  list(cv.values())[sdsObservedRange[0]: sdsObservedRange[1]], 
                                  linestyle=":", 
                                  color=colorset[idx], 
                                  label=f"{sessionID.split('_')[1]} - cv")
                
        # added these three lines
        labels = [l.get_label() for l in lns]
        ax[0].legend(lns, labels, loc='upper center')
        axtmp_0.set_ylabel("CV [-]")
            
        ax[0].set_yscale("log")
        ax[0].set_xlabel("SDS [mm]")
        ax[0].set_ylabel("Reflectance Mean [-]")
        ax[0].set_title(f"{sessionID.split('_')[4]} - reflectance info")
        
        # analyze cnr
        rMax = np.array(list(reflectanceSet[mus]["ijv_col"].values()))
        rMin = np.array(list(reflectanceSet[mus]["ijv_dis"].values()))
        contrast = rMax / rMin
        contrastMean = contrast.mean(axis=2)
        contrastStd = contrast.std(ddof=1, axis=2)
        contrastSnr = contrastMean / contrastStd
        contrastCv = contrastStd / contrastMean
        # rMaxMean = rMax.mean(axis=-1)
        # rMinMean = rMin.mean(axis=-1)
        # contrast = rMaxMean / rMinMean
        # rMean = (rMax + rMin) / 2
        # rMeanStd = rMean.std(axis=-1, ddof=1)
        # rMeanMean = rMean.mean(axis=-1)
        # rMeanCV = rMeanStd / rMeanMean
        # cnr = ((rMaxMean - rMinMean) / rMeanMean) / rMeanCV
        
        lns = []
        for idx, key in enumerate(reflectanceSet[mus]["ijv_col"].keys()):
            # for sdsIdx, c in enumerate(contrast[idx, sdsObservedRange[0]:sdsObservedRange[1], :]):
            #     ax[-1].scatter(np.repeat(targetSdsSet[sdsIdx], len(c)), c, marker=".")
            # ax[-1].fill_between(targetSdsSet, 
            #                     contrast[idx, sdsObservedRange[0]:sdsObservedRange[1], :].min(axis=1),
            #                     contrast[idx, sdsObservedRange[0]:sdsObservedRange[1], :].max(axis=1),
            #                     alpha=0.4)
            lns += ax[-1].plot(targetSdsSet, 
                               contrastMean[idx, sdsObservedRange[0]:sdsObservedRange[1]], 
                               color=utils.colorFader(c1="darkred", c2="lightcoral", mix=idx/(len(reflectanceSet[mus]["ijv_col"].keys())-1)), 
                               label=f"SijvO2 {key}", linestyle="-")
            # lns += axtmp_1.plot(targetSdsSet, contrastCv[idx, sdsObservedRange[0]:sdsObservedRange[1]],
            #                     label="CV", color=colorset[1], linestyle="--")
        labels = [l.get_label() for l in lns]
        ax[-1].legend(lns, labels)
        ax[-1].grid()
        # axtmp_1.set_ylabel("CV [-]")
        ax[-1].set_xlabel("SDS [mm]")
        ax[-1].set_ylabel("Rmax / Rmin [-]")
        ax[-1].set_title(f"{sessionID.split('_')[4]} - contrast")
        
        # show plot of this mus
        plt.tight_layout()
        plt.show()
        
        
        # delta contrast / delta so2 (contrast = rmax/rmin or ln(rmax/rmin))
        sijvo2 = [int(key[:2]) for key in reflectanceSet[mus]["ijv_col"].keys()]
        deltaCon = abs(np.diff(contrastMean, axis=0))
        deltaLnCon = abs(np.diff(np.log(contrastMean), axis=0))
        deltasijvo2 = np.diff(sijvo2)[:, None]
        sens = deltaCon / deltasijvo2
        sensLn = deltaLnCon / deltasijvo2
        
        # fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        # for idx in range(sens.shape[0]):
        #     ax[0].plot(targetSdsSet, sens[idx][sdsObservedRange[0]:sdsObservedRange[1]], 
        #              label=f"SijvO2: {sijvo2[idx]}% → {sijvo2[idx+1]}%", 
        #              color=utils.colorFader(c1="darkblue", c2="cornflowerblue", mix=idx/(sens.shape[0]-1)), 
        #              linestyle="-")
        #     ax[-1].plot(targetSdsSet, sensLn[idx][sdsObservedRange[0]:sdsObservedRange[1]], 
        #              label=f"SijvO2: {sijvo2[idx]}% → {sijvo2[idx+1]}%", 
        #              color=utils.colorFader(c1="darkblue", c2="cornflowerblue", mix=idx/(sens.shape[0]-1)), 
        #              linestyle="-")
        # ax[0].grid()
        # ax[0].legend()
        # ax[0].set_xlabel("SDS [mm]")
        # ax[0].set_ylabel("Δ(Rmax/Rmin) / Δ(SijvO2)  [-]")
        # ax[0].set_title(f"{sessionID.split('_')[4]} - Δ(Rmax/Rmin) / Δ(SijvO2)")
        # ax[-1].grid()
        # ax[-1].legend()
        # ax[-1].set_xlabel("SDS [mm]")
        # ax[-1].set_ylabel("Δln(Rmax/Rmin) / Δ(SijvO2)  [-]")
        # ax[-1].set_title(f"{sessionID.split('_')[4]} - Δln(Rmax/Rmin) / Δ(SijvO2)")
        # plt.tight_layout()
        # plt.show()
        
        # if sessionID.split('_')[4] == "760nm" or sessionID.split('_')[4] == "850nm":
        if sessionID.split('_')[4] in ["730nm", "760nm", "780nm", "810nm", "850nm"]:
            # # compare rmax/rmin and sensitivity
            # fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
            # axtwin_0 = ax[0].twinx()
            # axtwin_1 = ax[1].twinx()
            # for idx, key in enumerate(reflectanceSet[mus]["ijv_col"].keys()):
            #     ax[0].plot(targetSdsSet, 
            #                contrastMean[idx, sdsObservedRange[0]:sdsObservedRange[1]], 
            #                color=utils.colorFader(c1="darkred", c2="lightcoral", mix=idx/(len(reflectanceSet[mus]["ijv_col"].keys())-1)), 
            #                label=f"$S_{{ijv}}O_2$: {key}", linestyle="-")
            #     ax[-1].plot(targetSdsSet, 
            #                 contrastMean[idx, sdsObservedRange[0]:sdsObservedRange[1]], 
            #                 color=utils.colorFader(c1="darkred", c2="lightcoral", mix=idx/(len(reflectanceSet[mus]["ijv_col"].keys())-1)), 
            #                 label=f"$S_{{ijv}}O_2$: {key}", linestyle="-")
            #     if idx < len(sens):
            #         axtwin_0.plot(targetSdsSet, sens[idx][sdsObservedRange[0]:sdsObservedRange[1]], 
            #                    label=f"$S_{{ijv}}O_2$: {sijvo2[idx]}% → {sijvo2[idx+1]}%", 
            #                    color=utils.colorFader(c1="darkblue", c2="cornflowerblue", mix=idx/(sens.shape[0]-1)), 
            #                    linestyle="-")
            #         axtwin_1.plot(targetSdsSet, sensLn[idx][sdsObservedRange[0]:sdsObservedRange[1]], 
            #                     label=f"$S_{{ijv}}O_2$: {sijvo2[idx]}% → {sijvo2[idx+1]}%", 
            #                     color=utils.colorFader(c1="darkblue", c2="cornflowerblue", mix=idx/(sensLn.shape[0]-1)), 
            #                     linestyle="-")
            # ax[0].grid()
            # ax[0].legend(loc="upper left", fontsize=9)
            # axtwin_0.legend(loc="upper center", bbox_to_anchor=(0.45, 1), fontsize=8)
            # ax[0].set_xlabel("SDS [mm]")
            # ax[0].set_ylabel("Rmax / Rmin [-]")
            # axtwin_0.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            # axtwin_0.set_ylabel("Δ(Rmax/Rmin) / Δ$S_{ijv}O_2$  [-]")
            # ax[0].set_title(f"{sessionID.split('_')[4]} - comparison")
            # ax[-1].grid()
            # ax[-1].legend(loc="upper left", fontsize=9)
            # axtwin_1.legend(loc="upper center", bbox_to_anchor=(0.45, 1), fontsize=8)
            # ax[-1].set_xlabel("SDS [mm]")
            # ax[-1].set_ylabel("Rmax / Rmin [-]")
            # axtwin_1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            # axtwin_1.set_ylabel("Δln(Rmax/Rmin) / Δ$S_{ijv}O_2$  [-]")
            # ax[-1].set_title(f"{sessionID.split('_')[4]} - comparison")
            # plt.tight_layout()
            # plt.show()
            
            # # check correlation
            # n = sdsObservedRange[1] - sdsObservedRange[0]
            # fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
            # contrastMeanMean = contrastMean[:, sdsObservedRange[0]:sdsObservedRange[1]].mean(axis=0)
            # sensMean = sens[:, sdsObservedRange[0]:sdsObservedRange[1]].mean(axis=0)
            # sensLnMean = sensLn[:, sdsObservedRange[0]:sdsObservedRange[1]].mean(axis=0)
            # r = np.corrcoef(contrastMeanMean, sensMean)
            # print("Δ(Rmax/Rmin) / Δ$S_{ijv}O_2$  [-]")
            # print(r)
            # ax[0].scatter(contrastMeanMean, sensMean, s=9, label=f"n={n}, r={np.around(r[0, 1], 2)}")
            # ax[0].legend()
            # ax[0].grid()
            # ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            # ax[0].set_xlabel("Mean of Rmax/Rmin [-]")
            # ax[0].set_ylabel("Mean of Δ(Rmax/Rmin)/Δ$S_{ijv}O_2$  [-]")
            # ax[0].set_title(f"{sessionID.split('_')[4]} - comparison")
            # r = np.corrcoef(contrastMeanMean, sensLnMean)
            # print("Δln(Rmax/Rmin) / Δ$S_{ijv}O_2$  [-]")
            # print(r)
            # ax[1].scatter(contrastMeanMean, sensLnMean, s=9, label=f"n={n}, r={np.around(r[0, 1], 2)}")
            # ax[1].legend()
            # ax[1].grid()
            # ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            # ax[1].set_xlabel("Mean of Rmax/Rmin [-]")
            # ax[1].set_ylabel("Mean of Δln(Rmax/Rmin)/Δ$S_{ijv}O_2$  [-]")
            # ax[1].set_title(f"{sessionID.split('_')[4]} - comparison")
            # plt.tight_layout()
            # plt.show()
            
            # plot comparison and correlation
            n = sdsObservedRange[1] - sdsObservedRange[0]
            fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.1))
            axtwin_0 = ax[0].twinx()
            for idx, key in enumerate(reflectanceSet[mus]["ijv_col"].keys()):
                ax[0].plot(targetSdsSet, 
                           contrastMean[idx, sdsObservedRange[0]:sdsObservedRange[1]] - 1, 
                           color=utils.colorFader(c1="darkred", c2="lightcoral", mix=idx/(len(reflectanceSet[mus]["ijv_col"].keys())-1)), 
                           label=f"$S_{{ijv}}O_2$: {key[:-1]}" + "\%", linestyle="-")
                if idx < len(sens):
                    axtwin_0.plot(targetSdsSet, sens[idx][sdsObservedRange[0]:sdsObservedRange[1]], 
                               label=f"$S_{{ijv}}O_2$: {sijvo2[idx]}\% → {sijvo2[idx+1]}\%", 
                               color=utils.colorFader(c1="darkblue", c2="cornflowerblue", mix=idx/(sens.shape[0]-1)), 
                               linestyle="-")
            ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            ax[0].grid(visible=False)
            axtwin_0.grid(visible=False)
            ax[0].legend(loc="upper left", bbox_to_anchor=(0.01, 0.98),
                         edgecolor="black", fontsize="x-small")
            axtwin_0.legend(loc="upper center", bbox_to_anchor=(0.44, 0.98), 
                            edgecolor="black", fontsize="x-small")
            ax[0].set_xlabel("SDS [mm]")
            ax[0].set_ylabel("Contrast [-]")
            ax[0].yaxis.label.set_color('darkred')
            ax[0].tick_params(axis='y', colors='darkred')
            axtwin_0.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            axtwin_0.set_ylabel("$\Delta$Contrast / $\Delta$$S_{ijv}O_2$  [-]")
            axtwin_0.yaxis.label.set_color('darkblue')
            axtwin_0.tick_params(axis='y', colors='darkblue')
            ax[0].set_title(f"{sessionID.split('_')[4]} - Comparison")
            
            contrastMeanMean = (contrastMean[:, sdsObservedRange[0]:sdsObservedRange[1]]-1).mean(axis=0)
            sensMean = sens[:, sdsObservedRange[0]:sdsObservedRange[1]].mean(axis=0)
            r = np.corrcoef(contrastMeanMean, sensMean)
            print("Δ(Rmax/Rmin) / Δ$S_{ijv}O_2$  [-]")
            print(r)
            ax[1].scatter(contrastMeanMean, sensMean, s=9, label=f"n={n}, r={np.around(r[0, 1], 2)}")
            ax[1].legend(edgecolor="black")
            ax[1].grid(visible=False)
            ax[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[1].set_xlabel("Mean of Contrast [-]")
            ax[1].set_ylabel("Mean of ($\Delta$Contrast / $\Delta$$S_{ijv}O_2$)  [-]")
            ax[1].set_title(f"{sessionID.split('_')[4]} - Correlation")
            plt.tight_layout()
            plt.show()
            
            
# %% compare contrast variation of different wl
for s in sijvo2:
    plt.figure(figsize=(2.7, 2))
    plt.text(0.1, 0.7, f"$S_{{ijv}}O_2$: {s}\%", 
                      color="navy",
                      fontsize="xx-large", horizontalalignment='left',
              verticalalignment='bottom', transform=plt.gca().transAxes,
               weight=500,
              )
    for idx, (wl, ref) in enumerate(reflectanceSet.items()):
        wlvalue = wl.split("_")[1]
        rmax = ref["ijv_col"][f"{s}%"][sdsObservedRange[0]:sdsObservedRange[1], :].mean(axis=1)
        rmin = ref["ijv_dis"][f"{s}%"][sdsObservedRange[0]:sdsObservedRange[1], :].mean(axis=1)
        con = rmax/rmin
        plt.plot(targetSdsSet, con-1, label=wlvalue, 
                 color=utils.colorFader(c1="blue", c2="red", mix=idx/(len(reflectanceSet.items())-1)))
    plt.grid(visible=False)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.legend(edgecolor="black", loc='upper center', bbox_to_anchor=(0.54, 1.2),
               ncol=5, fontsize="small")
    plt.xlabel("SDS [mm]")
    plt.ylabel("Contrast [-]")
    # plt.title(f"$S_{{ijv}}O_2$ = {s}\%")
    plt.show()
    
        
        
        
