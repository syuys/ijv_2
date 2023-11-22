#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:45:03 2023

@author: md703
"""

import numpy as np
import pandas as pd
import os
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

# %% parameters
compare = [
        {
            # "20230616_BY_contrast_trend": "Original",
            "20230703_BY_contrast_trend_upward": "Upward",
            },
        {
            # "20230619_EU_contrast_trend": "Original",
            # "20230630_EU_contrast_trend_repeat0619": "Repeat",
            "20230630_EU_contrast_trend_upward": "Upward"
            },
        {
            # "20230621_HY_contrast_trend": "Original",
            "20230706_HY_contrast_trend_upward": "Upward",
            },
    ]

# %% main
for folderSet in compare:
    for folder, label in folderSet.items():
        subject = folder.split("_")[1]
        sds_con_df = pd.read_csv(os.path.join(folder, f"{subject}_contrast_trend.csv"))
        plt.plot(sds_con_df['SDS [mm]'], sds_con_df['Rmax / Rmin  [-]'] - 1, 
                 label=label, marker=".")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.grid()
    plt.legend()
    plt.xlabel('SDS [mm]')
    plt.ylabel('Rmax/Rmin -1  [-]')
    plt.title(f"Comparison of {subject}'s data")
    plt.show()

plt.figure(figsize=(5.7, 3.5))
for folderSet in compare:
    for folder, label in folderSet.items():
        subject = folder.split("_")[1]
        sds_con_df = pd.read_csv(os.path.join(folder, f"{subject}_contrast_trend.csv"))
        plt.plot(sds_con_df['SDS [mm]'], sds_con_df['Rmax / Rmin  [-]'] - 1, 
                 label=f"Subject: {subject}", marker=".")
# plt.fill_between([plt.xlim()[0], 18], y1=plt.ylim()[0], y2=plt.ylim()[1], alpha=0.5)
# plt.vlines(x=18, ymin=0, ymax=0.06, color="red",
#            linestyle="--", linewidth=2)
# plt.arrow(x=16.5, y=0.04, dx=-5, dy=0, width=.002, 
#           # head_width=.01, head_length=.01, 
#           facecolor='red', edgecolor='none')
plt.text(9.5, 0.045, "SDS ↓, Contrast ↑", color="red",
          verticalalignment='bottom', 
           fontweight=0
          )
plt.ylim(0, plt.ylim()[1])
plt.xlim(plt.xlim()[0], plt.xlim()[1])
plt.fill_between([plt.xlim()[0], 18], y1=plt.ylim()[0], y2=plt.ylim()[1], alpha=0.2)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.grid(visible=False)
plt.legend(edgecolor="black", fontsize="small")
plt.xlabel('SDS [mm]')
plt.ylabel('Contrast  [-]')
# plt.title("Comparison of in-vivo data")
plt.show()
    
    