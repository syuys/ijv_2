#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:33:54 2023

@author: md703
"""

import numpy as np
import os
import json
from glob import glob
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

# %% parameters
subjectSet = ["BY", "EU", "HW", "HY", "KB"]
majorchange = []
minorchange = []

for subject in subjectSet:
    with open(os.path.join(subject, f"{subject}_geo.json")) as f:
        geo = json.load(f)
    for neckpos in geo.keys():
        if neckpos != "__comment__":
            if type(geo[neckpos]["IJV"]["MinorAxisChangePct"]) != type(None):
                majorchange.append(geo[neckpos]["IJV"]["MajorAxisChangePct"])
                minorchange.append(geo[neckpos]["IJV"]["MinorAxisChangePct"])

plt.plot(majorchange, marker=".", label="major")
plt.plot(minorchange, marker=".", label="minor")
plt.legend()
plt.show()

plt.scatter(majorchange, minorchange, label="Others")
plt.scatter([0.081], [0.127], label="Used")
plt.legend()
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.xlabel("Change percentage of semi-major axis [-]")
plt.ylabel("Change percentage of semi-minor axis [-]")
plt.show()