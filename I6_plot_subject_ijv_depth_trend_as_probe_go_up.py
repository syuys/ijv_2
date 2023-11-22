#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:50:41 2023

@author: md703
"""

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.interpolate import interp1d
import utils
import os
from glob import glob
import json
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% plot

shift = [0, 10, 20]
BY = [13.738-5.953, 15.825-4.624, 15.797-4.662]
EU = [(7.046+7.167)/2, 15.146-2.585, 14.418-2.609]
HY = [(5.942+6.087)/2, (8.398+8.841)/2, 12.439-2.657]

plt.figure(figsize=(4.1, 2.6))
plt.plot(shift, BY, "-o", label="BY")
plt.plot(shift, EU, "-o", label="EU")
plt.plot(shift, HY, "-o", label="HY")
plt.grid(visible=False)
# plt.xticks(shift)
plt.legend(edgecolor="black", fontsize="small")
plt.gca().invert_yaxis()
plt.xlabel("Ultrasound probe position from clavicle [mm]")
plt.ylabel("IJV upper edge to skin [mm]")
# plt.title("IJV depth trend")
plt.show()

