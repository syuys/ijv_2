#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:40:27 2023

@author: md703
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

data = pd.read_csv("Test.csv", usecols = lambda x: "Wavelength (nm)" not in x)
data = data[10:-10]

wl = [float(value) for value in data.columns]

plt.plot(np.diff(wl))
plt.title("wl difference")
plt.show()

for index, spec in data.iterrows():
    plt.plot(wl, spec)
plt.title("spectrum")
plt.show()


