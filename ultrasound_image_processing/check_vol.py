#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 23:25:31 2022

@author: md703
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

volPath = "20221218_contrast_investigate_ijvdepth_sdsrange_3to40/perturbed_tinyHolder_contrast_sds_3to40_ijv_depth_+2_std.npy"
vol = np.load(volPath)

plt.imshow(vol[vol.shape[0]//2, :, :].T)
plt.colorbar()
plt.title(volPath.split(".")[0])
plt.show()