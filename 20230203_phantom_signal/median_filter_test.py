#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:24:10 2023

@author: md703
"""

import random
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

x = np.arange(0, 1000, 20)
random.shuffle(x)
plt.plot(x, label="original")

for k in range(3, 22, 6):
    y = signal.medfilt(x, k)
    print(f"y: {y[0]}, {y[-1]}")
    plt.plot(y, label=f"window = {k}")
plt.legend()
plt.show()

x1 = np.array(x)
for _ in range(9):
    x1 = np.vstack((x1, x))
y = np.array(x1)
for idx in range(y.shape[0]):
    y[idx] = signal.medfilt(y[idx], 5)
for yy in y:
    plt.plot(yy)
plt.show()