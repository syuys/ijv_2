#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 21:18:25 2021

@author: md703
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
import jdata as jd

mua = 0.05
detpPath = "mcx/example/quicktest/grid2x_detp.jdat"
detp = jd.load(detpPath)

w0 = detp["MCXData"]["PhotonData"]["w0"]
ppath = detp["MCXData"]["PhotonData"]["ppath"] * 0.5
totalPhoton = detp["MCXData"]["Info"]["TotalPhoton"]

detweight = w0 * np.exp(-mua*ppath)
reflectance = detweight.sum()/totalPhoton
print("Reflectance: {:5e}".format(reflectance))