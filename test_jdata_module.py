#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 23:28:07 2021

@author: md703
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import json
import jdata as jd


# %% parameter setting
testPath = "ultrasound_image_processing/testIJV_mcxlab.json"


# %% read file
test = jd.load(testPath)


# %%
test = jd.encode(test, {'compression':'zlib','base64':1})
# # with open("ultrasound_image_processing/material.json", "w") as f:
# #     json.dump(material, f, indent=4)
jd.save(test, "ultrasound_image_processing/material.json")