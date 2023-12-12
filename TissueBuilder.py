# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:52:38 2023

@author: EricSyu
"""

import numpy as np

class TissueBuilder:
    
    def __init__(self, modelX, modelY, modelZ):
        
        # initiaize model
        self.modelX = modelX
        self.modelY = modelY
        self.modelZ = modelZ
        self.vol = np.ones((modelX, modelY, modelZ), dtype=int) # all air '1'
    
    
    def set_srcHolder(self, tag, srcHolderX, srcHolderY, srcHolderZ):
        self.vol[self.modelX//2-srcHolderX//2 : self.modelX//2+srcHolderX//2, 
                 self.modelY//2-srcHolderY//2 : self.modelY//2+srcHolderY//2,
                 :srcHolderZ] = tag
    
    
    def set_srcAir(self, tag, irraWinRadius):
        for x in range(self.modelX//2-irraWinRadius, self.modelX//2+irraWinRadius):
            for y in range(self.modelY//2-irraWinRadius, self.modelY//2+irraWinRadius):
                isDist1 = np.sqrt((self.modelX//2-x)**2 + (self.modelY//2-y)**2) < irraWinRadius
                isDist2 = np.sqrt((self.modelX//2-(x+1))**2 + (self.modelY//2-y)**2) < irraWinRadius
                isDist3 = np.sqrt((self.modelX//2-x)**2 + (self.modelY//2-(y+1))**2) < irraWinRadius
                isDist4 = np.sqrt((self.modelX//2-(x+1))**2 + (self.modelY//2-(y+1))**2) < irraWinRadius
                if isDist1 or isDist2 or isDist3 or isDist4:
                    self.vol[x][y] = tag
    
    
    def set_detHolder(self, tag, srcHolderX, detHolderX, detHolderY, detHolderZ):
        self.vol[self.modelX//2+srcHolderX//2 : self.modelX//2+srcHolderX//2+detHolderX, 
                 self.modelY//2-detHolderY//2 : self.modelY//2+detHolderY//2,
                 :detHolderZ] = tag  # first holder
        self.vol[self.modelX//2-srcHolderX//2-detHolderX : self.modelX//2-srcHolderX//2, 
                 self.modelY//2-detHolderY//2 : self.modelY//2+detHolderY//2,
                 :detHolderZ] = tag  # second holder
    
    
    def set_detPrism(self, tag, srcHolderX, detHolderX, detHolderZ, prismY, prismZ):
        self.vol[self.modelX//2+srcHolderX//2 : self.modelX//2+srcHolderX//2+detHolderX, 
                 self.modelY//2-prismY//2 : self.modelY//2+prismY//2,
                 detHolderZ-prismZ : detHolderZ] = tag  # first prism
        self.vol[self.modelX//2-srcHolderX//2-detHolderX : self.modelX//2-srcHolderX//2, 
                 self.modelY//2-prismY//2 : self.modelY//2+prismY//2,
                 detHolderZ-prismZ : detHolderZ] = tag  # second prism
    
    
    def set_detFiber(self, tag, srcHolderX, detHolderX, detHolderZ, prismY, prismZ):
        self.vol[self.modelX//2+srcHolderX//2 : self.modelX//2+srcHolderX//2+detHolderX, 
                 self.modelY//2-prismY//2 : self.modelY//2+prismY//2,
                 :detHolderZ-prismZ] = tag  # first fiber
        self.vol[self.modelX//2-srcHolderX//2-detHolderX : self.modelX//2-srcHolderX//2, 
                 self.modelY//2-prismY//2 : self.modelY//2+prismY//2,
                 :detHolderZ-prismZ] = tag  # second fiber
    
    
    def set_skin(self, tag, detHolderZ, skinDepth):
        self.vol[:, :, detHolderZ:detHolderZ+skinDepth] = tag
    
    
    def set_fat(self, tag, detHolderZ, skinDepth, fatDepth):
        self.vol[:, :, detHolderZ+skinDepth:detHolderZ+skinDepth+fatDepth] = tag
    
    
    def set_muscle(self, tag, detHolderZ):
        self.vol[:, :, detHolderZ:] = tag
    
    
    def set_vessel(self, tag, majorAxis, minorAxis, shiftY, shiftZ, 
                   detHolderZ, ijvDepth):
        
        for y in range(-majorAxis, majorAxis):
            
            for z in range(-minorAxis, minorAxis):
                
                isDist1 = isInEllipse(y, z, majorAxis, minorAxis)
                isDist2 = isInEllipse(y+1, z, majorAxis, minorAxis)
                isDist3 = isInEllipse(y, z+1, majorAxis, minorAxis)
                isDist4 = isInEllipse(y+1, z+1, majorAxis, minorAxis)
                
                if isDist1 or isDist2 or isDist3 or isDist4:
                    self.vol[:, y+self.modelY//2+shiftY, z+detHolderZ+ijvDepth+shiftZ] = tag
    
    
def isInEllipse(x, y, major, minor):  # if circle, major=minor
    state = x**2/major**2 + y**2/minor**2 < 1
    return state







