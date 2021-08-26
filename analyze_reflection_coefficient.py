# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 00:06:58 2021

@author: Eric
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
plt.close("all")
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% Function definition
def plotReflectionCoeff(incidentMaterials, refractiveMaterials):
    angLowerLim = 0
    angUpperLim = np.pi/2
    incidentAngs = np.arange(angLowerLim, angUpperLim, step=(angUpperLim-angLowerLim)/1000)
    # incidentAngs = angUpperLim * 0.9
    for incMat in incidentMaterials:
        for refMat in refractiveMaterials:
            reflectionCoeffs = calReflectionCoeff(incidentAngs, incMat["n"], refMat["n"])
            plt.plot(incidentAngs, reflectionCoeffs, label="n0={}, n1={} ({} to {})".format(incMat["n"], refMat["n"], incMat["Name"], refMat["Name"]))
    plt.axvline(x=angUpperLim, c='k', linestyle=':')
    # plt.text(1.48, 0.3, 'x = pi/2', fontsize=13, rotation=90)
    plt.xticks(np.linspace(angLowerLim, angUpperLim, num=10), np.linspace(np.rad2deg(angLowerLim), np.rad2deg(angUpperLim), num=10).astype(int))
    plt.legend()
    plt.xlabel("Incident angle [degree]")
    plt.ylabel("Reflection coefficient")
    plt.title("Reflection coefficient against incident angle in different materials")
    plt.show()


def calReflectionCoeffOfDiffuseLight(incidentMaterials, refractiveMaterials):
    angLowerLim = 0
    angUpperLim = np.pi/2
    for incMat in incidentMaterials:
        for refMat in refractiveMaterials:
            rd, err = quad(integrand, 
                           angLowerLim, angUpperLim, 
                           args=(incMat["n"], refMat["n"])
                           )
            print("{}(n:{}) to {}(n:{})  ==>  rd: {}".format(incMat["Name"], incMat["n"], refMat["Name"], refMat["n"], rd))


def integrand(theta, n1, n2):
    return 2*calReflectionCoeff(theta, n1, n2)*np.cos(theta)*np.sin(theta)


def calReflectionCoeff(incidentAng, n1, n2):
    """  
    
    Parameters
    ----------
    n1 : float
        refractive index of incident material.
    n2 : float
        refractive index of refractive material.
    incidentAng : float [rad]
        incident angle.

    Returns
    -------
    r : float
        reflection coefficient corresponding to each incident angle.

    """
    # total internal reflection may happen ?
    if n1 > n2:
        critAng = np.arcsin(n2/n1)
    else:
        critAng = np.inf
    # convert data type
    if type(incidentAng) != np.ndarray:
        incidentAng = np.array([incidentAng])
    # calculate the number of incident angles which are larger than critical angle
    largeIncidentAngNum = sum(incidentAng>=critAng)
    # choose incident angles which are smaller than critical angle
    incidentAng = incidentAng[incidentAng<critAng]
    # calculate refractive angle and final reflection coefficient
    refractiveAng = np.arcsin(n1/n2*np.sin(incidentAng))
    Rs = ( (n1*np.cos(incidentAng)-n2*np.cos(refractiveAng)) / (n1*np.cos(incidentAng)+n2*np.cos(refractiveAng)) )**2
    Rp = ( (n1*np.cos(refractiveAng)-n2*np.cos(incidentAng)) / (n1*np.cos(refractiveAng)+n2*np.cos(incidentAng)) )**2
    r = (Rs+Rp)/2
    # back-fill the reflection coefficient "1" (for those larger than critical angle)
    r = np.append(r, np.ones(largeIncidentAngNum))
    return r


# %% Run
if __name__ == "__main__":
    incidentMaterials = [{"Name": "Skin", "n": 1.4}]
    refractiveMaterials = [{"Name": "Air", "n": 1},
                            {"Name": "Water", "n": 1.33},
                            {"Name": "PLA", "n": 1.45},
                            {"Name": "Prism", "n": 1.51}
                           ]
    plotReflectionCoeff(incidentMaterials, refractiveMaterials)
    calReflectionCoeffOfDiffuseLight(incidentMaterials, refractiveMaterials)