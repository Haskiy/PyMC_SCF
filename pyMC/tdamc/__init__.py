#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-03-16 18:56:52
LastEditTime: 2022-07-28 10:51:34
LastEditors: Li Hao
Description: Initail file.

FilePath: /pyMC/tdamc/__init__.py
Motto: A + B = C!
'''

from pyMC.tdamc import tdamc_uks
from pyMC.tdamc import tdamc_gks
from pyMC.tdamc import tdalc_uks
from pyMC.tdamc import tdalc_gks
from pyMC.tdamc import tddft_mc_uks
from pyMC.tdamc import tddft_mc_gks

def TDAMC_UKS(mf):
    return tdamc_uks.TDAMC_UKS(mf)
    
def TDAMC_GKS(mf):
    # This part contains the ability of "uks to gks" transform. 
    return tdamc_gks.TDAMC_GKS(mf)

def TDANC_UKS(mf):
    return tdalc_uks.TDALC_UKS(mf)
    
def TDANC_GKS(mf):
    # This part contains the ability of "uks to gks" transform. 
    return tdalc_gks.TDALC_GKS(mf)

def TDDFT_MC_UKS(mf):
    return tddft_mc_uks.TDDFT_MC_UKS(mf)
    
def TDDFT_MC_GKS(mf):
    # This part contains the ability of "uks to gks" transform. 
    return tddft_mc_gks.TDDFT_MC_GKS(mf)