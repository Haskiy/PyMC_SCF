#!/usr/bin/env python
'''
Author: Li Hao
Date: 2021-09-19 15:55:00
LastEditTime: 2021-12-24 21:20:20
LastEditors: Li Hao
Description: 
    Generalized Kohn-Sham for Solid Calculations.
FilePath: \pyMC\pbc\__init__.py

    A + B = C!
'''

from pyscf.pbc.dft import rks
from pyMC.pbc import gks
from pyMC.pbc import kgks
from pyMC.pbc import gksm_new
from pyMC.pbc import gksm_ibp
from pyMC.pbc import kgksm_new
from pyMC.pbc import kgksm_ibp
from pyMC.pbc.gen_grid import GLegendreGrids

RKS = rks.RKS
GKS = gks.GKS
GKSM = gksm_new.GKSM_new
GKSM_IBP = gksm_ibp.GKSM_IBP

KGKS = kgks.KGKS
KGKSM = kgksm_new.KGKSM_new
KGKSM_IBP = kgksm_ibp.KGKSM_IBP


def KS(cell, *args, **kwargs): #add by lihao
    return gks.GKS(cell, *args, **kwargs)
RKS.__doc__ = rks.RKS.__doc__

def KKS(cell, *args, **kwargs):#add by lihao
    return kgks.KGKS(cell, *args, **kwargs)
RKS.__doc__ = rks.RKS.__doc__

def KSM(cell, *args, **kwargs): #add by lihao
    return gksm_new.GKSM_new(cell, *args, **kwargs)
RKS.__doc__ = rks.RKS.__doc__

def KSIB(cell, *args, **kwargs): #add by lihao
    return gksm_ibp.GKSM_IBP(cell, *args, **kwargs)
RKS.__doc__ = rks.RKS.__doc__

def KKSM(cell, *args, **kwargs):#add by lihao
    return kgksm_new.KGKSM_new(cell, *args, **kwargs)
RKS.__doc__ = rks.RKS.__doc__

def KSIB(cell, *args, **kwargs): #add by lihao
    return kgksm_ibp.KGKSM_IBP(cell, *args, **kwargs)
RKS.__doc__ = rks.RKS.__doc__
