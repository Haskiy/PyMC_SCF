#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-01-18 10:50:15
LastEditTime: 2022-04-10 10:31:37
LastEditors: Li Hao
Description: 
    Analysis.
FilePath: \pyMC\tools\toque_bxc.py

 May the force be with you!
'''

import time
import numpy
import pyscf
import scipy
from pyMC.tools import rotate_dm
from pyMC import gksmc
import scipy.linalg
from pyscf import __config__
from collections import Counter
from scipy.spatial.transform import Rotation as R

def get_m_times_r(coords, M, weights):
    """Calculate the \vec{M} \times \vec{r}

    Args:
        coords (numpy.array): [ngrid,3] coordinates of the grids.
        M (numpy.array): [3,ngrid] \vec{M}. Note: the ndim is different from coords.
        weights (numpy.array): [ngrid] weights of the grids.

    Returns:
        m_times_r (float): \vec{M} \times \vec{r} sum over all grids.
    """
    
    m_times_r = numpy.cross(M.T, coords)
    # m_times_r is [ngrid,3]
    m_times_r_tot = (m_times_r.T*weights).sum(axis=1)
    
    return m_times_r_tot
    


