#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2021-06-08 08:03:59
LastEditTime: 2021-07-01 17:58:15
LastEditors: Pu Zhichen
Description: This file is aimming at correct the Dirac4c calculations, which M is broken
             to directions that different from z-axis.
FilePath: \undefinedd:\PKU_msi\pyMC\tools\rotate_utils2.py

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

def get_z_oriented_atom(mol, mo_coeffu, dm):
    """Rotating the deviated M vector to Z-axis, for Dirac4c calculations.

    Args:
        mol (gto.Mole obj or Mole_symm obj): One atom.
        mo_coeffu (numpy.array): 4c MO orbitals.
        dm (numpy.array): 4c DM

    Raises:
        ValueError: Non-relativity calculations.

    Returns:
        mo_coeffu_f (numpy.array): z-oriented mo coefficients.
    """
    if mol.nao*4 != dm.shape[-1]:
        raise ValueError("Only Dirac4c calculations may cause the M vector deviated\
            from the Z-axis.")
    mol.Dsymmetry = False
    mf = gksm_util.GKSM_r(mol)
    mf.grids.level = 9
    grids = mf.grids
    make_rho,nset,nao = mf._numint._gen_rho_evaluator(mol,dm)
    Mvec = numpy.zeros((3))
    zaxis = numpy.array([0.0, 0.0, 1.0])
    
    for ao, mask, weight, coords \
            in mf._numint.block_loop(mol, grids, nao, 0, True, 50000):
        for idm in range(nset):
            rho, M = make_rho(idm, ao, mask, 'LDA')
            Mvec += M@weight
    norm = numpy.linalg.norm(Mvec)
    Mvec = Mvec/norm
    print("M = [{0:14.8e}, {1:14.8e}, {2:14.8e}]".format(Mvec[0], Mvec[1], Mvec[2]))
    rotvec = numpy.cross(Mvec, zaxis)
    norm = numpy.linalg.norm(rotvec)
    rotvec = rotvec/norm
    costheta = numpy.dot(Mvec, zaxis)/numpy.linalg.norm(Mvec)
    theta = numpy.arccos(costheta)
    robj = R.from_rotvec(rotvec*theta)
    euler = robj.as_euler('ZYZ')
    mo_coeffu_f = rotate_dm.get_gks_dm_guess_mo_4c(mol, mo_coeffu, 1, [[euler, rotvec, theta],])
    
    return mo_coeffu_f
    
def get_vec_M(mol, dm):
    """Rotating the deviated M vector to Z-axis, for Dirac4c calculations.

    Args:
        mol (gto.Mole obj or Mole_symm obj): One atom.
        mo_coeffu (numpy.array): 4c MO orbitals.
        dm (numpy.array): 4c DM

    Raises:
        ValueError: Non-relativity calculations.

    Returns:
        mo_coeffu_f (numpy.array): z-oriented mo coefficients.
    """
    if mol.nao*4 != dm.shape[-1]:
        raise ValueError("Only Dirac4c calculations may cause the M vector deviated\
            from the Z-axis.")
    mol.Dsymmetry = False
    mf = gksm_util.GKSM_r(mol)
    mf.grids.level = 9
    grids = mf.grids
    make_rho,nset,nao = mf._numint._gen_rho_evaluator(mol,dm)
    Mvec = numpy.zeros((3))
    zaxis = numpy.array([0.0, 0.0, 1.0])
    for ao, mask, weight, coords \
            in mf._numint.block_loop(mol, grids, nao, 0, True, 50000):
        ipart = 0
        for idm in range(nset):
            ipart += 1
            rho, M = make_rho(idm, ao, mask, 'LDA')
            numpy.save('M_relative_part'+str(ipart),M)
            Mvec += M@weight
    numpy.save('coords',grids.coords)
    numpy.save('weights',grids.weights)
    return Mvec