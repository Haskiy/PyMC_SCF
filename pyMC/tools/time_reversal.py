#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-06-28 19:55:21
LastEditTime: 2022-04-10 10:32:41
LastEditors: Li Hao
Description: 
    Some util functions for time-reversal orbitals
FilePath: \pyMC\tools\time_reversal.py

 May the force be with you!
'''

import numpy
import scipy
from collections import Counter
from pyscf import lib
from pyscf.dft import numint,r_numint
from pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, BLKSIZE
from pyMC.gksmc import numint_gksmc
from pyMC.tools import rotate_dm

# def time_reversal_orbital_coeff(mol, mo_coeff):
#     """get the mo_coefficients of the time-reversal orbital.

#     Args:
#         mol (gto.mole object): the atom or the cluster, that's to be calculated.
#         natom (int): number or the atoms
#         mo_coeff (numpy.arrays): mo_coeff

#     Returns:
#         mo_coeff_f (numpy.arrays): mo_coefficients of the time-reversal orbital
#     """
#     n2c = mo_coeff.shape[0]//2
#     nao = n2c//2
#     W_part = numpy.zeros((n2c, n2c), dtype = numpy.complex128)
#     # get the U_sph2spinor_part, Dpart and U_spin_part matrix.
#     # * Note that: part means only get the [natom*nao2c,natom*nao2c] for LL part
#     U_sph2spinor_part = rotate_dm.cal_sph2spinor_matrix(mol, 1)
#     W_part[:nao] = U_sph2spinor_part[nao:].conj()*(-1.0)
#     W_part[nao:] = U_sph2spinor_part[:nao].conj()*1.0
#     U_sph2spinor = numpy.array(scipy.linalg.block_diag(U_sph2spinor_part,U_sph2spinor_part))
#     W = numpy.array(scipy.linalg.block_diag(W_part,W_part))
    
#     mo_coeff_f = U_sph2spinor.conj().T@W@mo_coeff.conj()
    
#     return mo_coeff_f

def time_reversal_orbital_coeff_2(mol, mo_coeff):
    
    n2c = mo_coeff.shape[0]//2
    nao = n2c//2
    nhalf = n2c
    idx = mol.time_reversal_map()
    sign = idx/numpy.abs(idx)
    idx_change = numpy.abs(idx)-1
    mo_coeff_f = numpy.zeros(mo_coeff.shape,dtype = numpy.complex128)
    for i in range(n2c):
        ioff = numpy.where(idx_change==i)[0]
        mo_coeff_f[i] = mo_coeff[ioff].conj()*sign[ioff]
        mo_coeff_f[i + nhalf] = mo_coeff[ioff+ nhalf].conj()*sign[ioff]
        
    
    return mo_coeff_f
    
    
