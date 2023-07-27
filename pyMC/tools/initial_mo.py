#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-05-31 18:20:51
LastEditTime: 2022-04-10 10:31:58
LastEditors: Li Hao
Description: 
    get some initial info.
FilePath: \pyMC\tools\initial_mo.py

 May the force be with you!
'''


import numpy
import scipy
from collections import Counter
from pyscf import lib
from pyscf.dft import numint,r_numint
from pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, BLKSIZE
from pyMC.gksmc import numint_gksmc
from pyMC import tools

def get_mo_degeneracy(mo_energy,degeneracy):
    mo_energy_list = mo_energy.round(4).tolist()
    mo_degenerate = Counter(mo_energy_list)
    nl = mo_degenerate.__len__()
    idx = [i for i in range(nl) if list(mo_degenerate.values())[i] == degeneracy]
    mo_energy_degene = []
    mo_idx_degenerate = []
    offset = numpy.array(list(mo_degenerate.values()))
    for i in idx:
        mo_energy_degene.append(list(mo_degenerate.keys())[i])
        mo_idx_degenerate.append(offset[:i].sum())    
        
    return mo_energy_degene, mo_idx_degenerate, mo_degenerate

def get_init_mo(mol, mo_coeff, degeneracy, S, idx):
    """Get the Jz adapted basis.

    Args:
        mol (gto.Mole object): Atom
        mo_coeff (numpy.array): mo_coeff
        degeneracy (int): degeneracy of the target degenerate states.
        S (numpy.array): overlap matrix
        idx (tuple): index of the degeneracy orbitals, that to be diagonalise.

    Returns:
        mo_coeff2 (numpy.array): mo_coeff
    """
    jz = tools.rotate_dm._get_basjjz(mol)[1]
    n2c = mo_coeff.shape[-1]//2
    jz1 = numpy.zeros((n2c,n2c),dtype = numpy.complex128)
    jz2 = numpy.zeros((n2c,n2c),dtype = numpy.complex128)
    jz3 = numpy.zeros((n2c,n2c),dtype = numpy.complex128)
    jz4 = numpy.zeros((n2c,n2c),dtype = numpy.complex128)
    # ! The following may be problematic, because ao basis are the eigenstates of the 
    # ! Jz operator. 
    # TODO : Check the codes!
    for i in range(n2c):
        for j in range(n2c):
            if jz[i] == jz[j]:
                jz1[i,j] = jz[i]*S[i,j]
                jz2[i,j] = jz[i]*S[i+n2c,j+n2c]
                jz3[i,j] = jz[i]*S[i,j+n2c]
                jz4[i,j] = jz[i]*S[i+n2c,j]
    Jz = scipy.linalg.block_diag(jz1,jz2)
    Jz[:n2c,n2c:] = jz3
    Jz[n2c:,:n2c] = jz4
    Jzmo = mo_coeff.conj().T@Jz@mo_coeff
    mo_coeff2 = mo_coeff.copy()
    for ioff0 in idx:
        ioff1 = ioff0 + degeneracy
        jeig1, U1 = numpy.linalg.eigh(Jzmo[ioff0:ioff1, ioff0:ioff1])
        mo_coeff2[:,ioff0:ioff1] = mo_coeff[:,ioff0:ioff1]@U1
        
    # i=-1
    # for ioff0 in idx:
    #     i+=1
    #     ioff1 = ioff0 + degeneracy
    #     jeig1, U1 = numpy.linalg.eigh(Jzmo[ioff0:ioff1, ioff0:ioff1])
    #     if i==0:
    #         mo_coeff2[:,ioff0:ioff1] = mo_coeff[:,ioff0:ioff1]@U1
    #     elif i > 0:
    #         mo_coeff2[:,ioff0:ioff1] = mo_coeff2[:,idx[0]:idx[0]+degeneracy]\
    #             *mo_coeff[ioff0,ioff0]/mo_coeff[idx[0],idx[0]]
        
    return mo_coeff2
    
    
    
        