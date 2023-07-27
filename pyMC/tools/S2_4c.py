#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-07-20 09:20:43
LastEditTime: 2021-07-23 18:26:59
LastEditors: Pu Zhichen
Description: 
    Get the 4c <S^2>
FilePath: \pyMC\tools\S2_4c.py

 May the force be with you!
'''

from pyscf import gto,lib
import numpy
import scipy

def spin_square_4c(mol, mo, nocc):
    
    # some parameters.
    n2c = mol.nao_2c()
    c1 = 1.0 / lib.param.LIGHT_SPEED
    ca, cb = mol.sph2spinor_coeff()
    sL1c = mol.intor('int1e_ovlp_sph')
    sS1c = mol.intor('int1e_kin') * c1**2 * 0.5
    na, nb = mol.nelec
    nelec = na + nb
    # spinor based S matrix
    SxL_sp = ca.conj().T@sL1c@cb + cb.conj().T@sL1c@ca
    SxS_sp = ca.conj().T@sS1c@cb + cb.conj().T@sS1c@ca
    SyL_sp = -1.0j*ca.conj().T@sL1c@cb + 1.0j*cb.conj().T@sL1c@ca
    SyS_sp = -1.0j*ca.conj().T@sS1c@cb + 1.0j*cb.conj().T@sS1c@ca
    SzL_sp = ca.conj().T@sL1c@ca - cb.conj().T@sL1c@cb
    SzS_sp = ca.conj().T@sS1c@ca - cb.conj().T@sS1c@cb
    # mo informations
    idx = nocc != 0
    
    # 1 e part.
    S2 = 0.75*nelec
    import pdb
    pdb.set_trace()
    # 2 e part S_u matrix
    SxL = mo[:n2c,idx].conj().T@SxL_sp@mo[:n2c,idx] * 0.5
    SxS = mo[n2c:,idx].conj().T@SxS_sp@mo[n2c:,idx] * 0.5
    SyL = mo[:n2c,idx].conj().T@SyL_sp@mo[:n2c,idx] * 0.5
    SyS = mo[n2c:,idx].conj().T@SyS_sp@mo[n2c:,idx] * 0.5
    SzL = mo[:n2c,idx].conj().T@SzL_sp@mo[:n2c,idx] * 0.5
    SzS = mo[n2c:,idx].conj().T@SzS_sp@mo[n2c:,idx] * 0.5
    # 2 e part S_x 
    SLlist = [SxL, SyL, SzL]
    SSlist = [SxS, SyS, SzS]
    for i in range(3):
        trL = SLlist[i].trace()
        trS = SSlist[i].trace()
        S2+= trL*trL - trL*trS*2.0 + trS*trS
        tmp = SLlist[i] - SSlist[i]
        S2-= numpy.einsum('ij,ji', tmp, tmp)
        
    print('<S^2> = %.7f' % S2)
    
    return S2
    
