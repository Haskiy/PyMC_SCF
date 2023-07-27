#/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-03-03 19:12:56
LastEditTime: 2023-02-13 02:25:17
LastEditors: Li Hao
Description: 
    initiate
FilePath: /pyMC/gksmc/__init__.py

 May the force be with you!
'''

from pyMC.gksmc import gksmc
from pyMC.gksmc import gkslc
from pyMC.gksmc import gksmc_r
from pyMC.gksmc import gksmc_symm
from pyMC.gksmc import gks_sym_general
from pyMC.gksmc import gksmc_r_symm
from pyMC.gksmc import uks_nlc
from pyMC.gksmc import gksmc_nlc
# from pyMC.gksmc import gksmc_nlc_sd

try:
    from pyscf.dft import libxc
    XC = libxc.XC
except (ImportError, OSError):
    pass
try:
    from pyscf.dft import xcfun
    XC = xcfun.XC
except (ImportError, OSError):
    pass

from pyscf.scf import ghf
from pyscf.dft import gks

# TODO : Many files can be contracted into one file, which will largely decrease the lines and files
# TODO : in gksmc folder.

def GHF_sym(mol, xc='LDA,VWN'):
    if hasattr(mol, 'Dsymmetry'):
        if  mol.Dsymmetry :
            return gks_sym_general.GKS_symm(mol, xc)
        else:
            return ghf.GHF(mol)
    else:
        return ghf.GHF(mol)
        

def GKSMC(mol, xc='LDA,VWN'):
    if hasattr(mol, 'Dsymmetry'):
        if  mol.Dsymmetry :
            return gksmc_symm.GKSMC_symm(mol, xc)
        else:
            return gksmc.GKSMC(mol, xc)
    else:
        return gksmc.GKSMC(mol, xc)
    
    
def GKSMC_r(mol, xc='LDA,VWN'):
    if hasattr(mol, 'Dsymmetry'):
        if  mol.Dsymmetry :
            return gksmc_r_symm.GKSMC_r_symm(mol, xc)
        else:
            return gksmc_r.GKSMC_r(mol, xc)
    else:
        return gksmc_r.GKSMC_r(mol, xc)
    
    
def GKS_symmetry(mol, xc='LDA,VWN'):
    if hasattr(mol, 'Dsymmetry'):
        if  mol.Dsymmetry :
            return gks_sym_general.GKSM(mol, xc)
        else:
            return gks.GKS(mol, xc)
    else:
        return gks.GKS(mol, xc)
    

def GKSLC(mol, xc='LDA,VWN'):
    if hasattr(mol, 'Dsymmetry'):
        if  mol.Dsymmetry :
            raise NotImplementedError('Locally collinear GKS with double group symmetry is not implemented!')
        else:
            return gkslc.GKSLC(mol, xc)
    else:
        raise NotImplementedError('Locally collinear GKS with double group symmetry is not implemented!')


