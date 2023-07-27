#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-04-06 21:31:47
LastEditTime: 2022-09-08 08:17:44
LastEditors: Li Hao
Description: Non-Collinear TDA gives three kinds of excited energies for Locally collinear approach, including
            Spin-flip-down, Spin-flip-up and Spin-conserved.

FilePath: /pyMC/tdamc/tdalc_gks.py
Motto: A + B = C!
'''

# ToDo: Global hybrid functionals have not been compolished because 
# ToDo: locally collienar approach contains LDA functionals only.

import numpy
import scipy
from pyscf import dft
from pyMC.tdamc import numint_tdamc
from pyMC.tdamc import tdamc_gks

def uks_to_gks(mf1):
    return tdamc_gks.uks_to_gks(mf1)

def uks_to_gks_iAamt_and_mo_tda(mf1,mf2,xctype,ao,diff=1e-4):
    return tdamc_gks.uks_to_gks_iAamt_and_mo_tda(mf1,mf2,xctype,ao,diff=diff)

def get_iAmat_and_mo_tda(mf,xctype,ao):
    return tdamc_gks.get_iAmat_and_mo_tda(mf,xctype,ao)
    
def get_hartree_potential_tda(mol,C_ao):
   return tdamc_gks.get_hartree_potential_tda(mol,C_ao)

def get_hybrid_exchange_energy_tda(mol,C_ao):
    return tdamc_gks.get_hybrid_exchange_energy_tda(mol,C_ao)

def get_tdalc_Amat(mf,mf2,diff,LIBXCT_factor=1e-10,KST_factor=1e-10,ncpu=None):
    nitdamc = numint_tdamc.numint_tdamc()
    xctype = nitdamc._xc_type(mf.xc)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    elif xctype == 'MGGA':
        deriv = 2
        
    mol = mf.mol
    ao = nitdamc.eval_ao(mol, mf.grids.coords, deriv=deriv)
    nao = mol.nao
    if mf2 is not None:
        iAmat, C_mo, C_ao = uks_to_gks_iAamt_and_mo_tda(mf,mf2,xctype,ao,diff)
        mf = mf2
    else:
        iAmat, C_mo, C_ao = get_iAmat_and_mo_tda(mf,xctype,ao)
    
    K_aibj_hrp = get_hartree_potential_tda(mf.mol,C_ao)
    
    # enabling range-separated hybrids
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin)
    
    # Hybrid Exchange Energy.
    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        K_aibj_hyb = 0.0
    else:
        K_aibj_hyb = get_hybrid_exchange_energy_tda(mf.mol,C_ao)
        K_aibj_hyb *= hyb
        if abs(omega) > 1e-10:
            raise NotImplementedError('Range Seperation hybrid functionals have not been compolished.')
            # K_aibj_hyb_lr = get_hybrid_Exchange_energy_tda(mf.mol,C_ao)
            # K_aibj_hyb_lr *= (alpha - hyb)
            # K_aibj_hyb += K_aibj_hyb_lr
    
    dmi = mf.make_rdm1()
    dmaa = dmi[:nao,:nao]
    dmab = dmi[:nao,nao:]
    dmba = dmi[nao:,:nao]
    dmbb = dmi[nao:,nao:]
    # import pdb
    # pdb.set_trace()
    K_aibj = nitdamc.nr_noncollinear_tdalc(mol, mf.xc, mf.grids, (dmaa,dmab,dmba,dmbb), C_mo, 
                                           LIBXCT_factor=LIBXCT_factor, KST_factor=KST_factor,ncpu=ncpu)
    K_aibj += K_aibj_hrp   
    K_aibj -= K_aibj_hyb  
    # K_aibj.reshape() -> Kmat
    ndim1,ndim2 = K_aibj.shape[:2]
    ndim = ndim1*ndim2
    Kmat = K_aibj.reshape((ndim,ndim),order = 'C')
    # import pdb
    # pdb.set_trace()
    iAmat = numpy.diag(iAmat)
    Amat = iAmat + Kmat
    return Amat

def eigh_tda(self,Amat):
    E_ex = numpy.linalg.eigh(Amat)[0]*27.21138386
    self.Extd = E_ex
    # numpy.save('E_ex',E_ex)
    # for i in range(E_ex.shape[-1]):
    #     print(f"{E_ex[i]:16.14f}")
          
class TDALC_GKS:
    def __init__(self,mf):
        # Pre-scf-calculate results: mf:gks,uks.add()
        # uks object will be transform into gks object.
        self.scf = mf
        # LIBXCT_factor means THRESHOLD FOR LIBXC
        self.LIBXCT_factor = None
        # uks_to_gks: diff value
        self.diff = 1e-4 
        # KST_factor means KUBLER S THRESHOLD
        self.KST_factor = 1e-10 
        self.ncpu = None
        # Store Excited energy.
        self.Extd = None
        # Save the  A-matrix
        self.Amat_f = None
        
    get_iAmat_and_mo_tdam = get_iAmat_and_mo_tda
    get_hartree_potential_tda = get_hartree_potential_tda
    get_hybrid_exchange_energy_tda = get_hybrid_exchange_energy_tda
    eigh_tdam = eigh_tda

    def kernel(self, mf2=None,diff=None,LIBXCT_factor=None,KST_factor=None,ncpu=None,Extd=None,Amat_f=None):
        # This part should be more smart.
        if isinstance(self.scf,dft.uks.UKS) or isinstance(self.scf,dft.uks_symm.SymAdaptedUKS):
            mf2 = uks_to_gks(self.scf)
        if LIBXCT_factor is None:
            LIBXCT_factor = self.LIBXCT_factor
        if diff is None:
            diff = self.diff
        if KST_factor is None:
            KST_factor = self.KST_factor
        if ncpu is None:
            ncpu = self.ncpu
        if Extd is None:
            Extd = self.Extd
        Amat_tot = get_tdalc_Amat(self.scf,mf2,diff,LIBXCT_factor,KST_factor,ncpu)
        if Amat_f is None:
            self.Amat_f = Amat_tot
        eigh_tda(self,Amat_tot)
        
        