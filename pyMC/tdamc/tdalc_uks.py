#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-04-06 20:39:00
LastEditTime: 2022-07-09 15:14:01
LastEditors: Li Hao
Description: Collinear TDA gives three kinds of excited energies for Locally collinear approach, including
            Spin-flip-down, Spin-flip-up and Spin-conserved.

FilePath: \pyMC\tdamc\tdalc_uks.py
Motto: A + B = C!
'''

import numpy
from pyMC.tdamc import numint_tdamc
from pyMC.tdamc import tdamc_uks

# ToDo: Global hybrid functionals have not been compolished because 
# ToDo: locally collienar approach contains LDA functionals only.

def get_iAmat_and_mo_tda(mf,xctype,ao,Extype='SPIN_CONSERVED'):
    return tdamc_uks.get_iAmat_and_mo_tda(mf,xctype,ao,Extype=Extype)
    
def get_hartree_potential_tda(mol,C_ao):
    return tdamc_uks.hartree_potential_tda(mol,C_ao)

def get_hybrid_exchange_energy_tda(mol,C_ao,Extype):
    return tdamc_uks.get_hybrid_exchange_energy_tda(mol,C_ao,Extype)
    
def get_tda_Amat(mf,Extype='SPIN_CONSERVED',LIBXCT_factor=1e-10,KST_factor=1e-10,ncpu=None):
    nitdamc = numint_tdamc.numint_tdamc()
    xc_code = mf.xc
    xctype = nitdamc._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    elif xctype == 'MGGA':
        deriv = 2
    ao = nitdamc.eval_ao(mf.mol, mf.grids.coords, deriv=deriv)
    dma,dmb = mf.make_rdm1()
    # rho and s should be calcualted in the numint_tdam.py.
    rho = nitdamc.eval_rho(mf.mol, ao, dma+dmb, xctype=xctype)
    Mz = nitdamc.eval_rho(mf.mol, ao, dma-dmb, xctype=xctype)
    # s = numpy.abs(Mz)
    # import pdb
    # pdb.set_trace()
    
    # enabling range-separated hybrids
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin)
    
    if Extype =='SPIN_FLIP_UP' or Extype =='SPIN_FLIP_DOWN':
        iAmat,C_mo,C_ao = get_iAmat_and_mo_tda(mf,xctype,ao,Extype=Extype)
        K_aibj = nitdamc.nr_collinear_tdalc(xc_code,(rho,Mz),mf.grids,C_mo,Extype=Extype,LIBXCT_factor=LIBXCT_factor,
                                            KST_factor=KST_factor,ncpu=ncpu)
        
        # Hybrid Exchange Energy.
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            K_aibj_hye = 0.0
        else:
            K_aibj_hye = get_hybrid_exchange_energy_tda(mf.mol,C_ao,Extype)
            K_aibj_hye *= hyb
            if abs(omega) > 1e-10:
                raise NotImplementedError('Range Seperation hybrid functionals have not been compolished.')
                # K_aibj_hye_lr = get_hybrid_Exchange_energy_tda(mf.mol,C_ao,Extype)
                # K_aibj_hye_lr *= (alpha - hyb)
                # K_aibj_hye += K_aibj_hye_lr
        # 0.5*2 = 1 for the double on calculation of xx,yy parts.          
        K_aibj -= 0.5*K_aibj_hye
        ndim1,ndim2 = K_aibj.shape[:2]
        ndim = ndim1*ndim2
        Kmat = K_aibj.reshape((ndim,ndim),order = 'C')
        # *2 means xx,yy parts
        Amat = iAmat + 2*Kmat
        return Amat
    
    elif Extype =='SPIN_CONSERVED':
        # C_ao is only given in spin-conserved case. 
        iAmat,C_mo,C_ao = get_iAmat_and_mo_tda(mf,xctype,ao,Extype=Extype)
        K_aibj_aaaa,K_aibj_aabb,K_aibj_bbaa,K_aibj_bbbb = \
            nitdamc.nr_collinear_tdalc(xc_code,(rho,Mz),mf.grids,C_mo,Extype=Extype,ncpu=ncpu)
            
        Kaibj_aibj_hrp_aaaa,Kaibj_aibj_hrp_aabb,Kaibj_aibj_hrp_bbaa,Kaibj_aibj_hrp_bbbb = \
            get_hartree_potential_tda(mf.mol,C_ao)
        
        # Hybrid Exchange Energy.
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            K_aibj_hyb_aaaa,K_aibj_hyb_bbbb = (0,0)
        else:
            K_aibj_hyb_aaaa,K_aibj_hyb_bbbb = get_hybrid_exchange_energy_tda(mf.mol,C_ao,Extype)
            K_aibj_hyb_aaaa *= hyb
            K_aibj_hyb_bbbb *= hyb
            if abs(omega) > 1e-10:
                raise NotImplementedError('Range Seperation hybrid functionals have not been compolished.')
                # K_aibj_hyb_aaaa_lr,K_aibj_hyb_aabb_lr,K_aibj_hyb_bbaa_lr,K_aibj_hyb_bbbb_lr \
                #     = get_hybrid_Exchange_energy_tda(mf.mol,C_ao,Extype)*(alpha - hyb)
                
                # K_aibj_hyb_aaaa_lr *= hyb
                # K_aibj_hyb_bbbb_lr *= hyb
                
                # K_aibj_hyb_aaaa += K_aibj_hyb_aaaa_lr
                # K_aibj_hyb_bbbb += K_aibj_hyb_bbbb_lr
        
        ndim1_aaaa,ndim2_aaaa = K_aibj_aaaa.shape[:2]
        ndim_aaaa = ndim1_aaaa*ndim2_aaaa
        ndim1_aabb,ndim2_aabb = K_aibj_aabb.shape[:2]
        ndim_aabb = ndim1_aabb*ndim2_aabb
        ndim1_bbaa,ndim2_bbaa = K_aibj_bbaa.shape[:2]
        ndim_bbaa = ndim1_bbaa*ndim2_bbaa
        ndim1_bbbb,ndim2_bbbb = K_aibj_bbbb.shape[:2]
        ndim_bbbb = ndim1_bbbb*ndim2_bbbb
        
        Kmat_aaaa = K_aibj_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C')
        Kmat_aabb = K_aibj_aabb.reshape((ndim_aabb,ndim_bbaa),order='C')
        Kmat_bbaa = K_aibj_bbaa.reshape((ndim_bbaa,ndim_aabb),order='C')
        Kmat_bbbb = K_aibj_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
        
        # Hartree potential parts.
        Kaibj_aibj_hrp_aaaa = Kaibj_aibj_hrp_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C')
        Kaibj_aibj_hrp_aabb = Kaibj_aibj_hrp_aabb.reshape((ndim_aabb,ndim_bbaa),order='C')
        Kaibj_aibj_hrp_bbaa = Kaibj_aibj_hrp_bbaa.reshape((ndim_bbaa,ndim_aabb),order='C')
        Kaibj_aibj_hrp_bbbb = Kaibj_aibj_hrp_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
        
        # Hartree-Fock Exchange parts.
        if type(K_aibj_hyb_aaaa) is numpy.ndarray:
            K_aibj_hyb_aaaa = K_aibj_hyb_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C') 
            K_aibj_hyb_bbbb = K_aibj_hyb_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
        
        ndim1 = Kmat_aaaa.shape[0]
        ndim2 = Kmat_bbbb.shape[0]
        ndim_dm = ndim1 + ndim2
        Amat = numpy.zeros((ndim_dm,ndim_dm))
        Kmat = numpy.zeros((ndim_dm,ndim_dm))
        
        Amat_aa = iAmat[0]
        Amat_bb = iAmat[1]
        Amat[:ndim1,:ndim1] = Amat_aa
        Amat[ndim1:,ndim1:] = Amat_bb
        
        Kmat_aaaa += Kaibj_aibj_hrp_aaaa
        Kmat_aabb += Kaibj_aibj_hrp_aabb
        Kmat_bbaa += Kaibj_aibj_hrp_bbaa
        Kmat_bbbb += Kaibj_aibj_hrp_bbbb
        
        Kmat_aaaa -= K_aibj_hyb_aaaa
        Kmat_bbbb -= K_aibj_hyb_bbbb
        
        Kmat[:ndim1,:ndim1] = Kmat_aaaa
        Kmat[:ndim1,ndim1:] = Kmat_aabb
        Kmat[ndim1:,:ndim1] = Kmat_bbaa
        Kmat[ndim1:,ndim1:] = Kmat_bbbb
        Amat += Kmat
    
        return Amat
    
def eigh_tda(self,Amat,Extype):
    E_ex = numpy.linalg.eigh(Amat)[0]*27.21138386
    self.Extd = E_ex
    # numpy.save('E_ex_'+ str(Extype) ,E_ex)
    # for i in range(E_ex.shape[-1]):
    #     print(f"{E_ex[i]:16.14f}")

class TDALC_UKS:
    def __init__(self,mf):
        # Pre-scf-calculate results: mf
        self.scf = mf
        # Method: Multi-Collinear, Tri-direction, Kubler
        self.METHOD = 'KUBLER'
        # Excited energy type.
        self.Extype = 'SPIN_CONSERVED'
        # Numstable Appraoch: Threshold and num_aprch.
        self.LIBXCT_factor = None
        # KST_factor means KUBLER S THRESHOLD
        self.KST_factor = 1e-10
        self.ncpu = None 
        # Store Excited energy.
        self.Extd = None
        # Save the  A-matrix
        self.Amat_f = None
        
    get_iAmat_and_mo_tda = get_iAmat_and_mo_tda
    get_hartree_potential_tda = get_hartree_potential_tda
    get_hybrid_exchange_energy_tda = get_hybrid_exchange_energy_tda
    eigh_tda = eigh_tda

    def kernel(self,Extype=None,LIBXCT_factor=None,KST_factor=None,ncpu=None,Extd=None,Amat_f=None):
        # This part should be more smart.
        if Extype is None:
            Extype = self.Extype
        else:
            Extype = Extype.upper()
        if LIBXCT_factor is None:
            LIBXCT_factor = self.LIBXCT_factor
        if KST_factor is None:
            KST_factor = self.KST_factor
        if ncpu is None:
            ncpu = self.ncpu
        if Extd is None:
            Extd = self.Extd
        Amat_tot = get_tda_Amat(self.scf,Extype,LIBXCT_factor,KST_factor,ncpu)
        if Amat_f is None:
            self.Amat_f = Amat_tot
        eigh_tda(self,Amat_tot,Extype)