#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-03-14 21:06:41
LastEditTime: 2023-03-24 05:05:44
LastEditors: Li Hao
Description: Collinear TDA gives three kinds of excited energies, including
            Spin-flip-down, Spin-flip-up and Spin-conserved. 

FilePath: /pyMC/tdamc/tdamc_uks.py
Motto: A + B = C!
'''

import numpy
from pyMC.tdamc import numint_tdamc
from pyMC.tdamc import solver

def get_iAmat_and_mo_tda(mf,xctype,ao,Extype='SPIN_CONSERVED'):
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    
    nao = mf.mol.nao
    nocca = int(mo_occ[0].sum())
    noccb = int(mo_occ[1].sum())
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    
    if Extype == 'SPIN_FLIP_UP':
        nvira = nao - nocca   
        
        e_ib = mo_energy[1][occidxb]
        e_aa = mo_energy[0][viridxa]
        e_ai_b2a = numpy.array([e_aa[i_aa] - e_ib[i_ib]
                                for i_aa in range(nvira)
                                for i_ib in range(noccb)])
        # Calculate Amat
        # iAmat = numpy.diag(e_ai_b2a)
        
        Ca, Cb = mf.mo_coeff
        Ca_vir = Ca[:,viridxa]
        Cb_occ = Cb[:,occidxb]
         # n means ngrid
        if xctype == 'LDA':
            mo_a_vir = numpy.einsum('ui,nu->ni',Ca_vir,ao)
            mo_b_occ = numpy.einsum('ui,nu->ni',Cb_occ,ao)
        # g means ao[0].shape, meaning \rho,\nabla \rho - 3, lapl\rho - 6, 
        elif xctype == 'GGA':
            mo_a_vir = numpy.einsum('ui,gnu->gni',Ca_vir,ao)
            mo_b_occ = numpy.einsum('ui,gnu->gni',Cb_occ,ao)
        elif xctype == 'MGGA':
            mo_a_vir = numpy.einsum('ui,gnu->gni',Ca_vir,ao)
            mo_b_occ = numpy.einsum('ui,gnu->gni',Cb_occ,ao)
        
        return e_ai_b2a,(mo_a_vir,mo_b_occ),(Ca_vir,Cb_occ)
        
    elif Extype == 'SPIN_FLIP_DOWN':
        nvirb = nao - noccb
    
        e_ia = mo_energy[0][occidxa]
        e_ab = mo_energy[1][viridxb]
        e_ai_a2b = numpy.array([e_ab[i_ab] - e_ia[i_ia]
                                for i_ab in range(nvirb)
                                for i_ia in range(nocca)])
        # iAmat = numpy.diag(e_ai_a2b)
        
        Ca, Cb = mf.mo_coeff
        Cb_vir = Cb[:,viridxb]
        Ca_occ = Ca[:,occidxa]
        
        if xctype == 'LDA':
            mo_b_vir = numpy.einsum('ui,nu->ni',Cb_vir,ao)
            mo_a_occ = numpy.einsum('ui,nu->ni',Ca_occ,ao)
        # g means ao[0].shape, meaning \rho,\nabla \rho - 3, lapl\rho - 6, 
        elif xctype == 'GGA':
            mo_b_vir = numpy.einsum('ui,gnu->gni',Cb_vir,ao)
            mo_a_occ = numpy.einsum('ui,gnu->gni',Ca_occ,ao)
        elif xctype == 'MGGA':
            mo_b_vir = numpy.einsum('ui,gnu->gni',Cb_vir,ao)
            mo_a_occ = numpy.einsum('ui,gnu->gni',Ca_occ,ao)
        
        return e_ai_a2b,(mo_b_vir,mo_a_occ),(Cb_vir,Ca_occ)

    elif Extype == 'SPIN_CONSERVED':
        nvira = nao - nocca
        nvirb = nao - noccb
    
        e_ia = mo_energy[0][occidxa]
        e_ib = mo_energy[1][occidxb]
        e_aa = mo_energy[0][viridxa]
        e_ab = mo_energy[1][viridxb]
        e_ai_aa = numpy.array([e_aa[i_aa] - e_ia[i_ia]
                            for i_aa in range(nvira)
                            for i_ia in range(nocca)])
    
        e_ai_bb = numpy.array([e_ab[i_ab] - e_ib[i_ib]
                            for i_ab in range(nvirb)
                            for i_ib in range(noccb)])
        # iAmat_aa = numpy.diag(e_ai_aa)
        # iAmat_bb = numpy.diag(e_ai_bb)
        
        Ca, Cb = mf.mo_coeff
        Ca_vir = Ca[:,viridxa]
        Ca_occ = Ca[:,occidxa]
        Cb_vir = Cb[:,viridxb]
        Cb_occ = Cb[:,occidxb]
        # n means ngrid
        if xctype == 'LDA':
            mo_a_occ = numpy.einsum('ui,nu->ni',Ca_occ,ao ,optimize = True)
            mo_a_vir = numpy.einsum('ui,nu->ni',Ca_vir,ao ,optimize = True)
            mo_b_occ = numpy.einsum('ui,nu->ni',Cb_occ,ao ,optimize = True)
            mo_b_vir = numpy.einsum('ui,nu->ni',Cb_vir,ao ,optimize = True)
        # g means ao[0].shape, meaning \rho,\nabla \rho - 3, lapl\rho - 6, 
        elif xctype == 'GGA' or xctype == 'MGGA':
            mo_a_occ = numpy.einsum('ui,gnu->gni',Ca_occ,ao,
                                optimize = True)[:4]
            mo_a_vir = numpy.einsum('ui,gnu->gni',Ca_vir,ao,
                                    optimize = True)[:4]
            mo_b_occ = numpy.einsum('ui,gnu->gni',Cb_occ,ao,
                                    optimize = True)[:4]
            mo_b_vir = numpy.einsum('ui,gnu->gni',Cb_vir,ao,
                                    optimize = True)[:4] 
   
        return (e_ai_aa,e_ai_bb),(mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ),(Ca_vir,Ca_occ,Cb_vir,Cb_occ)
    
def get_hybrid_exchange_energy_tda(mol,C_ao,Extype,omega,alpha,hyb):
    # Columb Interaction, which isn't zero in spin-conserved tda case only.
    # eri in ao space
    
    eri_ao = mol.intor('int2e')
    eri_ao*=hyb
   
    if abs(omega) >= 1e-10:
        with mol.with_range_coulomb(omega=omega):
            eri_ao+= mol.intor('int2e')*(alpha - hyb)

    # transform the eri to mo space
    if Extype == 'SPIN_FLIP_UP':
        Ca_vir,Cb_occ = C_ao
        K_aibj_hyb = numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Ca_vir.conj(),Ca_vir,Cb_occ,Cb_occ.conj()
                          ,optimize = True)
        K_aibj_hyb = numpy.transpose(K_aibj_hyb,(0,2,1,3))
        return  K_aibj_hyb
    
    elif Extype == 'SPIN_FLIP_DOWN':
        Cb_vir,Ca_occ = C_ao
        K_aibj_hyb = numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Cb_vir.conj(),Cb_vir,Ca_occ,Ca_occ.conj()
                          ,optimize = True)
        K_aibj_hyb = numpy.transpose(K_aibj_hyb,(0,2,1,3))
        return  K_aibj_hyb
    
    elif Extype == 'SPIN_CONSERVED':
        Ca_vir,Ca_occ,Cb_vir,Cb_occ = C_ao
        K_aibj_hyb_aaaa = numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Ca_vir.conj(),Ca_vir,Ca_occ,Ca_occ.conj()
                          ,optimize = True)
        # aabb and bbaa blocks of K_aibj_hyb are zero.
        K_aibj_hyb_bbbb = numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Cb_vir.conj(),Cb_vir,Cb_occ,Cb_occ.conj()
                          ,optimize = True)
        K_aibj_hyb_aaaa = numpy.transpose(K_aibj_hyb_aaaa,(0,2,1,3))
        K_aibj_hyb_bbbb = numpy.transpose(K_aibj_hyb_bbbb,(0,2,1,3))
        return  (K_aibj_hyb_aaaa,K_aibj_hyb_bbbb)

def get_hartree_potential_tda(mol,C_ao):
    # Columb Interaction, which isn't zero in spin-conserved tda case only.
    # eri in ao space
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = C_ao
    eri_ao = mol.intor('int2e')
    # transform the eri to mo space
    K_aibj_hrp_aaaa = numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Ca_vir,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_hrp_aabb = numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Cb_vir,Cb_occ.conj()
                          ,optimize = True)
    K_aibj_hrp_bbaa = numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Ca_vir,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_hrp_bbbb = numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Cb_vir,Cb_occ.conj()
                          ,optimize = True)
    return K_aibj_hrp_aaaa, K_aibj_hrp_aabb, K_aibj_hrp_bbaa, K_aibj_hrp_bbbb
    
def get_tda_Amat(mf,Ndirect,Extype='SPIN_CONSERVED',LIBXCT_factor=1e-10,ncpu=None):
    # import pdb
    # pdb.set_trace()
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
    s = nitdamc.eval_rho(mf.mol, ao, dma-dmb, xctype=xctype)
    # import pdb
    # pdb.set_trace()
    
    # enabling range-separated hybrids
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin)
        
    if Extype =='SPIN_FLIP_UP' or Extype =='SPIN_FLIP_DOWN':
        iAmat,C_mo,C_ao = get_iAmat_and_mo_tda(mf,xctype,ao,Extype=Extype)
        K_aibj = nitdamc.nr_collinear_tdamc(xc_code,(rho,s),mf.grids,Ndirect,C_mo,
                                          Extype=Extype,LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
        # Hybrid Exchange Energy.
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            K_aibj_hye = 0.0
        else:
            K_aibj_hye = get_hybrid_exchange_energy_tda(mf.mol,C_ao,Extype,omega,alpha,hyb)
        
        # Whole diag for e_ai
        iAmat = numpy.diag(iAmat)
        # import pdb
        # pdb.set_trace()
        
        # Just test Shao work.
        # K_aibj*=0.0
        # Test finishes!
        
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
            nitdamc.nr_collinear_tdamc(xc_code,(rho,s),mf.grids,Ndirect,C_mo,
                                        Extype=Extype,ncpu=ncpu)
        K_aibj_hrp_aaaa,K_aibj_hrp_aabb,K_aibj_hrp_bbaa,K_aibj_hrp_bbbb = \
            get_hartree_potential_tda(mf.mol,C_ao)
        
    
        # Hybrid Exchange Energy.
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            K_aibj_hyb_aaaa,K_aibj_hyb_bbbb = (0,0)
        else:
            K_aibj_hyb_aaaa,K_aibj_hyb_bbbb = get_hybrid_exchange_energy_tda(mf.mol,C_ao,Extype,omega,alpha,hyb)
            
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
        K_aibj_hrp_aaaa = K_aibj_hrp_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C')
        K_aibj_hrp_aabb = K_aibj_hrp_aabb.reshape((ndim_aabb,ndim_bbaa),order='C')
        K_aibj_hrp_bbaa = K_aibj_hrp_bbaa.reshape((ndim_bbaa,ndim_aabb),order='C')
        K_aibj_hrp_bbbb = K_aibj_hrp_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
        
        # Hartree-Fock Exchange parts.
        # import pdb
        # pdb.set_trace()
        if type(K_aibj_hyb_aaaa) is numpy.ndarray:
            K_aibj_hyb_aaaa = K_aibj_hyb_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C') 
            K_aibj_hyb_bbbb = K_aibj_hyb_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
        
        ndim1 = Kmat_aaaa.shape[0]
        ndim2 = Kmat_bbbb.shape[0]
        ndim_dm = ndim1 + ndim2
        Amat = numpy.zeros((ndim_dm,ndim_dm))
        Kmat = numpy.zeros((ndim_dm,ndim_dm))
        
        # Whole diag for e_ai
        Amat_aa = numpy.diag(iAmat[0])
        Amat_bb = numpy.diag(iAmat[1])
        Amat[:ndim1,:ndim1] = Amat_aa
        Amat[ndim1:,ndim1:] = Amat_bb
        
        # just Test Shao's work.
        # Kmat_aaaa *= 0.0
        # Kmat_aabb *= 0.0
        # Kmat_bbaa *= 0.0
        # Kmat_bbbb *= 0.0
        
        Kmat_aaaa += K_aibj_hrp_aaaa
        Kmat_aabb += K_aibj_hrp_aabb
        Kmat_bbaa += K_aibj_hrp_bbaa
        Kmat_bbbb += K_aibj_hrp_bbbb
        
        Kmat_aaaa -= K_aibj_hyb_aaaa
        Kmat_bbbb -= K_aibj_hyb_bbbb
        
        Kmat[:ndim1,:ndim1] = Kmat_aaaa
        Kmat[:ndim1,ndim1:] = Kmat_aabb
        Kmat[ndim1:,:ndim1] = Kmat_bbaa
        Kmat[ndim1:,ndim1:] = Kmat_bbbb
        
        Amat += Kmat
    
        return Amat

def get_kernel(mf,Ndirect,Extype='SPIN_CONSERVED',LIBXCT_factor=1e-10,ncpu=None):
    # import pdb
    # pdb.set_trace()
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
    s = nitdamc.eval_rho(mf.mol, ao, dma-dmb, xctype=xctype)
    # import pdb
    # pdb.set_trace()
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin)
    # import pdb
    # pdb.set_trace()
    kernel = nitdamc.collinear_tdamc_kernel(xc_code,(rho,s),Ndirect, Extype=Extype,LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
    
    return ao,xctype,(kernel,(omega, alpha,hyb))
    
def eigh_tda(self,Amat,Ndirect,Extype):
    E_ex,U = numpy.linalg.eigh(Amat)
    self.Extd = E_ex*27.21138386
    self.U = U
    # if Extype =='SPIN_FLIP_UP' or Extype =='SPIN_FLIP_DOWN':
    #     print('Spin_Space_Sample_Points: '+str(Ndirect))
    # elif Extype =='SPIN_CONSERVED':
    #     pass
    # numpy.save('E_ex_'+ str(Extype) ,E_ex)
    # for i in range(E_ex.shape[-1]):
    #     print(f"{E_ex[i]:16.14f}")

class TDAMC_UKS:
    def __init__(self,mf,method='AX'):
        # Pre-scf-calculate results: mf
        self.scf = mf
        # Spin-space picking points.
        self.Ndirect = 10
        # Excited energy type.
        self.Extype = 'SPIN_CONSERVED'
        # Libxc stable Threshold.
        self.LIBXCT_factor = -1 
        self.ncpu = None
        # Store Excited energy.
        self.Extd = None
        # Save the  A-matrix
        self.Amat_f = None
        # The method of matrix multiplies vector.
        self.method = method
        # The eigh_vector given by AX mtethod. 
        self.U = None
        
    get_iAmat_and_mo_tda = get_iAmat_and_mo_tda
    get_hartree_potential_tda = get_hartree_potential_tda
    get_hybrid_exchange_energy_tda = get_hybrid_exchange_energy_tda
    eigh_tdam = eigh_tda
    excited_mag_structure = numint_tdamc.excited_mag_structure

    def kernel(self,Ndirect=None,Extype=None,LIBXCT_factor=None,ncpu=None,Extd=None,nstates=3,
               init_guess=None, max_cycle = 50, conv_tol = 1.0E-8, scheme='LAN', cutoff=8, Whkerl=False,parallel= False,Amat_f=None):
        # This part should be more smart.
        if Ndirect is None:
            Ndirect = self.Ndirect
        if Extype is None:
            Extype = self.Extype.upper()
        else:
            Extype = Extype.upper()
        if LIBXCT_factor is None:
            LIBXCT_factor = self.LIBXCT_factor
        if ncpu is None:
            self.ncpu = ncpu
        if Extd is None:
            self.Extd = Extd
        
        if self.method.upper() == 'AX':
            ao,xctype,kernel = get_kernel(self.scf, Ndirect, Extype, LIBXCT_factor, ncpu)
            sol = solver.Solver(self.scf,mf2=None,Extype=Extype,kernel=kernel, nstates=nstates, 
                                   init_guess=init_guess, scheme=scheme, max_cycle=max_cycle, 
                                   conv_tol=conv_tol, cutoff = cutoff, Whkerl=Whkerl,parallel=parallel,ncpu=ncpu)
            self.Extd, self.U = sol.solver(ao,xctype)
        else:    
            Amat_tot = get_tda_Amat(self.scf,Ndirect,Extype,LIBXCT_factor,ncpu)
            if Amat_f is None:
                self.Amat_f = Amat_tot
            eigh_tda(self,Amat_tot,Ndirect,Extype)
        