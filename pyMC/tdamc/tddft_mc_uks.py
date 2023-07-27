#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-07-28 18:48:41
LastEditTime: 2023-05-30 06:19:50
LastEditors: Li Hao
Description: Collinear TDDFT gives three kinds of excited energies, including
            Spin-flip-down, Spin-flip-up and Spin-conserved. 

FilePath: /pyMC/tdamc/tddft_mc_uks.py
Motto: A + B = C!
'''

# ToDo: Lots of similar codes as TDA part will in this file, meaning a simple
# ToDo: rewrite the two in the future.

import numpy
from pyscf import lib
from pyMC.tdamc import numint_tdamc
from pyMC.tdamc import tddft_solver

def get_e_ai_and_mo_tddft(mf,xctype,ao,Extype='SPIN_CONSERVED'):
    # import pdb
    # pdb.set_trace()
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    
    nao = mf.mol.nao
    nocca = int(mo_occ[0].sum())
    noccb = int(mo_occ[1].sum())
    nvira = nao - nocca 
    nvirb = nao - noccb
    
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    
    Ca, Cb = mf.mo_coeff
    Ca_vir = Ca[:,viridxa]
    Ca_occ = Ca[:,occidxa]
    Cb_vir = Cb[:,viridxb]
    Cb_occ = Cb[:,occidxb]
    
    # Construct the molecular orbitals.
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
    
    # Calculate the diff value between orbitals.
    if 'SPIN_FLIP' in Extype:
        e_ib = mo_energy[1][occidxb]
        e_aa = mo_energy[0][viridxa]
        e_ai_b2a = numpy.array([e_aa[i_aa] - e_ib[i_ib]
                                for i_aa in range(nvira)
                                for i_ib in range(noccb)])

    
        e_ia = mo_energy[0][occidxa]
        e_ab = mo_energy[1][viridxb]
        e_ai_a2b = numpy.array([e_ab[i_ab] - e_ia[i_ia]
                                for i_ab in range(nvirb)
                                for i_ia in range(nocca)])

        e_ai = (e_ai_b2a,e_ai_a2b)

    elif Extype == 'SPIN_CONSERVED':
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
        e_ai = (e_ai_aa,e_ai_bb)
        
    return e_ai,(mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ),(Ca_vir,Ca_occ,Cb_vir,Cb_occ)

def get_hybrid_exchange_energy_tddft(mol,C_ao,Extype,omega,alpha,hyb):
    # Columb Interaction, which isn't zero in spin-conserved tda case only.
    # eri in ao space
    # import pdb
    # pdb.set_trace()
    eri_ao = mol.intor('int2e')
    eri_ao *= hyb
   
    if abs(omega) >= 1e-10:
        with mol.with_range_coulomb(omega=omega):
            eri_ao += mol.intor('int2e')*(alpha - hyb)

    if 'SPIN_FLIP' in Extype:
        Ca_vir,Ca_occ,Cb_vir,Cb_occ= C_ao
        K_aibj_A_hyb_sfd = numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                        Cb_vir.conj(),Cb_vir,Ca_occ,Ca_occ.conj(),
                        optimize = True)
        K_aibj_B_hyb_sfd = numpy.einsum('uvwy,ua,vj,yb,wi->ajbi', eri_ao, 
                        Cb_vir.conj(),Cb_occ,Ca_vir.conj(),Ca_occ,
                        optimize = True)
        
        K_aibj_A_hyb_sfd = numpy.transpose(K_aibj_A_hyb_sfd,(0,2,1,3))
        K_aibj_B_hyb_sfd = numpy.transpose(K_aibj_B_hyb_sfd,(0,3,2,1))
        
        K_aibj_A_hyb_sfu = numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                        Ca_vir.conj(),Ca_vir,Cb_occ,Cb_occ.conj(),
                        optimize = True)
        K_aibj_B_hyb_sfu = numpy.einsum('uvwy,ua,vj,yb,wi->ajbi', eri_ao,
                          Ca_vir.conj(),Ca_occ,Cb_vir.conj(),Cb_occ,
                          optimize = True)
        
        K_aibj_A_hyb_sfu = numpy.transpose(K_aibj_A_hyb_sfu,(0,2,1,3))
        K_aibj_B_hyb_sfu = numpy.transpose(K_aibj_B_hyb_sfu,(0,3,2,1))
        
        return (K_aibj_A_hyb_sfd,K_aibj_B_hyb_sfd,K_aibj_A_hyb_sfu,K_aibj_B_hyb_sfu)
    
    elif Extype == 'SPIN_CONSERVED':
        Ca_vir,Ca_occ,Cb_vir,Cb_occ = C_ao
    
        K_aibj_A_hyb_aaaa = numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Ca_vir.conj(),Ca_vir,Ca_occ,Ca_occ.conj()
                          ,optimize = True)
        # aabb and bbaa blocks of K_aibj_hyb are zero.
        K_aibj_A_hyb_bbbb = numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Cb_vir.conj(),Cb_vir,Cb_occ,Cb_occ.conj()
                          ,optimize = True)
        
        # transpose can be written into enisum directly.
        K_aibj_A_hyb_aaaa = numpy.transpose(K_aibj_A_hyb_aaaa,(0,2,1,3))
        K_aibj_A_hyb_bbbb = numpy.transpose(K_aibj_A_hyb_bbbb,(0,2,1,3))
        # aabb and bbaa blocks of K_aibj_hyb are zero.

        K_aibj_B_hyb_aaaa = numpy.einsum('uvwy,ua,vj,wb,yi->ajbi', eri_ao,
                          Ca_vir.conj(),Ca_occ,Ca_vir.conj(),Ca_occ
                          ,optimize = True)
        K_aibj_B_hyb_bbbb = numpy.einsum('uvwy,ua,vj,wb,yi->ajbi', eri_ao,
                          Cb_vir.conj(),Cb_occ,Cb_vir.conj(),Cb_occ
                          ,optimize = True)
        
        K_aibj_B_hyb_aaaa = numpy.transpose(K_aibj_B_hyb_aaaa,(0,3,2,1))
        K_aibj_B_hyb_bbbb = numpy.transpose(K_aibj_B_hyb_bbbb,(0,3,2,1))
        
        return ((K_aibj_A_hyb_aaaa,K_aibj_A_hyb_bbbb),(K_aibj_B_hyb_aaaa,K_aibj_B_hyb_bbbb))

def get_hartree_potential_tddft(mol,C_ao):
    # Columb Interaction, which isn't zero in spin-conserved tda case only.
    # eri in ao space
    # import pdb
    # pdb.set_trace()
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = C_ao
    eri_ao = mol.intor('int2e')
    # transform the eri to mo space
    K_aibj_A_hrp_aaaa = numpy.einsum('uvwy,ua,vi,wj,yb->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Ca_occ.conj(),Ca_vir
                          ,optimize = True)
    K_aibj_A_hrp_aabb = numpy.einsum('uvwy,ua,vi,wj,yb->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Cb_occ.conj(),Cb_vir
                          ,optimize = True)
    K_aibj_A_hrp_bbaa = numpy.einsum('uvwy,ua,vi,wj,yb->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Ca_occ.conj(),Ca_vir
                          ,optimize = True)
    K_aibj_A_hrp_bbbb = numpy.einsum('uvwy,ua,vi,wj,yb->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Cb_occ.conj(),Cb_vir
                          ,optimize = True)
    
    K_aibj_B_hrp_aaaa = numpy.einsum('uvwy,ua,vi,wb,yj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Ca_vir.conj(),Ca_occ
                          ,optimize = True)
    K_aibj_B_hrp_aabb = numpy.einsum('uvwy,ua,vi,wb,yj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Cb_vir.conj(),Cb_occ
                          ,optimize = True)
    K_aibj_B_hrp_bbaa = numpy.einsum('uvwy,ua,vi,wb,yj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Ca_vir.conj(),Ca_occ
                          ,optimize = True)
    K_aibj_B_hrp_bbbb = numpy.einsum('uvwy,ua,vi,wb,yj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Cb_vir.conj(),Cb_occ
                          ,optimize = True)
    return ((K_aibj_A_hrp_aaaa,K_aibj_A_hrp_aabb,K_aibj_A_hrp_bbaa,K_aibj_A_hrp_bbbb),
            (K_aibj_B_hrp_aaaa,K_aibj_B_hrp_aabb,K_aibj_B_hrp_bbaa,K_aibj_B_hrp_bbbb))

def get_tddft_Matrix(mf,Ndirect,Extype='SPIN_CONSERVED',LIBXCT_factor=1e-10,ncpu=None):
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
    
    # Get e_ia and the atom and molecular orbitals.
    e_ai,C_mo,C_ao = get_e_ai_and_mo_tddft(mf,xctype,ao,Extype=Extype)
    # mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ -> C_mo
   
    if 'SPIN_FLIP' in Extype:
    #if Extype == 'SPIN_FLIP_UP' or Extype == 'SPIN_FLIP_DOWN':
        K_aibj_A_sfd,K_aibj_B_sfd,K_aibj_A_sfu,K_aibj_B_sfu = (
                            nitdamc.nr_collinear_tddft_mc(xc_code,(rho,s),
                            mf.grids,Ndirect,C_mo,Extype=Extype,
                            LIBXCT_factor=LIBXCT_factor,ncpu=ncpu))
        
        # Hybrid Exchange Energy.
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            K_aibj_hyb_A_sfd = 0.0
            K_aibj_hyb_B_sfd = 0.0
            K_aibj_hyb_A_sfu = 0.0
            K_aibj_hyb_B_sfu = 0.0
        else:
            K_aibj_hyb_A_sfd,K_aibj_hyb_B_sfd,K_aibj_hyb_A_sfu,K_aibj_hyb_B_sfu = (
                                                get_hybrid_exchange_energy_tddft(
                                                mf.mol,C_ao,Extype,omega,alpha,hyb))
        
        # Whole diag for e_ai
        e_ai_b2a,e_ai_a2b = e_ai
        e_ai_b2a = numpy.diag(e_ai_b2a)
        e_ai_a2b = numpy.diag(e_ai_a2b)
        
    
        # 0.5*2 = 1 for the double on calculation of xx,yy parts.    
        # import pdb
        # pdb.set_trace()
        
        # Just test Kroxy's work.
        # K_aibj_A_sfd *= 0.0
        # K_aibj_B_sfd *= 0.0
        # K_aibj_A_sfu *= 0.0
        # K_aibj_B_sfu *= 0.0
        # Test down!
        
        K_aibj_A_sfd -= 0.5*K_aibj_hyb_A_sfd
        K_aibj_B_sfd -= 0.5*K_aibj_hyb_B_sfd
        K_aibj_A_sfu -= 0.5*K_aibj_hyb_A_sfu
        K_aibj_B_sfu -= 0.5*K_aibj_hyb_B_sfu
        
        ndim_a,ndim_i = K_aibj_A_sfd.shape[:2]
        ndim_1 = ndim_a*ndim_i
        ndim_b,ndim_j = K_aibj_B_sfd.shape[-2:]
        ndim_2 = ndim_b*ndim_j
        
        Kmat_A_sfd = K_aibj_A_sfd.reshape((ndim_1,ndim_1),order = 'C')
        Kmat_B_sfd = K_aibj_B_sfd.reshape((ndim_1,ndim_2),order = 'C')
        Kmat_A_sfu = K_aibj_A_sfu.reshape((ndim_2,ndim_2),order = 'C')
        Kmat_B_sfu = K_aibj_B_sfu.reshape((ndim_2,ndim_1),order = 'C')
        
        # *2 means xx,yy parts
        TD_Matx_A_sfd = e_ai_a2b + 2*Kmat_A_sfd
        TD_Matx_B_sfd = 2*Kmat_B_sfd
        TD_Matx_A_sfu = e_ai_b2a + 2*Kmat_A_sfu
        TD_Matx_B_sfu = 2*Kmat_B_sfu
        
        if Extype == 'SPIN_FLIP_UP':
            TD_Matx = numpy.block([[TD_Matx_A_sfu,TD_Matx_B_sfu],
                                   [-TD_Matx_B_sfd,-TD_Matx_A_sfd]])
            
        elif Extype == 'SPIN_FLIP_DOWN':
            TD_Matx = numpy.block([[TD_Matx_A_sfd,TD_Matx_B_sfd],
                                   [-TD_Matx_B_sfu,-TD_Matx_A_sfu]])
            
        return TD_Matx
      
    
    elif Extype =='SPIN_CONSERVED':
        # C_ao is only given in spin-conserved case. 
        e_ai,C_mo,C_ao = get_e_ai_and_mo_tddft(mf,xctype,ao,Extype=Extype)
        e_ai_aa,e_ai_bb = e_ai
     
        K_aibj = nitdamc.nr_collinear_tddft_mc(xc_code,(rho,s),mf.grids,Ndirect,C_mo,
                                       Extype=Extype,ncpu=ncpu)
        K_aibj_A_aaaa,K_aibj_A_aabb,K_aibj_A_bbaa,K_aibj_A_bbbb = K_aibj[0]
        K_aibj_B_aaaa,K_aibj_B_aabb,K_aibj_B_bbaa,K_aibj_B_bbbb = K_aibj[1]
            
        K_aibj_hrp = get_hartree_potential_tddft(mf.mol,C_ao)
        K_aibj_A_hrp_aaaa,K_aibj_A_hrp_aabb,K_aibj_A_hrp_bbaa,K_aibj_A_hrp_bbbb = K_aibj_hrp[0]
        K_aibj_B_hrp_aaaa,K_aibj_B_hrp_aabb,K_aibj_B_hrp_bbaa,K_aibj_B_hrp_bbbb = K_aibj_hrp[1]

        # import pdb
        # pdb.set_trace()
        # Hybrid Exchange Energy.
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            K_aibj_A_hyb_aaaa,K_aibj_A_hyb_bbbb = (0,0)
            K_aibj_B_hyb_aaaa,K_aibj_B_hyb_bbbb = (0,0)
        else:
            K_aibj_hyb = get_hybrid_exchange_energy_tddft(mf.mol,C_ao,Extype,omega,alpha,hyb)
            K_aibj_A_hyb_aaaa,K_aibj_A_hyb_bbbb = K_aibj_hyb[0]
            K_aibj_B_hyb_aaaa,K_aibj_B_hyb_bbbb = K_aibj_hyb[1]
        
        # This part need to simpliation  
        ndim1_a2a,ndim2_a2a = K_aibj_A_aaaa.shape[:2]
        ndim_aaaa = ndim1_a2a*ndim2_a2a
        ndim1_a2b,ndim2_a2b = K_aibj_A_aabb.shape[:2]
        ndim_aabb = ndim1_a2b*ndim2_a2b
        ndim1_b2a,ndim2_b2a = K_aibj_A_bbaa.shape[:2]
        ndim_bbaa = ndim1_b2a*ndim2_b2a
        ndim1_b2b,ndim2_b2b = K_aibj_A_bbbb.shape[:2]
        ndim_bbbb = ndim1_b2b*ndim2_b2b
        
        Kmat_A_aaaa = K_aibj_A_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C')
        Kmat_A_aabb = K_aibj_A_aabb.reshape((ndim_aabb,ndim_bbaa),order='C')
        Kmat_A_bbaa = K_aibj_A_bbaa.reshape((ndim_bbaa,ndim_aabb),order='C')
        Kmat_A_bbbb = K_aibj_A_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
        
        Kmat_B_aaaa = K_aibj_B_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C')
        Kmat_B_aabb = K_aibj_B_aabb.reshape((ndim_aabb,ndim_bbaa),order='C')
        Kmat_B_bbaa = K_aibj_B_bbaa.reshape((ndim_bbaa,ndim_aabb),order='C')
        Kmat_B_bbbb = K_aibj_B_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
        
        # import pdb
        # pdb.set_trace() 
        # Hartree potential parts.
        K_aibj_A_hrp_aaaa = K_aibj_A_hrp_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C')
        K_aibj_A_hrp_aabb = K_aibj_A_hrp_aabb.reshape((ndim_aabb,ndim_bbaa),order='C')
        K_aibj_A_hrp_bbaa = K_aibj_A_hrp_bbaa.reshape((ndim_bbaa,ndim_aabb),order='C')
        K_aibj_A_hrp_bbbb = K_aibj_A_hrp_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
        
        K_aibj_B_hrp_aaaa = K_aibj_B_hrp_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C')
        K_aibj_B_hrp_aabb = K_aibj_B_hrp_aabb.reshape((ndim_aabb,ndim_bbaa),order='C')
        K_aibj_B_hrp_bbaa = K_aibj_B_hrp_bbaa.reshape((ndim_bbaa,ndim_aabb),order='C')
        K_aibj_B_hrp_bbbb = K_aibj_B_hrp_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
        
        # Hartree-Fock Exchange parts.
        if type(K_aibj_A_hyb_aaaa) is numpy.ndarray:
            K_aibj_A_hyb_aaaa = K_aibj_A_hyb_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C') 
            K_aibj_A_hyb_bbbb = K_aibj_A_hyb_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
            K_aibj_B_hyb_aaaa = K_aibj_B_hyb_aaaa.reshape((ndim_aaaa,ndim_aaaa),order='C')
            K_aibj_B_hyb_bbbb = K_aibj_B_hyb_bbbb.reshape((ndim_bbbb,ndim_bbbb),order='C')
         
        # Whole diag for e_ai
        e_ai_aa = numpy.diag(e_ai_aa)
        e_ai_bb = numpy.diag(e_ai_bb)

        Kmat_A_aaaa += K_aibj_A_hrp_aaaa  
        Kmat_A_aabb += K_aibj_A_hrp_aabb  
        Kmat_A_bbaa += K_aibj_A_hrp_bbaa  
        Kmat_A_bbbb += K_aibj_A_hrp_bbbb  
        
        Kmat_B_aaaa += K_aibj_B_hrp_aaaa  
        Kmat_B_aabb += K_aibj_B_hrp_aabb  
        Kmat_B_bbaa += K_aibj_B_hrp_bbaa  
        Kmat_B_bbbb += K_aibj_B_hrp_bbbb
        
        # Debug.
        # Kmat_A_aaaa *= 0.0
        # Kmat_A_aabb *= 0.0
        # Kmat_A_bbaa *= 0.0
        # Kmat_A_bbbb *= 0.0
        # # 
        # Kmat_B_aaaa *=0.0
        # Kmat_B_aabb *=0.0
        # Kmat_B_bbaa *=0.0
        # Kmat_B_bbbb *=0.0
        
        Kmat_A_aaaa -= K_aibj_A_hyb_aaaa
        Kmat_A_bbbb -= K_aibj_A_hyb_bbbb
        Kmat_B_aaaa -= K_aibj_B_hyb_aaaa
        Kmat_B_bbbb -= K_aibj_B_hyb_bbbb
        
        # Add the difference value of orbitals.
        Kmat_A_aaaa += e_ai_aa
        Kmat_A_bbbb += e_ai_bb
        
        numpy.save('Kmat_A',(Kmat_A_aaaa,Kmat_A_aabb,Kmat_A_bbaa,Kmat_A_bbbb))
        numpy.save('Kmat_B',(Kmat_B_aaaa,Kmat_B_aabb,Kmat_B_bbaa,Kmat_B_bbbb))
        
        TD_Matx_A = numpy.block([[Kmat_A_aaaa,Kmat_A_aabb],
                                 [Kmat_A_bbaa,Kmat_A_bbbb]])
        TD_Matx_B = numpy.block([[Kmat_B_aaaa,Kmat_B_aabb],
                                 [Kmat_B_bbaa,Kmat_B_bbbb]])
        
        TD_Matx = numpy.block([[TD_Matx_A,TD_Matx_B],
                               [-TD_Matx_B.conj(),-TD_Matx_A.conj()]])
        
        return TD_Matx

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

def pick_states(self,U,E_ex):
    mf = self.scf
    mo_occ = mf.mo_occ
    nao = mf.mol.nao
    nocca = int(mo_occ[0].sum())
    noccb = int(mo_occ[1].sum())
    nvira = nao - nocca 
    nvirb = nao - noccb
    # import pdb
    # pdb.set_trace()
    if self.Extype == 'SPIN_FLIP_UP':
        ndim_ai = nvira*noccb
    elif self.Extype == 'SPIN_FLIP_DOWN':
        ndim_ai = nvirb*nocca
    elif self.Extype == 'SPIN_CONSERVED':
        # ndim_ai = ndim_ai_aa + ndim_ai_bb
        ndim_ai = nvira*nocca + nvirb*noccb
            
    X = U[:ndim_ai]
    Y = U[ndim_ai:]
    X_norm = numpy.linalg.norm(X,axis=0)
    Y_norm = numpy.linalg.norm(Y,axis=0)
    
    # The idx of excited states, which is determined by the differrnce
    # of the norm for X and Y.  
    idx_X_lead = numpy.where((X_norm-Y_norm)>-1e-4)[0]
    E_ex_X_lead = E_ex[idx_X_lead]
    ### May bug !!!
    ### Finished.
    U_X_lead = U[:,idx_X_lead]
    return E_ex_X_lead,U_X_lead


def eig_tddft(self,TD_Matx,Ndirect,Extype):
    # numpy.linalg.eigh, that is different from TDA because of not symmetric martrix.
    E_ex,U = numpy.linalg.eig(TD_Matx)
    # import pdb
    # pdb.set_trace()
    if self.Pick_states:
        E_ex, U = pick_states(self,U,E_ex)
    
    idx_sort = numpy.argsort(E_ex.real)
    self.Extd = numpy.sort(E_ex.real)*27.21138386
    # self.Extd = numpy.sort(E_ex.real[:E_ex.shape[0]//2])*27.21138386
    # self.Extd = E_ex
    self.U = U[:,idx_sort]
    # if Extype =='SPIN_FLIP_UP' or Extype =='SPIN_FLIP_DOWN':
    #     print('Spin_Space_Sample_Points: '+str(Ndirect))
    # elif Extype =='SPIN_CONSERVED':
    #     pass
    # numpy.save('E_ex_'+ str(Extype) ,E_ex)
    # for i in range(E_ex.shape[-1]):
    #     print(f"{E_ex[i]:16.14f}")

# The functions for calculating the oscillator_strength are from pySCF.
def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return numpy.einsum('z,zr->r', charges, coords)/charges.sum()

def transition_dipole(tdobj, xy=None):
    '''Transition dipole moments in the length gauge'''
    mol = tdobj.scf.mol
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor_symmetric('int1e_r', comp=3)
    return tdobj._contract_multipole(ints, hermi=True, xy=xy)

def oscillator_strength(tdobj,gauge='length'):
    # 只能用全对角化做了，因为非相对论在迭代求解中，用的是 (A-B)(A+B)(X+Y)=w(X+Y)
    # 因为全对角化和迭代求解保存矩阵的指标分别是 ai 和 ia，因此该函数只负责 ai，也就是全对角化。
    mf = tdobj.scf
    mo_occ = mf.mo_occ
    nao = mf.mol.nao
    nocca = int(mo_occ[0].sum())
    noccb = int(mo_occ[1].sum())
    nvira = nao - nocca 
    nvirb = nao - noccb
    # import pdb
    # pdb.set_trace()
    
    E_ex = tdobj.Extd/27.21138386
    e = []
    xy = []
    
    # import  pdb
    # pdb.set_trace()
    if tdobj.Extype == 'SPIN_FLIP_UP' or tdobj.Extype == 'SPIN_FLIP_DOWN':
       raise NotImplementedError('There is no need to calculate oscillator_strength and all values equal to 0.')
           
    
    elif tdobj.Extype == 'SPIN_CONSERVED':
        ndim = nvira*nocca + nvirb*noccb
        x = tdobj.U[:ndim]
        y = tdobj.U[ndim:]
        
        for i in range(len(E_ex)):
            norm = lib.norm(x[:,i])**2 - lib.norm(y[:,i])**2
            if norm > 0:
                norm = 1/numpy.sqrt(norm)
                e.append(E_ex[i])
                xy.append(((x[:nocca*nvira,i].reshape(nvira,nocca).transpose(1,0) *norm,  # X_alpha
                            x[nocca*nvira:,i].reshape(nvirb,noccb).transpose(1,0) *norm), # X_beta
                           (y[:nocca*nvira,i].reshape(nvira,nocca).transpose(1,0) *norm,  # Y_alpha
                            y[nocca*nvira:,i].reshape(nvirb,noccb).transpose(1,0) *norm)))# Y_beta
    tdobj.xy = xy
    tdobj.e = e
    
    if gauge == 'length':
        trans_dip = transition_dipole(tdobj, xy)
        f = 2./3. * numpy.einsum('s,sx,sx->s', e, trans_dip, trans_dip)
        tdobj.f_osas=f
        return f.real
    
    else:
        # velocity gauge
        # Ref. JCP, 143, 234103
        raise NotImplementedError('Calculation for oscillator_strength at velocity gauge has not been implemented.')

class TDDFT_MC_UKS:
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
        # Save the TD_Matx
        self.TD_Matx = None
        # The method of matrix multiplies vector.
        self.method = method
        # The eigh_vector given by AX mtethod. 
        self.U = None
        # Pick the excited states leading by A block of TD_Matrix.add()
        self.Pick_states = True
        
    get_e_ai_and_mo_tddft = get_e_ai_and_mo_tddft
    # get_hartree_potential_tda = get_hartree_potential_tda
    get_hybrid_exchange_energy_tddft = get_hybrid_exchange_energy_tddft
    eig_tddft = eig_tddft
    excited_mag_structure = numint_tdamc.excited_mag_structure

    def kernel(self,Ndirect=None,Extype=None,LIBXCT_factor=None,ncpu=None,Extd=None,nstates=3,
               init_guess=None, max_cycle = 50, conv_tol = 1.0E-8, scheme='LAN', cutoff=8, Whkerl=False,parallel= False,TD_Matx=None):
        # This part should be more smart.
        if Ndirect is None:
            Ndirect = self.Ndirect
        
        # ToDo: Extype should be correct as the self.object.
        if Extype is None:
            Extype = self.Extype.upper()
            self.Extype = Extype
        else:
            Extype = Extype.upper()
            self.Extype = Extype
        if LIBXCT_factor is None:
            LIBXCT_factor = self.LIBXCT_factor
        if ncpu is None:
            self.ncpu = ncpu
        if Extd is None:
            self.Extd = Extd
        # import pdb
        # pdb.set_trace()
        if self.method.upper() == 'AX':
            ao,xctype,kernel = get_kernel(self.scf, Ndirect, Extype, LIBXCT_factor, ncpu)
            sol = tddft_solver.Solver_TDDFT(self.scf,mf2=None,Extype=Extype,kernel=kernel, nstates=nstates, 
                                   init_guess=init_guess, scheme=scheme, max_cycle=max_cycle, 
                                   conv_tol=conv_tol, cutoff = cutoff, Whkerl=Whkerl,parallel=parallel,ncpu=ncpu)
            self.Extd, self.U = sol.solver(ao,xctype)
        else:
            TD_Matx = get_tddft_Matrix(self.scf,Ndirect,Extype,LIBXCT_factor,ncpu)
            if TD_Matx is not None:
                self.TD_Matx = TD_Matx
            eig_tddft(self,TD_Matx,Ndirect,Extype)
            # Amat_tot = get_tda_Amat(self.scf,Ndirect,Extype,LIBXCT_factor,ncpu)
            # if Amat_f is None:
            #     self.Amat_f = Amat_tot
            # eigh_tda(self,Amat_tot,Ndirect,Extype)
    
    oscillator_strength = oscillator_strength    
      
    def _contract_multipole(tdobj, ints, hermi=True, xy=None):
        if xy is None: xy = tdobj.xy
        mo_coeff = tdobj.scf.mo_coeff
        mo_occ = tdobj.scf.mo_occ
        orbo_a = mo_coeff[0][:,mo_occ[0]==1]
        orbv_a = mo_coeff[0][:,mo_occ[0]==0]
        orbo_b = mo_coeff[1][:,mo_occ[1]==1]
        orbv_b = mo_coeff[1][:,mo_occ[1]==0]
        
        ints_a = numpy.einsum('...pq,pi,qj->...ij', ints, orbo_a.conj(), orbv_a)
        ints_b = numpy.einsum('...pq,pi,qj->...ij', ints, orbo_b.conj(), orbv_b)
        pol = [(numpy.einsum('...ij,ij->...', ints_a, x[0]) +
                numpy.einsum('...ij,ij->...', ints_b, x[1])) for x,y in xy]
        pol = numpy.array(pol)
        y = xy[0][1]
        if isinstance(y[0], numpy.ndarray):
            pol_y = [(numpy.einsum('...ij,ij->...', ints_a, y[0]) +
                    numpy.einsum('...ij,ij->...', ints_b, y[1])) for x,y in xy]
            if hermi:
                pol += pol_y
            else:  # anti-Hermitian
                pol -= pol_y

        return pol
    