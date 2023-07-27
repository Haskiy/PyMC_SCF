#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-07-28 18:49:06
LastEditTime: 2023-05-26 08:57:53
LastEditors: Li Hao
Description: Non-Collinear TDDFT

FilePath: /pyMC/tdamc/tddft_mc_gks.py
Motto: A + B = C!
'''
import numpy
import scipy
from pyscf import dft
from pyscf import lib
from pyscf import ao2mo
from pyMC.tdamc import numint_tdamc
from pyMC.tdamc import tddft_solver

def uks_to_gks(mf1):
    dm_uks = mf1.make_rdm1()
    dm_gks = scipy.linalg.block_diag(dm_uks[0],dm_uks[1])
    mf = dft.GKS(mf1.mol)
    # ToDo: IBP need to add.
    # mf.ibp = False
    mf.xc = mf1.xc
    # mf2.grids.level may need give a new character to calculate.
    mf.grids = mf1.grids
    # mf.grids.level = 5
    mf.max_cycle = 0
    mf.kernel(dm_gks)
    return mf

def uks_to_gks_e_ai_and_mo_tddft(mf1,mf2,xctype,ao,diff=1e-4):
    # import pdb
    # pdb.set_trace()
    mo_uks_energy = mf1.mo_energy
    mo_uks_occ = mf1.mo_occ
    idxa_occ = numpy.where(mo_uks_occ[0]>0)[0]
    idxb_occ = numpy.where(mo_uks_occ[1]>0)[0]
    nocca = int(mo_uks_occ[0].sum())
    noccb = int(mo_uks_occ[1].sum())
    
    mo_occ_energy = numpy.zeros((nocca+noccb))
    mo_occ_energy[:nocca] = mo_uks_energy[0][idxa_occ]
    mo_occ_energy[nocca:] = mo_uks_energy[1][idxb_occ]
    
    # mf2 = uks_to_gks(mf1)
    mo_occ = mf2.mo_occ # here shallow copy is used.
    mo_occ *= 0
    mo_energy = mf2.mo_energy
    # diff means the value to judge occ orbitals of gks object based on uks.  
    # TODO: how to give a suitable diff ?
    for i in range(mo_occ_energy.shape[-1]):
        idx_occ = numpy.where(abs(mo_energy - mo_occ_energy[i])<=diff)[0]
        mo_occ[idx_occ] = 1
    # print(mo_occ)
    
    occidx = numpy.where(mo_occ>0)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nao = mf2.mol.nao
    nso = 2*nao
    nocc = int(mo_occ.sum())
    nvir = nso - nocc
    
    e_i = mo_energy[occidx]
    e_a = mo_energy[viridx]
    e_a_i = numpy.array([e_a[vir] - e_i[occ]
                            for vir in range(nvir)
                            for occ in range(nocc)],dtype=numpy.complex128) # 

    # iAmat = numpy.diag(e_i_a) # Δ𝑎𝑖=𝜀(0)𝑎−𝜀(0)𝑖
    C = mf2.mo_coeff
    Ca_occ = C[:nao,occidx]
    Cb_occ = C[nao:,occidx]
    Ca_vir = C[:nao,viridx]
    Cb_vir = C[nao:,viridx]

    # Construct molecular orbitals.
    if xctype == 'LDA':
        mo_a_occ = numpy.einsum('ui,nu->ni',Ca_occ,ao)
        mo_b_occ = numpy.einsum('ui,nu->ni',Cb_occ,ao)
        mo_a_vir = numpy.einsum('ui,nu->ni',Ca_vir,ao)
        mo_b_vir = numpy.einsum('ui,nu->ni',Cb_vir,ao)
    elif xctype == 'GGA':
        mo_a_occ = numpy.einsum('ui,gnu->gni',Ca_occ,ao)[:4]
        mo_b_occ = numpy.einsum('ui,gnu->gni',Cb_occ,ao)[:4]
        mo_a_vir = numpy.einsum('ui,gnu->gni',Ca_vir,ao)[:4]
        mo_b_vir = numpy.einsum('ui,gnu->gni',Cb_vir,ao)[:4]
    return e_a_i,(mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ),(Ca_vir,Ca_occ,Cb_vir,Cb_occ) 

def get_e_ai_and_mo_tddft(mf,xctype,ao):
    # import pdb
    # pdb.set_trace()
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidx = numpy.where(mo_occ>0)[0]
    viridx = numpy.where(mo_occ==0)[0]
    
    nao = mf.mol.nao
    nso = 2*nao
    nocc = int(mo_occ.sum())
    nvir = nso - nocc
    
    e_i = mo_energy[occidx]
    e_a = mo_energy[viridx]
    # Δ𝑎𝑖=𝜀(0)𝑎−𝜀(0)𝑖
    e_a_i = numpy.array([e_a[vir] - e_i[occ]
                            for vir in range(nvir)
                            for occ in range(nocc)],dtype=numpy.complex128) # 
    # iAmat = numpy.diag(e_i_a) 
    C = mf.mo_coeff
    Ca_occ = C[:nao,occidx]
    Cb_occ = C[nao:,occidx]
    Ca_vir = C[:nao,viridx]
    Cb_vir = C[nao:,viridx]
    
    # import pdb
    # pdb.set_trace()

    # Construct molecular orbitals.
    if xctype == 'LDA':
        mo_a_occ = numpy.einsum('ui,nu->ni',Ca_occ,ao)
        mo_b_occ = numpy.einsum('ui,nu->ni',Cb_occ,ao)
        mo_a_vir = numpy.einsum('ui,nu->ni',Ca_vir,ao)
        mo_b_vir = numpy.einsum('ui,nu->ni',Cb_vir,ao)
    elif xctype == 'GGA':
        mo_a_occ = numpy.einsum('ui,gnu->gni',Ca_occ,ao)[:4]
        mo_b_occ = numpy.einsum('ui,gnu->gni',Cb_occ,ao)[:4]
        mo_a_vir = numpy.einsum('ui,gnu->gni',Ca_vir,ao)[:4]
        mo_b_vir = numpy.einsum('ui,gnu->gni',Cb_vir,ao)[:4]
        
    return e_a_i,(mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ),(Ca_vir,Ca_occ,Cb_vir,Cb_occ) 

def get_e_ai_and_mo_tddft_r(mf,xctype,ao):
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    n4c = mo_occ.shape[-1]
    occidx = numpy.where(mo_occ>0)[0]
    viridx = numpy.where(mo_occ==0)[0][n4c//2:]
    
    nao = mf.mol.nao
    nso = 2*nao
    nocc = int(mo_occ.sum())
    nvir = nso - nocc
    
    e_i = mo_energy[occidx]
    e_a = mo_energy[viridx]
    e_a_i = numpy.array([e_a[vir] - e_i[occ]
                            for vir in range(nvir)
                            for occ in range(nocc)],dtype=numpy.complex128) # 
    # iAmat = numpy.diag(e_i_a) # Δ𝑎𝑖=𝜀(0)𝑎−𝜀(0)𝑖
    # import pdb
    # pdb.set_trace()
    C = mf.mo_coeff
    C_occ = C[:,occidx]
    C_vir = C[:,viridx]

    # Construct molecular orbitals.
    mo_occ_L = numpy.einsum('ui,cxpu->cxpi',C_occ[:nso],ao[:2])
    mo_occ_S = numpy.einsum('ui,cxpu->cxpi',C_occ[nso:],ao[2:])
    mo_vir_L = numpy.einsum('ua,cxpu->cxpa',C_vir[:nso],ao[:2])
    mo_vir_S = numpy.einsum('ua,cxpu->cxpa',C_vir[nso:],ao[2:])
        
    return e_a_i, (mo_vir_L, mo_vir_S, mo_occ_L, mo_occ_S), (C_vir, C_occ)

def get_hartree_potential_tddft_r_base_mo(mf):
    # Copy pyscf parts.
    # import pdb
    # pdb.set_trace()
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    
    nao, nmo = mo_coeff.shape
    n2c = nmo // 2
    occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
    viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    nmo = nocc + nvir
    mo = numpy.hstack((orbo, orbv))
    c1 = .5 / lib.param.LIGHT_SPEED
    moL = numpy.asarray(mo[:n2c], order='F')
    moS = numpy.asarray(mo[n2c:], order='F') * c1
    orboL = moL[:,:nocc]
    orboS = moS[:,:nocc]

    # import pdb
    # pdb.set_trace()
    
    eri_mo = ao2mo.kernel(mol, [orboL, moL, moL, moL], intor='int2e_spinor')
    eri_mo+= ao2mo.kernel(mol, [orboS, moS, moS, moS], intor='int2e_spsp1spsp2_spinor')
    eri_mo+= ao2mo.kernel(mol, [orboS, moS, moL, moL], intor='int2e_spsp1_spinor')
    eri_mo+= ao2mo.kernel(mol, [moS, moS, orboL, moL], intor='int2e_spsp1_spinor').T
    eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
    
    K_aibj_A_hrp = numpy.einsum('iabj->bjai', eri_mo[:nocc,nocc:,nocc:,:nocc])
    K_aibj_B_hrp = numpy.einsum('iajb->bjai', eri_mo[:nocc,nocc:,:nocc,nocc:]).conj()
    return K_aibj_A_hrp,K_aibj_B_hrp

def get_hybrid_exchange_energy_tddft_r_base_mo(mf):
    # Copy pyscf parts.
    # import pdb
    # pdb.set_trace()
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    
    nao, nmo = mo_coeff.shape
    n2c = nmo // 2
    occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
    viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    nmo = nocc + nvir
    mo = numpy.hstack((orbo, orbv))
    c1 = .5 / lib.param.LIGHT_SPEED
    moL = numpy.asarray(mo[:n2c], order='F')
    moS = numpy.asarray(mo[n2c:], order='F') * c1
    orboL = moL[:,:nocc]
    orboS = moS[:,:nocc]

    eri_mo = ao2mo.kernel(mol, [orboL, moL, moL, moL], intor='int2e_spinor')
    eri_mo+= ao2mo.kernel(mol, [orboS, moS, moS, moS], intor='int2e_spsp1spsp2_spinor')
    eri_mo+= ao2mo.kernel(mol, [orboS, moS, moL, moL], intor='int2e_spsp1_spinor')
    eri_mo+= ao2mo.kernel(mol, [moS, moS, orboL, moL], intor='int2e_spsp1_spinor').T
    eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
    
    K_aibj_A_hye = numpy.einsum('ijba->bjai', eri_mo[:nocc,:nocc,nocc:,nocc:])
    K_aibj_B_hye = numpy.einsum('jaib->bjai', eri_mo[:nocc,nocc:,:nocc,nocc:]).conj()
    return K_aibj_A_hye,K_aibj_B_hye

def get_hybrid_exchange_energy_tddft(mol,C_ao,omega,alpha,hyb):
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = C_ao
    eri_ao = mol.intor('int2e').astype(numpy.complex128)
    eri_ao*=hyb
    
    if abs(omega) >= 1e-10:
        with mol.with_range_coulomb(omega=omega):
            eri_ao+= mol.intor('int2e')*(alpha-hyb)
            
    # transform the eri to mo space.
    K_aibj_A_hyb = numpy.einsum('uvwy,ua,vb,yi,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_vir,Ca_occ,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_A_hyb += numpy.einsum('uvwy,ua,vb,yi,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_vir,Cb_occ,Cb_occ.conj()
                          ,optimize = True)
    K_aibj_A_hyb += numpy.einsum('uvwy,ua,vb,yi,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_vir,Ca_occ,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_A_hyb += numpy.einsum('uvwy,ua,vb,yi,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_vir,Cb_occ,Cb_occ.conj()
                          ,optimize = True)
    # K_aibj_A_hyb = numpy.transpose(K_aibj_A_hyb,(0,2,1,3))
    
    K_aibj_B_hyb = numpy.einsum('uvwy,ua,vj,wb,yi->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Ca_vir.conj(),Ca_occ
                          ,optimize = True)
    K_aibj_B_hyb += numpy.einsum('uvwy,ua,vj,wb,yi->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Cb_vir.conj(),Cb_occ
                          ,optimize = True)
    K_aibj_B_hyb += numpy.einsum('uvwy,ua,vj,wb,yi->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Ca_vir.conj(),Ca_occ
                          ,optimize = True)
    K_aibj_B_hyb += numpy.einsum('uvwy,ua,vj,wb,yi->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Cb_vir.conj(),Cb_occ
                          ,optimize = True)
    
    return K_aibj_A_hyb,K_aibj_B_hyb

def get_hybrid_exchange_energy_tddft_r(mol,C_ao,omega,alpha,hyb):
    # import pdb
    # pdb.set_trace()
    C_vir, C_occ = C_ao
    # spinor -> large component, spsp1 -> small component.
    eri_LL = mol.intor('int2e_spinor')*hyb
    eri_LS = mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2*hyb
    eri_SS = mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4*hyb
    # import pdb
    # pdb.set_trace()
    if abs(omega) >= 1e-10:
        with mol.with_range_coulomb(omega=omega):
            eri_LL += eri_LL*alpha/hyb
            eri_LS += eri_LS*alpha/hyb
            eri_SS += eri_SS*alpha/hyb

    n2c = C_occ.shape[0]//2
    
    K_aibj_A_hyb = numpy.einsum('uvwy,ua,vb,yi,wj->aibj', eri_LL,
                          C_vir[:n2c].conj(),C_vir[:n2c],C_occ[:n2c],C_occ[:n2c].conj()
                          ,optimize = True)
    K_aibj_A_hyb += numpy.einsum('uvwy,ua,vb,yi,wj->aibj', eri_LS,
                          C_vir[:n2c].conj(),C_vir[:n2c],C_occ[n2c:],C_occ[n2c:].conj()
                          ,optimize = True)
    K_aibj_A_hyb += numpy.einsum('uvwy,ua,vb,yi,wj->aibj', eri_LS.transpose(2,3,0,1),
                          C_vir[n2c:].conj(),C_vir[n2c:],C_occ[:n2c],C_occ[:n2c].conj()
                          ,optimize = True)
    K_aibj_A_hyb += numpy.einsum('uvwy,ua,vb,yi,wj->aibj', eri_SS,
                          C_vir[n2c:].conj(),C_vir[n2c:],C_occ[n2c:],C_occ[n2c:].conj()
                          ,optimize = True)
    
    K_aibj_B_hyb = numpy.einsum('uvwy,ua,vj,wb,yi->aibj', eri_LL,
                          C_vir[:n2c].conj(),C_occ[:n2c],C_vir[:n2c].conj(),C_occ[:n2c]
                          ,optimize = True)
    K_aibj_B_hyb += numpy.einsum('uvwy,ua,vj,wb,yi->aibj', eri_LS,
                          C_vir[:n2c].conj(),C_occ[:n2c],C_vir[n2c:].conj(),C_occ[n2c:]
                          ,optimize = True)
    K_aibj_B_hyb += numpy.einsum('uvwy,ua,vj,wb,yi->aibj', eri_LS.transpose(2,3,0,1),
                          C_vir[n2c:].conj(),C_occ[n2c:],C_vir[:n2c].conj(),C_occ[:n2c]
                          ,optimize = True)
    K_aibj_B_hyb += numpy.einsum('uvwy,ua,vj,wb,yi->aibj', eri_SS,
                          C_vir[n2c:].conj(),C_occ[n2c:],C_vir[n2c:].conj(),C_occ[n2c:]
                          ,optimize = True)
    
    return K_aibj_A_hyb,K_aibj_B_hyb
  
def get_hartree_potential_tddft(mol,C_ao):
    # Columb Interaction, which isn't zero in spin-conserved tda case only.
    # eri in ao space
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = C_ao
    eri_ao = mol.intor('int2e').astype(numpy.complex128)
    # transform the eri to mo space.
    # a complex implementation -> simpltation. 
    K_aibj_A_hrp = numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Ca_vir,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_A_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Cb_vir,Cb_occ.conj()
                          ,optimize = True)
    K_aibj_A_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Ca_vir,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_A_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Cb_vir,Cb_occ.conj()
                          ,optimize = True)
    
    K_aibj_B_hrp = numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Ca_vir.conj(),Ca_occ
                          ,optimize = True)
    K_aibj_B_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Cb_vir.conj(),Cb_occ
                          ,optimize = True)
    K_aibj_B_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Ca_vir.conj(),Ca_occ
                          ,optimize = True)
    K_aibj_B_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Cb_vir.conj(),Cb_occ
                          ,optimize = True)
    
    return K_aibj_A_hrp,K_aibj_B_hrp

def get_hartree_potential_tddft_r(mol, C):
    # import pdb
    # pdb.set_trace()
    C_vir, C_occ = C
    # Columb Interaction, which isn't zero in spin-conserved tda case only.
    # eri in ao space
    # spinor -> large component, spsp1 -> small component.
    eri_LL = mol.intor('int2e_spinor')
    eri_LS = mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2
    eri_SS = mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4
    # transform the eri to mo space.
    n2c = C_occ.shape[0]//2
    
    K_aibj_A_hrp = numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_LL,
                          C_vir[:n2c].conj(),C_occ[:n2c],C_vir[:n2c],C_occ[:n2c].conj()
                          ,optimize = True)
    K_aibj_A_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_LS,
                          C_vir[:n2c].conj(),C_occ[:n2c],C_vir[n2c:],C_occ[n2c:].conj()
                          ,optimize = True)
    K_aibj_A_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_LS.transpose(2,3,0,1),
                          C_vir[n2c:].conj(),C_occ[n2c:],C_vir[:n2c],C_occ[:n2c].conj()
                          ,optimize = True)
    K_aibj_A_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_SS,
                          C_vir[n2c:].conj(),C_occ[n2c:],C_vir[n2c:],C_occ[n2c:].conj()
                          ,optimize = True)
    
    K_aibj_B_hrp  = numpy.einsum('uvwy,ua,vi,wb,yj->aibj', eri_LL,
                          C_vir[:n2c].conj(),C_occ[:n2c],C_vir[:n2c].conj(),C_occ[:n2c]
                          ,optimize = True)
    K_aibj_B_hrp += numpy.einsum('uvwy,ua,vi,wb,yj->aibj', eri_LS,
                          C_vir[:n2c].conj(),C_occ[:n2c],C_vir[n2c:].conj(),C_occ[n2c:]
                          ,optimize = True)
    K_aibj_B_hrp += numpy.einsum('uvwy,ua,vi,wb,yj->aibj', eri_LS.transpose(2,3,0,1),
                          C_vir[n2c:].conj(),C_occ[n2c:],C_vir[:n2c].conj(),C_occ[:n2c]
                          ,optimize = True)
    K_aibj_B_hrp += numpy.einsum('uvwy,ua,vi,wb,yj->aibj', eri_SS,
                          C_vir[n2c:].conj(),C_occ[n2c:],C_vir[n2c:].conj(),C_occ[n2c:]
                          ,optimize = True)
    
    return K_aibj_A_hrp,K_aibj_B_hrp

def get_tddft_Matrix_r(mf,diff,Ndirect=None,Ndirect_lc=None,MSL_factor=None,LIBXCT_factor=None,ncpu=None):
    # No need for diff.
    nir = dft.r_numint.RNumInt()
    nitdamc = numint_tdamc.numint_tdamc()
    xctype = nir._xc_type(mf.xc)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    elif xctype == 'MGGA':
        deriv = 1

    mol = mf.mol
    ao = nir.eval_ao(mol, mf.grids.coords, deriv=deriv)
    if xctype == 'LDA':
        ao = numpy.expand_dims(ao,1)

    e_ai, mo, C = get_e_ai_and_mo_tddft_r(mf,xctype,ao)
    
    K_aibj_A_hrp,K_aibj_B_hrp = get_hartree_potential_tddft_r(mol,C)
    # K_aibj_A_hrp,K_aibj_B_hrp = get_hartree_potential_tddft_r_base_mo(mf)
    # enabling range-separated hybrids
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin)
    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        K_aibj_A_hyb,K_aibj_B_hyb = (0.0,0.0)
    else:
        K_aibj_A_hyb,K_aibj_B_hyb = get_hybrid_exchange_energy_tddft_r(mf.mol,C,omega, alpha, hyb)
        # K_aibj_A_hyb,K_aibj_B_hyb = get_hybrid_exchange_energy_tddft_r_base_mo(mf)
        # K_aibj_A_hyb *= hyb
        # K_aibj_B_hyb *= hyb
    dms = mf.make_rdm1()
    # import pdb
    # pdb.set_trace()
    K_aibj_A,K_aibj_B = nitdamc.r_noncollinear_tddft_mc(nir, mol,  mf.xc, mf.grids, dms, mo, Ndirect=Ndirect,
                                       Ndirect_lc=Ndirect_lc, MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
    
    # K_aibj_A = numpy.einsum('aibj->bjai',K_aibj_A)
    # K_aibj_B = numpy.einsum('aibj->bjai',K_aibj_A)
    
    K_aibj_A += K_aibj_A_hrp
    K_aibj_B += K_aibj_B_hrp
    
    # K_aibj_A = K_aibj_A + K_aibj_A_hrp.transpose(2,3,0,1)
    # K_aibj_B = K_aibj_B_hrp #K_aibj_B.transpose(2,3,0,1)*0.0 + 
    
    K_aibj_A -= K_aibj_A_hyb
    K_aibj_B -= K_aibj_B_hyb
        
    # K_aibj.reshape() -> Kmat
    ndim1,ndim2 = K_aibj_A.shape[:2]
    ndim = ndim1*ndim2
    Kmat_A = K_aibj_A.reshape((ndim,ndim),order='C')
    Kmat_B = K_aibj_B.reshape((ndim,ndim),order='C')
  
    # import pdb
    # pdb.set_trace()
    e_ai = numpy.diag(e_ai)
    TD_Matx_A = Kmat_A + e_ai
    TD_Matx_B = Kmat_B
    
    TD_Matx = numpy.block([[TD_Matx_A,TD_Matx_B],
                           [-TD_Matx_B.conj(),-TD_Matx_A.conj()]])
    return TD_Matx

def get_tddft_Matrix(mf,mf2,diff,Ndirect=None,Ndirect_lc=None,MSL_factor=None,LIBXCT_factor=None,ncpu=None):
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
    # import pdb
    # pdb.set_trace()
    # mf2 -> uks_to_gks object.
    if mf2 is not None:
        e_ai, C_mo, C_ao = uks_to_gks_e_ai_and_mo_tddft(mf,mf2,xctype,ao,diff)
        mf=mf2
    else:
        e_ai, C_mo, C_ao = get_e_ai_and_mo_tddft(mf,xctype,ao)
    
    K_aibj_A_hrp,K_aibj_B_hrp = get_hartree_potential_tddft(mol,C_ao)
    
    # enabling range-separated hybrids
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin)
    
    # Hybrid Exchange Energy.
    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        K_aibj_A_hyb,K_aibj_B_hyb = (0.0,0.0)
    else:
        K_aibj_A_hyb,K_aibj_B_hyb = get_hybrid_exchange_energy_tddft(mf.mol,C_ao,omega, alpha, hyb)

    dmi = mf.make_rdm1()
    dmaa = dmi[:nao,:nao]
    dmab = dmi[:nao,nao:]
    dmba = dmi[nao:,:nao]
    dmbb = dmi[nao:,nao:]
    
    # import pdb
    # pdb.set_trace()
    K_aibj_A,K_aibj_B = nitdamc.nr_noncollinear_tddft_mc(mol, mf.xc, mf.grids, (dmaa,dmab,dmba,dmbb), C_mo, Ndirect=Ndirect,
                                           Ndirect_lc=Ndirect_lc, MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
    K_aibj_A += K_aibj_A_hrp
    K_aibj_B += K_aibj_B_hrp
    
    K_aibj_A -= K_aibj_A_hyb
    K_aibj_B -= K_aibj_B_hyb
    
    # K_aibj.reshape() -> Kmat
    ndim1,ndim2 = K_aibj_A.shape[:2]
    ndim = ndim1*ndim2
    Kmat_A = K_aibj_A.reshape((ndim,ndim),order = 'C')
    Kmat_B = K_aibj_B.reshape((ndim,ndim),order = 'C')
    # import pdb
    # pdb.set_trace()
    e_ai = numpy.diag(e_ai)
    TD_Matx_A = Kmat_A + e_ai
    TD_Matx_B = Kmat_B
    
    TD_Matx = numpy.block([[TD_Matx_A,TD_Matx_B],
                           [-TD_Matx_B.conj(),-TD_Matx_A.conj()]])
    return TD_Matx

def get_kernel(mf,mf2,diff,Ndirect=None,Ndirect_lc=None,MSL_factor=None,LIBXCT_factor=None,ncpu=None):
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

    # import pdb
    # pdb.set_trace()
    # mf2 -> uks_to_gks object.
    if mf2 is not None:
        e_ai, ais, uvs = uks_to_gks_e_ai_and_mo_tddft(mf,mf2,xctype,ao,diff)
        mf = mf2
    else:
        # e_ai, ais, uvs = get_iAmat_and_mo_tda(mf,xctype,ao,diff)
        pass
    
    dmi = mf.make_rdm1()
    dmaa = dmi[:nao,:nao]
    dmab = dmi[:nao,nao:]
    dmba = dmi[nao:,:nao]
    dmbb = dmi[nao:,nao:]

    # import pdb
    # pdb.set_trace()
    
    kernel = nitdamc.noncollinear_tdamc_kernel(mol, mf.xc, mf.grids, (dmaa,dmab,dmba,dmbb), Ndirect=Ndirect,
                                           Ndirect_lc=Ndirect_lc, MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
    
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin)
    
    return ao, xctype, (kernel,(omega, alpha, hyb))

def get_kernel_r(mf,Ndirect=None,Ndirect_lc=None,MSL_factor=None,LIBXCT_factor=None,ncpu=None):
    # No need for diff.
    nitdamc = numint_tdamc.numint_tdamc()
    nir = dft.r_numint.RNumInt()
    xctype = nir._xc_type(mf.xc)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    elif xctype == 'MGGA':
        deriv = 1

    mol = mf.mol
    ao = nir.eval_ao(mol, mf.grids.coords, deriv=deriv)
    if xctype == 'LDA':
        ao = numpy.expand_dims(ao,1)
   
    dmr = mf.make_rdm1()
    
    kernel = nitdamc.r_noncollinear_tdamc_kernel(mol, nir, mf.xc, mf.grids, dmr, Ndirect=Ndirect,Ndirect_lc=Ndirect_lc,
                                                 MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin) 
    
    return ao, xctype, (kernel,(omega, alpha, hyb))

def pick_states(self,mf2,U,E_ex):
    # import pdb
    # pdb.set_trace()
    mf = self.scf
    if mf2 is not None:
        mf = mf2
    # mf = self.scf
    nso = mf.mol.nao*2
    nocc = int(mf.mo_occ.sum())
    nvir = nso - nocc
    ndim_ai = nvir*nocc
  
    X = U[:ndim_ai]
    Y = U[ndim_ai:]
    X_norm = numpy.linalg.norm(X,axis=0)
    Y_norm = numpy.linalg.norm(Y,axis=0)
    
    # The idx of excited states, which is determined by the differrnce
    # of the norm for X and Y.  
    idx_X_lead = numpy.where((X_norm-Y_norm)>-1e-4)[0]
    E_ex_X_lead = E_ex[idx_X_lead]
    U_X_lead = U[:,idx_X_lead]
    return E_ex_X_lead,U_X_lead

def eig_tddft(self,mf2,TD_Matx,Ndirect,Ndirect_lc):
    E_ex,U = numpy.linalg.eig(TD_Matx)
    self.E_ex_nsort,self.U_nsort= E_ex,U
    
    print('################## \n')
    print(TD_Matx)
    if self.Pick_states:
        E_ex, U = pick_states(self,mf2,U,E_ex)
    
    print('2D_Spin_Space_Sample_Points: '+str(Ndirect))
    print('1D_Spin_Space_Sample_Points: '+str(Ndirect_lc))
    # numpy.save('E_ex',E_ex)
    self.Extd = numpy.sort(E_ex.real)*27.21138386
    self.U = U
    
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
    # ToDo: for UKS 对象。
    mf = tdobj.scf
    nso = mf.mol.nao*2
    nocc = int(mf.mo_occ.sum())
    nvir = nso - nocc
    
    E_ex = tdobj.E_ex_nsort
    e = []
    xy = []
    
    import  pdb
    pdb.set_trace()
    if tdobj.Extype == 'DKS':
       raise NotImplementedError('Calculation for oscillator_strength at DKS has not been implemented.')
           
    else:
        ndim_ia = nvir*nocc
        x = tdobj.U_nsort[:ndim_ia]
        y = tdobj.U_nsort[ndim_ia:]
        
        for i in range(len(E_ex)):
            norm = lib.norm(x[:,i])**2 - lib.norm(y[:,i])**2
            if norm > 0:
                norm = 1/numpy.sqrt(norm)
                e.append(E_ex[i].real)
                xy.append((x[:,i].reshape(nvir,nocc).transpose(1,0) *norm,  # X
                           y[:,i].reshape(nvir,nocc).transpose(1,0) *norm)) # Y
    tdobj.xy = xy
    tdobj.e = e
    
    if gauge == 'length':
        trans_dip = transition_dipole(tdobj, xy)
        f = 2./3. * numpy.einsum('s,sx,sx->s', e, trans_dip, trans_dip.conj())
        tdobj.f_osas=f.real
        
        idx = numpy.where(numpy.array(tdobj.e).real>0)[0]
        ids = numpy.argsort(numpy.array(tdobj.e)[idx].real)
        tdobj.e = numpy.sort(numpy.array(tdobj.e)[idx].real)*27.21138386
        tdobj.f_osas = numpy.array(tdobj.f_osas.real)[idx][ids]
        return f.real
    
    else:
        # velocity gauge
        # Ref. JCP, 143, 234103
        raise NotImplementedError('Calculation for oscillator_strength at velocity gauge has not been implemented.')

          
class TDDFT_MC_GKS:
    def __init__(self,mf,method='AX'):
        # Pre-scf-calculate results: mf:gks,uks.add()
        # uks object will be transform into gks object.
        self.scf = mf
        # Number of sample points on the surface.
        # Based on the Lebedev method.
        self.Ndirect = None
        # Locally collinear 1D Spin-space picking points.
        # Based on the Gauss-legendre method.
        self.Ndirect_lc = 0
        # Numstable Appraoch: Threshold and locally collinear approach.
        self.LIBXCT_factor = -1
        self.MSL_factor = None
        # uks_to_gks: diff value.
        self.diff = 1e-4  
        self.ncpu = None
        # Store Excited energy.
        self.Extd = None
        # Save the  A-matrix
        self.TD_Matx = None
        # The method to whether uses matrix-vector multipiation or not. 
        self.method = method
        self.Extype = 'GKS'
        self.U = None
        # Pick the excited states leading by A block of TD_Matrix.add()
        self.Pick_states = True
        
    get_e_ai_and_mo_tddft = get_e_ai_and_mo_tddft
    get_hartree_potential = get_hartree_potential_tddft
    get_hartree_potential_r = get_hartree_potential_tddft_r
    get_hybrid_exchange_energy_tddft = get_hybrid_exchange_energy_tddft
    eig_tddft = eig_tddft
    excited_mag_structure = numint_tdamc.excited_mag_structure

    def kernel(self, mf2=None,diff=None,Ndirect=None, Ndirect_lc=None, MSL_factor=None,
               LIBXCT_factor=None,ncpu=None,Extd=None,TD_Matx=None,nstates=3, parallel= False,
               init_guess=None, max_cycle = 50, conv_tol = 1.0E-8, scheme='LAN', cutoff=8,Whkerl=False,Extype=None):
        # This part should be more smart.
        # import pdb
        # pdb.set_trace()
        if isinstance(self.scf,dft.uks.UKS) or isinstance(self.scf,dft.uks_symm.SymAdaptedUKS):
            mf2 = uks_to_gks(self.scf)
        if Ndirect is None:
            Ndirect = self.Ndirect
        if Ndirect_lc is None:
            Ndirect_lc = self.Ndirect_lc
        if LIBXCT_factor is None:
            LIBXCT_factor = self.LIBXCT_factor
        if MSL_factor is None:
            MSL_factor = self.MSL_factor
            if MSL_factor is not None:
                assert (Ndirect_lc > 0 and isinstance(Ndirect_lc,int)),'Ndirect_lc should be a positive integral.'
        if diff is None:
            diff = self.diff
        if ncpu is None:
            ncpu = self.ncpu
        if Extd is None:
            Extd = self.Extd
        if Extype is None:
            Extype = self.Extype.upper()
        
        # import pdb
        # pdb.set_trace()
        if self.method == 'AX':
            if Extype == 'DKS':
                ao, xctype, kernel = get_kernel_r(self.scf, Ndirect, Ndirect_lc, MSL_factor, LIBXCT_factor, ncpu)
            else:
                ao, xctype, kernel = get_kernel(self.scf, mf2, diff, Ndirect, Ndirect_lc, MSL_factor, LIBXCT_factor, ncpu)
            sol = tddft_solver.Solver_TDDFT(self.scf,mf2,Extype, kernel=kernel, nstates=nstates, 
                                   init_guess=init_guess, scheme=scheme, max_cycle=max_cycle, 
                                   conv_tol=conv_tol, cutoff = cutoff,Whkerl=Whkerl,parallel=parallel,ncpu=ncpu)
            self.Extd, self.U = sol.solver(ao,xctype)
            # For calculating oscillator_strength.
            self.ApBz0 = sol.ApBz0
        else:
            if Extype == 'GKS':
                TD_Matx = get_tddft_Matrix(self.scf,mf2,diff,Ndirect,Ndirect_lc,MSL_factor,LIBXCT_factor,ncpu)
            
            elif Extype == 'DKS':
                TD_Matx = get_tddft_Matrix_r(self.scf,diff,Ndirect,Ndirect_lc,MSL_factor,LIBXCT_factor,ncpu)
            
            
            self.TD_Matx = TD_Matx
            # import pdb
            # pdb.set_trace()
            eig_tddft(self,mf2,TD_Matx,Ndirect,Ndirect_lc)
            
    oscillator_strength = oscillator_strength    
      
    def _contract_multipole(tdobj, ints, hermi=True, xy=None):
        if xy is None: xy = tdobj.xy
        mo_coeff = tdobj.scf.mo_coeff
        mo_occ = tdobj.scf.mo_occ
        orbo = mo_coeff[:,mo_occ==1]
        orbv = mo_coeff[:,mo_occ==0]
        import pdb
        pdb.set_trace()
        
        nao = ints.shape[-1]
        ints_g = numpy.zeros((3,2*nao,2*nao))
        for i in range(3):
            ints_g[i] += scipy.linalg.block_diag(ints[i],ints[i])
        ints_gc = numpy.einsum('...pq,pi,qj->...ij', ints_g, orbo.conj(), orbv)
        pol = [(numpy.einsum('...ij,ij->...', ints_gc, x)) for x,y in xy]            
        pol = numpy.array(pol)
        y = xy[1]
        if isinstance(y[0], numpy.ndarray):
            ints_gc = numpy.einsum('...pq,pi,qj->...ij', ints_g, orbo, orbv.conj())
            pol_y = [(numpy.einsum('...ij,ij->...', ints_gc, y)) for x,y in xy]
            if hermi:
                pol += pol_y
            else:  # anti-Hermitian
                pol -= pol_y
        
        return pol

        
    if __name__ == '__main__':
        from pyscf import gto,dft
        from pyMC.tdamc import tddft_mc_gks
        mol = gto.Mole()
        mol.verbose = 6
        mol.output = '/dev/null'
        mol.atom.extend([['He', (0.,0.,0.)], ])
        mol.basis = { 'He': 'cc-pvdz'}
        mol.build()

        mf = dft.GKS(mol)
        mf.xc = 'pbe'
        mf.kernel()
        
        mf_tdamc = tddft_mc_gks.TDDFT_MC_GKS(mf)
        mf.Ndirect = 266
        mf_tdamc.kernel()
        
    