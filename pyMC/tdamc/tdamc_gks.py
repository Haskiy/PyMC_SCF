#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-03-17 14:23:06
LastEditTime: 2023-02-25 08:54:30
LastEditors: Li Hao
Description: Non-Collinear TDA

FilePath: /pyMC/tdamc/tdamc_gks.py
Motto: A + B = C!
'''
import numpy
import scipy
from pyscf import dft
from pyscf import lib
from pyscf import ao2mo
from pyMC.tdamc import numint_tdamc
from pyMC.tdamc import solver

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

def uks_to_gks_iAamt_and_mo_tda(mf1,mf2,xctype,ao,diff=1e-4):
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
    # import pdb
    # pdb.set_trace()
    occidx = numpy.where(mo_occ>0)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nao = mf2.mol.nao
    nso = 2*nao
    nocc = int(mo_occ.sum())
    nvir = nso - nocc
    
    e_i = mo_energy[occidx]
    e_a = mo_energy[viridx]
    e_i_a = numpy.array([e_a[vir] - e_i[occ]
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
    return e_i_a,(mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ),(Ca_vir,Ca_occ,Cb_vir,Cb_occ) 

def get_iAmat_and_mo_tda(mf,xctype,ao):
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
    e_i_a = numpy.array([e_a[vir] - e_i[occ]
                            for vir in range(nvir)
                            for occ in range(nocc)],dtype=numpy.complex128) # 
    # iAmat = numpy.diag(e_i_a) 
    C = mf.mo_coeff
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
        
    return e_i_a,(mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ),(Ca_vir,Ca_occ,Cb_vir,Cb_occ) 

def get_iAmat_and_mo_tda_r(mf,xctype,ao):
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
    e_i_a = numpy.array([e_a[vir] - e_i[occ]
                            for vir in range(nvir)
                            for occ in range(nocc)],dtype=numpy.complex128) # 
    # iAmat = numpy.diag(e_i_a) # Δ𝑎𝑖=𝜀(0)𝑎−𝜀(0)𝑖
    C = mf.mo_coeff
    C_occ = C[:,occidx]
    C_vir = C[:,viridx]

    # Construct molecular orbitals.
    mo_occ_L = numpy.einsum('ui,cxpu->cxpi',C_occ[:nso],ao[:2])
    mo_occ_S = numpy.einsum('ui,cxpu->cxpi',C_occ[nso:],ao[2:])
    mo_vir_L = numpy.einsum('ua,cxpu->cxpa',C_vir[:nso],ao[:2])
    mo_vir_S = numpy.einsum('ua,cxpu->cxpa',C_vir[nso:],ao[2:])
        
    return e_i_a, (mo_vir_L, mo_vir_S, mo_occ_L, mo_occ_S), (C_vir, C_occ)

def get_hybrid_exchange_energy_tda(mol,C_ao,omega,alpha,hyb):
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = C_ao
    eri_ao = mol.intor('int2e').astype(numpy.complex128)
    eri_ao*=hyb
    
    if abs(omega) >= 1e-10:
        with mol.with_range_coulomb(omega=omega):
            eri_ao+= mol.intor('int2e')*(alpha-hyb)
            
    # transform the eri to mo space.
    K_aibj_hyb = numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Ca_vir.conj(),Ca_vir,Ca_occ,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_hyb += numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Ca_vir.conj(),Ca_vir,Cb_occ,Cb_occ.conj()
                          ,optimize = True)
    K_aibj_hyb += numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Cb_vir.conj(),Cb_vir,Ca_occ,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_hyb += numpy.einsum('uvwy,ua,vb,yi,wj->abij', eri_ao,
                          Cb_vir.conj(),Cb_vir,Cb_occ,Cb_occ.conj()
                          ,optimize = True)
    K_aibj_hyb = numpy.transpose(K_aibj_hyb,(0,2,1,3))
    return K_aibj_hyb

def get_hybrid_exchange_energy_tda_r(mol,C_ao,omega,alpha,hyb):
    # import pdb
    # pdb.set_trace()
    C_vir, C_occ = C_ao
    # spinor -> large component, spsp1 -> small component.
    eri_LL = mol.intor('int2e_spinor')*hyb
    eri_LS = mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2*hyb
    eri_SS = mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4*hyb
    if abs(omega) >= 1e-10:
        with mol.with_range_coulomb(omega=omega):
            eri_LL = eri_LL*alpha/hyb
            eri_LS = eri_LS*alpha/hyb
            eri_SS = eri_SS*alpha/hyb

    n2c = C_occ.shape[0]//2
    
    K_aibj_hyb = numpy.einsum('uvwy,uj,vi,yb,wa->aibj', eri_LL,
                          C_occ[:n2c].conj(),C_occ[:n2c],C_vir[:n2c],C_vir[:n2c].conj()
                          ,optimize = True)
    K_aibj_hyb += numpy.einsum('uvwy,uj,vi,yb,wa->aibj', eri_LS,
                          C_occ[n2c:].conj(),C_occ[n2c:],C_vir[:n2c],C_vir[:n2c].conj()
                          ,optimize = True)
    K_aibj_hyb += numpy.einsum('uvwy,uj,vi,yb,wa->aibj', eri_LS.transpose(2,3,0,1),
                          C_occ[:n2c].conj(),C_occ[:n2c],C_vir[n2c:],C_vir[n2c:].conj()
                          ,optimize = True)
    K_aibj_hyb += numpy.einsum('uvwy,uj,vi,yb,wa->aibj', eri_SS,
                          C_occ[n2c:].conj(),C_occ[n2c:],C_vir[n2c:],C_vir[n2c:].conj()
                          ,optimize = True)
    
    return K_aibj_hyb
  

def get_hartree_potential_tda(mol,C_ao):
    # Columb Interaction, which isn't zero in spin-conserved tda case only.
    # eri in ao space
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = C_ao
    eri_ao = mol.intor('int2e').astype(numpy.complex128)
    # transform the eri to mo space.
    K_aibj_hrp = numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Ca_vir,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Ca_vir.conj(),Ca_occ,Cb_vir,Cb_occ.conj()
                          ,optimize = True)
    K_aibj_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Ca_vir,Ca_occ.conj()
                          ,optimize = True)
    K_aibj_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_ao,
                          Cb_vir.conj(),Cb_occ,Cb_vir,Cb_occ.conj()
                          ,optimize = True)
    
    return K_aibj_hrp

def get_hartree_potential_tda_r(mol, C):
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
    # C_vir[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    # C_occ[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    
    K_aibj_hrp = numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_LL,
                          C_vir[:n2c].conj(),C_occ[:n2c],C_vir[:n2c],C_occ[:n2c].conj()
                          ,optimize = True)
    K_aibj_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_LS,
                          C_vir[:n2c].conj(),C_occ[:n2c],C_vir[n2c:],C_occ[n2c:].conj()
                          ,optimize = True)
    K_aibj_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_LS.transpose(2,3,0,1),
                          C_vir[n2c:].conj(),C_occ[n2c:],C_vir[:n2c],C_occ[:n2c].conj()
                          ,optimize = True)
    K_aibj_hrp += numpy.einsum('uvwy,ua,vi,yb,wj->aibj', eri_SS,
                          C_vir[n2c:].conj(),C_occ[n2c:],C_vir[n2c:],C_occ[n2c:].conj()
                          ,optimize = True)
    
    return K_aibj_hrp

def get_hartree_potential_tda_r_base_mo(mf):
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
    
    # import pdb
    # pdb.set_trace()
     
    K_aibj_hrp = numpy.einsum('iabj->bjai', eri_mo[:nocc,nocc:,nocc:,:nocc])
    # K_aibj_hrp = numpy.einsum('iabj->aibj', eri_mo[:nocc,nocc:,nocc:,:nocc])
    # K_aibj_hrp = K_aibj_hrp.conj()
    # K_aibj_hrp = K_aibj_hrp.transpose(1,0,3,2)
    return K_aibj_hrp

def get_hybrid_exchange_energy_tda_r_base_mo(mf):
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
    
    # import pdb
    # pdb.set_trace()
     
    K_aibj_hye = numpy.einsum('ijba->bjai', eri_mo[:nocc,:nocc,nocc:,nocc:])
    # K_aibj_hrp = numpy.einsum('iabj->aibj', eri_mo[:nocc,nocc:,nocc:,:nocc])
    # K_aibj_hrp = K_aibj_hrp.conj()
    # K_aibj_hrp = K_aibj_hrp.transpose(1,0,3,2)
    return K_aibj_hye

def get_tdamc_Amat_r(mf,diff,Ndirect=None,Ndirect_lc=None,MSL_factor=None,LIBXCT_factor=None,ncpu=None):
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

    iAmat, mo, C = get_iAmat_and_mo_tda_r(mf,xctype,ao)
    
    # K_aibj_hrp = get_hartree_potential_tda_r(mol,C)
    K_aibj_hrp = get_hartree_potential_tda_r_base_mo(mf)
    # enabling range-separated hybrids
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin)
    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        K_aibj_hyb = 0.0
    else:
        K_aibj_hyb = get_hybrid_exchange_energy_tda_r(mf.mol,C,omega, alpha, hyb)
        # K_aibj_hyb = get_hybrid_exchange_energy_tda_r_base_mo(mf)*hyb
        
    dms = mf.make_rdm1()
    # import pdb
    # pdb.set_trace()
    K_aibj = nitdamc.r_noncollinear_tdamc(nir, mol,  mf.xc, mf.grids, dms, mo, Ndirect=Ndirect,
                                       Ndirect_lc=Ndirect_lc, MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
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
    # Amat = iAmat
    return Amat

def get_tdamc_Amat(mf,mf2,diff,Ndirect=None,Ndirect_lc=None,MSL_factor=None,LIBXCT_factor=None,ncpu=None):
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
        iAmat, C_mo, C_ao = uks_to_gks_iAamt_and_mo_tda(mf,mf2,xctype,ao,diff)
        mf=mf2
    else:
        iAmat, C_mo, C_ao = get_iAmat_and_mo_tda(mf,xctype,ao)
    
    K_aibj_hrp = get_hartree_potential_tda(mol,C_ao)
    
    # enabling range-separated hybrids
    omega, alpha, hyb = nitdamc.rsh_and_hybrid_coeff(mf.xc, spin= mf.mol.spin)
    
    # Hybrid Exchange Energy.
    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        K_aibj_hyb = 0.0
    else:
        K_aibj_hyb = get_hybrid_exchange_energy_tda(mf.mol,C_ao,omega, alpha, hyb)

    dmi = mf.make_rdm1()
    dmaa = dmi[:nao,:nao]
    dmab = dmi[:nao,nao:]
    dmba = dmi[nao:,:nao]
    dmbb = dmi[nao:,nao:]
    
    # import pdb
    # pdb.set_trace()
    K_aibj = nitdamc.nr_noncollinear_tdamc(mol, mf.xc, mf.grids, (dmaa,dmab,dmba,dmbb), C_mo, Ndirect=Ndirect,
                                           Ndirect_lc=Ndirect_lc, MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
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
        e_ai, ais, uvs = uks_to_gks_iAamt_and_mo_tda(mf,mf2,xctype,ao,diff)
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

def eigh_tda(self,Amat,Ndirect,Ndirect_lc):
    E_ex,U = numpy.linalg.eigh(Amat)
    # import pdb
    # pdb.set_trace()
    print('2D_Spin_Space_Sample_Points: '+str(Ndirect))
    print('1D_Spin_Space_Sample_Points: '+str(Ndirect_lc))
    # numpy.save('E_ex',E_ex)
    self.Extd = E_ex *27.21138386
    self.U = U
    # for i in range(E_ex.shape[-1]):
    #     print(f"{E_ex[i]:16.14f}")

def excitation_analysis(self):
    mf = self.scf
    mol = mf.mol
    nocc = len(numpy.where(mf.mo_occ>0)[0])
    ntot = len(mf.mo_occ)//2
    nvir = ntot-nocc

    print('Excitation Analysis:\n')
    for i in range(len(self.Extd)):
        norm = self.U[:,i].conj()*self.U[:,i]
        idx_u = numpy.argmax(norm)
        a_i_mo_idx = (idx_u//nvir,idx_u%nvir)
        ai_pair_i = numpy.argmax(mf.mo_coeff[:ntot,ntot+a_i_mo_idx[0]].conj()*mf.mo_coeff[:ntot,ntot+a_i_mo_idx[0]])
        ai_pair_a = numpy.argmax(mf.mo_coeff[:ntot,ntot+nocc*a_i_mo_idx[1]+a_i_mo_idx[0]].conj()*mf.mo_coeff[:ntot,ntot+nocc*a_i_mo_idx[1]+a_i_mo_idx[0]])
        print(mol.spinor_labels()[ai_pair_i],mol.spinor_labels()[ai_pair_a])
    
def Excited_states_Analysis(nocc,U,mo_coeff,orbital_labels,states_number=(0,1),ana_order=1):      
    ntot = mo_coeff.shape[0]//2
    state_s,state_e = states_number
    nvir = ntot-nocc
    if ana_order == 1:
        for i in range(state_s,state_e):
            norm = U[:,i].conj()*U[:,i]
            idx_u = numpy.argmax(norm)
            a_i_mo_idx = (idx_u//nvir,idx_u%nvir)
            ai_pair_i = numpy.argmax(mo_coeff[:ntot,ntot+a_i_mo_idx[0]].conj()*mo_coeff[:ntot,ntot+a_i_mo_idx[0]])
            ai_pair_a = numpy.argmax(mo_coeff[:ntot,ntot+nocc+a_i_mo_idx[1]].conj()*mo_coeff[:ntot,ntot+nocc+a_i_mo_idx[1]])
            print('State'+ str(i+1) + '   ' + orbital_labels[ai_pair_i],orbital_labels[ai_pair_a])
    elif ana_order == 2:
        for i in range(state_s,state_e):
            norm = U[:,i].conj()*U[:,i]
            idx_u = numpy.argmax(norm)
            idx_mo = numpy.argsort(norm)
            
            print('Excitation orbitals based on mo: \n')
            print('State'+ str(i+1) +'\n ---> main and second mo contribution for ai_pair')
            print(norm[idx_mo[-1]],norm[idx_mo[-2]])
            a_i_mo_idx = (idx_u//nvir,idx_u%nvir)
            norm_i = mo_coeff[:ntot,ntot+a_i_mo_idx[0]].conj()*mo_coeff[:ntot,ntot+a_i_mo_idx[0]]
            norm_a = mo_coeff[:ntot,ntot+nocc+a_i_mo_idx[1]].conj()*mo_coeff[:ntot,ntot+nocc+a_i_mo_idx[1]]
            idx_ao_i = numpy.argsort(norm_i)
            idx_ao_a = numpy.argsort(norm_a)
            
            print('---> main and second ao contribution for occupied orbitals')
            print(norm_i[idx_ao_i[-1]],norm_i[idx_ao_i[-2]])
            print('---> main and second ao contribution for virtual orbitals')
            print(norm_a[idx_ao_a[-1]],norm_a[idx_ao_a[-2]])
            print('\nTransform to ao base:')
            print('i: '+ str((orbital_labels[idx_ao_i[-1]],orbital_labels[idx_ao_i[-2]])))
            print('a: '+ str((orbital_labels[idx_ao_a[-1]],orbital_labels[idx_ao_a[-2]])))
            

class TDAMC_GKS:
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
        self.Amat_f = None
        # The method to whether uses matrix-vector multipiation or not. 
        self.method = method
        self.Extype = 'GKS'
        self.U = None
        
    get_iAmat_and_mo_tda = get_iAmat_and_mo_tda
    get_hartree_potential = get_hartree_potential_tda
    get_hartree_potential_r = get_hartree_potential_tda_r
    get_hybrid_exchange_energy_tda = get_hybrid_exchange_energy_tda
    eigh_tda = eigh_tda
    excited_mag_structure = numint_tdamc.excited_mag_structure
    excitation_analysis = excitation_analysis

    def kernel(self, mf2=None,diff=None,Ndirect=None, Ndirect_lc=None, MSL_factor=None,
               LIBXCT_factor=None,ncpu=None,Extd=None,Amat_f=None,nstates=3, parallel= False,
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
            sol = solver.Solver(self.scf,mf2,Extype, kernel=kernel, nstates=nstates, 
                                   init_guess=init_guess, scheme=scheme, max_cycle=max_cycle, 
                                   conv_tol=conv_tol, cutoff = cutoff,Whkerl=Whkerl,parallel=parallel,ncpu=ncpu)
            self.Extd, self.U = sol.solver(ao,xctype)
        else:
            if Extype == 'GKS':
                Amat_tot = get_tdamc_Amat(self.scf,mf2,diff,Ndirect,Ndirect_lc,MSL_factor,LIBXCT_factor,ncpu)
            
            elif Extype == 'DKS':
                Amat_tot = get_tdamc_Amat_r(self.scf,diff,Ndirect,Ndirect_lc,MSL_factor,LIBXCT_factor,ncpu)
            
            if Amat_f is None:
                self.Amat_f = Amat_tot
            eigh_tda(self,Amat_tot,Ndirect,Ndirect_lc)
        # excitation_analysis(self)
            
        
    if __name__ == '__main__':
        from pyscf import gto,dft
        from pyMC.tdamc import tdamc_gks
        mol = gto.Mole()
        mol.verbose = 6
        mol.output = '/dev/null'
        mol.atom.extend([['He', (0.,0.,0.)], ])
        mol.basis = { 'He': 'cc-pvdz'}
        mol.build()

        mf = dft.GKS(mol)
        mf.xc = 'pbe'
        mf.kernel()
        
        mf_tdamc = tdamc_gks.TDAMC_GKS(mf)
        mf.Ndirect = 266
        mf_tdamc.kernel()
        
    