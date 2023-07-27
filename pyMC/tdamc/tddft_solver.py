#!/usr/bin/env python
'''
Author: Li Hao
Date: 2022-08-04 16:43:46
LastEditTime: 2023-05-26 08:52:39
LastEditors: Li Hao
Description: 
FilePath: /pyMC/tdamc/tddft_solver.py

 A + B = C!
'''

import numpy
from pyscf import lib
from pyMC.tdamc import tddft_mc_uks 
from pyMC.tdamc import tddft_mc_gks 
from pyMC.gksmc import numint_gksmc 
from pyscf import dft
import scipy

def init_guess_naive(Extype,nstates, e_ia,*args):
    """Get the initial guess!
        Note that the number of states will enlarged, if degeneracy occur!

    Args:
        nstates (int): number of states intrested
        e_ia (numpy.array): (nocc,nvir) of Delta_{ia}

    Returns:
        x0 (numpy.array): initial guess vector shape of (nov, nstates)
            nov = nocc*nvir
            x0 will be in shape (nocc, nvir, nstates), if xo.reshape(nocc, nvir, nstates)
         
    """
    # import pdb
    # pdb.set_trace()
    if Extype == 'SPIN_CONSERVED':
        e_ia_aa,e_ia_bb = e_ia
        e_ia_tot = numpy.append(e_ia_aa,e_ia_bb)
        e_ia_max = e_ia_tot.max()
        nov = e_ia_tot.size
        nstates = min(nstates, nov)
        e_ia_tot = e_ia_tot.ravel()
        e_threshold = min(e_ia_max, e_ia_tot[numpy.argsort(e_ia_tot)[nstates-1]])
        # Handle degeneracy, include all degenerated states in initial guess
        e_threshold += 1e-6
        idx = numpy.where(e_ia_tot <= e_threshold)[0]
        x0 = numpy.zeros((nov, idx.size))
        for i, j in enumerate(idx):
            x0[j, i] = 1  # Koopmans' excitations
        # x0 = numpy.concatenate((x0,x0),axis=0)
        # y0 = numpy.zeros_like(x0)
        # z0 = numpy.concatenate((x0,y0),axis=0)
        return x0
    elif 'SPIN_FLIP' in Extype:
        # TODO: put codes above together.
        # b2a -> spin-flip-up; a2b -> spin-flip-down
        e_ia_b2a,e_ia_a2b = e_ia
        e_ia_tot = numpy.append(e_ia_b2a,e_ia_a2b)
        e_ia_max = e_ia_tot.max()
        nov = e_ia_tot.size
        nstates = min(nstates, nov)
        e_ia = e_ia_tot.ravel()
        e_threshold = min(e_ia_max, e_ia[numpy.argsort(e_ia_tot)[nstates-1]])
        # Handle degeneracy, include all degenerated states in initial guess
        e_threshold += 1e-6
        idx = numpy.where(e_ia_tot <= e_threshold)[0]
        x0 = numpy.zeros((nov, idx.size))
        for i, j in enumerate(idx):
            x0[j, i] = 1  # Koopmans' excitations
            
    elif Extype == 'DKS':
        e_ia_max = e_ia.max()
        nov = e_ia.size
        nstates = min(nstates, nov)
        e_ia = e_ia.ravel()
        e_threshold = min(e_ia_max, e_ia[numpy.argsort(e_ia)[nstates-1]])
        # Handle degeneracy, include all degenerated states in initial guess
        e_threshold += 1e-6
        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((nov, idx.size),dtype=numpy.complex128)
        for i, j in enumerate(idx):
            x0[j, i] = 1.0+0.0j # Koopmans' excitations
        y0 = numpy.zeros_like(x0)
        # z0 = numpy.block([[x0,y0.conj()],
        #                   [y0,x0.conj()]])
        z0 = numpy.concatenate((x0,y0),axis=0)
        return z0
    else:
        # For GKS and DKS Object.
        e_ia_max = e_ia.max()
        nov = e_ia.size
        nstates = min(nstates, nov)
        e_ia = e_ia.ravel()
        e_threshold = min(e_ia_max, e_ia[numpy.argsort(e_ia)[nstates-1]])
        # Handle degeneracy, include all degenerated states in initial guess
        e_threshold += 1e-6
        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((nov, idx.size))
        for i, j in enumerate(idx):
            x0[j, i] = 1  # Koopmans' excitations
        # x0 = x0.astype(numpy.complex128)
    return x0
    
def get_e_ia_and_mo(mf,xctype,ao,Extype='SPIN_CONSERVED'):
    if 'SPIN_FLIP' in Extype:
        e_ai,ais,uvs = tddft_mc_uks.get_e_ai_and_mo_tddft(mf,xctype,ao,Extype)
        # From e_ai to e_ia
        e_ai_b2a,e_ai_a2b = e_ai
        nvir_a = ais[0].shape[-1]
        nvir_b = ais[2].shape[-1]
        e_ia_b2a = e_ai_b2a.reshape(nvir_a,-1).T.ravel()
        e_ia_a2b = e_ai_a2b.reshape(nvir_b,-1).T.ravel()
        e_ia = (e_ia_b2a,e_ia_a2b)
    elif Extype == 'SPIN_CONSERVED':
        # import pdb
        # pdb.set_trace()
        e_ai,ais,uvs = tddft_mc_uks.get_e_ai_and_mo_tddft(mf,xctype,ao,Extype)
        e_ai_aa,e_ai_bb = e_ai
        nvir_a = ais[0].shape[-1]
        nvir_b = ais[2].shape[-1]
        e_ia_aa = e_ai_aa.reshape(nvir_a,-1).T.ravel()
        e_ia_bb = e_ai_bb.reshape(nvir_b,-1).T.ravel()
        e_ia = (e_ia_aa,e_ia_bb)
    return e_ia,ais,uvs
    
def get_nc_e_ia_and_mo(mf,mf2,xctype,ao,diff,Extype='GKS'):
    e_ai,ais,uvs = tddft_mc_gks.get_e_ai_and_mo_tddft(mf,xctype,ao)
    # From e_ai to e_ia
    nvir = ais[0].shape[-1]
    e_ia = e_ai.reshape(nvir,-1).T.ravel()
    return e_ia,ais,uvs

def get_nc_utg_e_ia_and_mo(mf1,mf2,xctype,ao,diff,Extype='GKS'):
    e_ai,ais,uvs = tddft_mc_gks.uks_to_gks_e_ai_and_mo_tddft(mf1,mf2,xctype,ao,diff)
    # From e_ai to e_ia
    nvir = ais[0].shape[-1]
    e_ia = e_ai.reshape(nvir,-1).T.ravel()
    return e_ia,ais,uvs

def get_nc_r_e_ia_and_mo(mf,xctype,ao,Extype='DKS'):
    e_ai,ais,uvs = tddft_mc_gks.get_e_ai_and_mo_tddft_r(mf,xctype,ao)
    # From e_ai to e_ia
    nvir = uvs[0].shape[-1]
    e_ia = e_ai.reshape(nvir,-1).T.ravel()
    return e_ia,ais,uvs
    
def get_Diagelemt_of_O_sc_simply(e_ia):
    # (A+B)(A-B)= AA+BA-AB-BB -> [[e_ia_aa^2,0],[0,[e_ia_aa^2]] + 0 - 0 -0
    e_ia_aa,e_ia_bb = e_ia
    D_ia_A_aa = e_ia_aa*e_ia_aa
    D_ia_A_bb = e_ia_bb*e_ia_bb
    D_ia = numpy.concatenate((D_ia_A_aa,D_ia_A_bb))
    return D_ia

def get_Diagelemt_of_O_sf_simply(e_ia):
    # (A+B)(A-B)= AA+BA-AB-BB -> [[e_ia_aa^2,0],[0,[e_ia_aa^2]] + 0 - 0 -0
    e_ia_b2a,e_ia_a2b = e_ia
    D_ia_A_b2a = e_ia_b2a*e_ia_b2a
    D_ia_A_a2b = e_ia_a2b*e_ia_a2b
    D_ia = numpy.concatenate((D_ia_A_b2a ,D_ia_A_a2b))
    return D_ia

def get_Diagelemt_of_O_nc_simply(e_ia):
    D_ia = e_ia*e_ia
    return D_ia

def get_Diagelemt_of_O_nc_simply_r(e_ia):
    # [[ A   B ],
    #  [-B*,-A*]] for relativistic case.
    D_ia = numpy.concatenate((e_ia,-e_ia),axis=0)
    # D_ia = e_ia
    return D_ia

def A_ia_spin_flip_AmB(xctype,x0,ais,kernel,weights):
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = ais
     
    nstates = x0.shape[-1]
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    ndim_vb2a = nvir_a*nocc_b
    x0_b2a = x0[:ndim_vb2a].conj()
    x0_a2b = x0[ndim_vb2a:].conj()
    
    if xctype == 'LDA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_a_occ,optimize=True)
        # eri_ao has hybrid factor produced!
        
        s_s = fxc
        # The pseudo-density is calculated!
        rho1_b2a = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(),x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(),x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
        # *2 for xx,yy parts. 
        A_ia_b2a = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a, optimize=True)*2.0
        A_ia_a2b = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b, optimize=True)*2.0
        
        # Pay attention this point
        B_ia_a2b = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a.conj(), optimize=True)*2.0
        B_ia_b2a = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b.conj(), optimize=True)*2.0
        
    elif xctype == 'GGA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        # \nabla 
        nabla_ai_b2a = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        nabla_ai_a2b = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        # eri_ao has hybrid factor produced!
        fxc, hyec = fxc
        omega, alpha,hyb = hyec
        s_s, s_Ns, Ns_Ns = fxc
        ngrid = s_s.shape[-1]

        # The pseudo-density is calculated!
        rho1_b2a = numpy.zeros((4, ngrid, nstates))
        rho1_a2b = numpy.zeros((4, ngrid, nstates))
        rho1_b2a[0] = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b[0] = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
        A_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_b2a[1:], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_b2a[1:], optimize=True)
        
        B_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_b2a[1:].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_b2a[1:].conj(), optimize=True)
        
        A_ia_a2b  =     numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b +=   numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_a2b[1:], optimize=True)
        A_ia_a2b +=   numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_a2b[1:], optimize=True)
        
        B_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_a2b[1:].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_a2b[1:].conj(), optimize=True)
        
        # *2 for xx,yy parts. 
        A_ia_b2a *= 2.0
        A_ia_a2b *= 2.0
        B_ia_b2a *= 2.0
        B_ia_a2b *= 2.0

    elif xctype == 'MGGA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        nabla_ai_b2a = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        nabla_ai_a2b = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        tau_ai_b2a = 0.5*numpy.einsum('xpa,xpi->pai',mo_a_vir[1:4].conj(),mo_b_occ[1:4],optimize=True)   
        tau_ai_a2b = 0.5*numpy.einsum('xpa,xpi->pai',mo_b_vir[1:4].conj(),mo_a_occ[1:4],optimize=True) 
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s, s_Ns, Ns_Ns, u_u, s_u, Ns_u = fxc
        ngrid = s_s.shape[-1]
    
        # The pseudo-density is calculated!
        rho1_b2a = numpy.zeros((5, ngrid, nstates))
        rho1_a2b = numpy.zeros((5, ngrid, nstates))
        
        rho1_b2a[0] = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[4] = numpy.einsum('pbj,jbn->pn', tau_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b[0] = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[4] = numpy.einsum('pbj,jbn->pn', tau_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)

        A_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_b2a[1:4], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_b2a[1:4], optimize=True)
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, u_u*weights, rho1_b2a[4], optimize=True) #u_u
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', ai_b2a, s_u*weights, rho1_b2a[4], optimize=True) #s_u
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, s_u*weights, rho1_b2a[0], optimize=True) #u_s
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', tau_ai_b2a, Ns_u*weights, rho1_b2a[1:4], optimize=True) #u_Ns
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, Ns_u*weights, rho1_b2a[4], optimize=True) #u_Ns
        
        A_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_a2b[1:4], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_a2b[1:4], optimize=True)
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, u_u*weights, rho1_a2b[4], optimize=True) #u_u
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', ai_a2b, s_u*weights, rho1_a2b[4], optimize=True) #s_u
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, s_u*weights, rho1_a2b[0], optimize=True) #u_s
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', tau_ai_a2b, Ns_u*weights, rho1_a2b[1:4], optimize=True) #u_Ns
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, Ns_u*weights, rho1_a2b[4], optimize=True) #u_Ns
        
        B_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_a2b[1:4].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_a2b[1:4].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, u_u*weights, rho1_a2b[4].conj(), optimize=True) #u_u
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', ai_b2a, s_u*weights, rho1_a2b[4].conj(), optimize=True) #s_u
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, s_u*weights, rho1_a2b[0].conj(), optimize=True) #u_s
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', tau_ai_b2a, Ns_u*weights, rho1_a2b[1:4].conj(), optimize=True) #u_Ns
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, Ns_u*weights, rho1_a2b[4].conj(), optimize=True) #u_Ns
        
        B_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_b2a[1:4].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_b2a[1:4].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, u_u*weights, rho1_b2a[4].conj(), optimize=True) #u_u
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', ai_a2b, s_u*weights, rho1_b2a[4].conj(), optimize=True) #s_u
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, s_u*weights, rho1_b2a[0].conj(), optimize=True) #u_s
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', tau_ai_a2b, Ns_u*weights, rho1_b2a[1:4].conj(), optimize=True) #u_Ns
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, Ns_u*weights, rho1_b2a[4].conj(), optimize=True) #u_Ns
        
        A_ia_b2a *= 2.0
        A_ia_a2b *= 2.0
        B_ia_b2a *= 2.0
        B_ia_a2b *= 2.0
        
    return (A_ia_b2a,A_ia_a2b,B_ia_b2a,B_ia_a2b)

def A_ia_spin_flip_ApB(xctype,x0,ais,fxc,weights):
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = ais
     
    nstates = x0.shape[-1]
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    ndim_vb2a = nvir_a*nocc_b
    x0_b2a = x0[:ndim_vb2a]
    x0_a2b = x0[ndim_vb2a:]
    
    if xctype == 'LDA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_a_occ,optimize=True)

        s_s = fxc
        # The pseudo-density is calculated!
        rho1_b2a = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(),x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(),x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
      
        # *2 for xx,yy parts. 
        A_ia_b2a = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a, optimize=True)*2.0
        A_ia_a2b = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b, optimize=True)*2.0
        
        B_ia_b2a = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b.conj(), optimize=True)*2.0
        B_ia_a2b = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a.conj(), optimize=True)*2.0
        
    elif xctype == 'GGA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        # \nabla 
        nabla_ai_b2a = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        nabla_ai_a2b = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        
        s_s, s_Ns, Ns_Ns = fxc
        ngrid = s_s.shape[-1]

        # The pseudo-density is calculated!
        rho1_b2a = numpy.zeros((4, ngrid, nstates))
        rho1_a2b = numpy.zeros((4, ngrid, nstates))
        rho1_b2a[0] = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b[0] = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
        A_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_b2a[1:], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_b2a[1:], optimize=True)
        
        B_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_a2b[1:].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_a2b[1:].conj(), optimize=True)
        
        A_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_a2b[1:], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_a2b[1:], optimize=True)
        
        B_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_b2a[1:].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_b2a[1:].conj(), optimize=True)
        
        # *2 for xx,yy parts. 
        A_ia_b2a *= 2.0
        B_ia_b2a *= 2.0
        A_ia_a2b *= 2.0
        B_ia_a2b *= 2.0

    elif xctype == 'MGGA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        nabla_ai_b2a = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        nabla_ai_a2b = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        tau_ai_b2a = 0.5*numpy.einsum('xpa,xpi->pai',mo_a_vir[1:4].conj(),mo_b_occ[1:4],optimize=True)   
        tau_ai_a2b = 0.5*numpy.einsum('xpa,xpi->pai',mo_b_vir[1:4].conj(),mo_a_occ[1:4],optimize=True) 
       
        s_s, s_Ns, Ns_Ns, u_u, s_u, Ns_u = fxc
        ngrid = s_s.shape[-1]
    
        # The pseudo-density is calculated!
        rho1_b2a = numpy.zeros((5, ngrid, nstates))
        rho1_a2b = numpy.zeros((5, ngrid, nstates))
        
        rho1_b2a[0] = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[4] = numpy.einsum('pbj,jbn->pn', tau_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b[0] = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[4] = numpy.einsum('pbj,jbn->pn', tau_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)

        A_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_b2a[1:4], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_b2a[1:4], optimize=True)
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, u_u*weights, rho1_b2a[4], optimize=True) #u_u
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', ai_b2a, s_u*weights, rho1_b2a[4], optimize=True) #s_u
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, s_u*weights, rho1_b2a[0], optimize=True) #u_s
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', tau_ai_b2a, Ns_u*weights, rho1_b2a[1:4], optimize=True) #u_Ns
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, Ns_u*weights, rho1_b2a[4], optimize=True) #u_Ns
        
        A_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_a2b[1:4], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_a2b[1:4], optimize=True)
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, u_u*weights, rho1_a2b[4], optimize=True) #u_u
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', ai_a2b, s_u*weights, rho1_a2b[4], optimize=True) #s_u
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, s_u*weights, rho1_a2b[0], optimize=True) #u_s
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', tau_ai_a2b, Ns_u*weights, rho1_a2b[1:4], optimize=True) #u_Ns
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, Ns_u*weights, rho1_a2b[4], optimize=True) #u_Ns
        
        B_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_a2b[1:4].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_a2b[1:4].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, u_u*weights, rho1_a2b[4].conj(), optimize=True) #u_u
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', ai_b2a, s_u*weights, rho1_a2b[4].conj(), optimize=True) #s_u
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, s_u*weights, rho1_a2b[0].conj(), optimize=True) #u_s
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', tau_ai_b2a, Ns_u*weights, rho1_a2b[1:4].conj(), optimize=True) #u_Ns
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, Ns_u*weights, rho1_a2b[4].conj(), optimize=True) #u_Ns
        
        B_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_b2a[1:4].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_b2a[1:4].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, u_u*weights, rho1_b2a[4].conj(), optimize=True) #u_u
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', ai_a2b, s_u*weights, rho1_b2a[4].conj(), optimize=True) #s_u
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, s_u*weights, rho1_b2a[0].conj(), optimize=True) #u_s
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', tau_ai_a2b, Ns_u*weights, rho1_b2a[1:4].conj(), optimize=True) #u_Ns
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, Ns_u*weights, rho1_b2a[4].conj(), optimize=True) #u_Ns
        
        A_ia_b2a *= 2.0
        A_ia_a2b *= 2.0
        B_ia_b2a *= 2.0
        B_ia_a2b *= 2.0
        
    return (A_ia_b2a,A_ia_a2b,B_ia_b2a,B_ia_a2b)

def spin_flip_ApB_matx_parallel(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,ncpu):
    ngrid = weights.shape[-1]
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = ais
    e_ia_b2a,e_ia_a2b = e_ia
    fxc, hyec = kernel
    omega, alpha,hyb = hyec
    
    nstates = x0.shape[-1]
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    ndim_vb2a = nvir_a*nocc_b
    x0_b2a = x0[:ndim_vb2a]
    x0_a2b = x0[ndim_vb2a:]
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    # import pdb
    # pdb.set_trace()
    
    if xctype == 'LDA':
        for para in NX_list:
            idxi,idxf = para
            para_results.append(pool.apply_async(A_ia_spin_flip_ApB,
                                (xctype, x0, (mo_a_vir[idxi:idxf], mo_a_occ[idxi:idxf], 
                                mo_b_vir[idxi:idxf],mo_b_occ[idxi:idxf]),fxc[idxi:idxf],
                                weights[idxi:idxf])))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA' or xctype == 'MGGA':
        for para in NX_list:
            idxi,idxf = para
            fxc_para = []
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(A_ia_spin_flip_ApB,
                                (xctype, x0, (mo_a_vir[idxi:idxf], mo_a_occ[idxi:idxf], 
                                mo_b_vir[idxi:idxf],mo_b_occ[idxi:idxf]),fxc[idxi:idxf],
                                weights[idxi:idxf])))
            
        pool.close()
        pool.join()
   
    A_ia_b2a = 0.0
    A_ia_a2b = 0.0
    B_ia_b2a = 0.0
    B_ia_a2b = 0.0
    
    for result_para in para_results:
        result = result_para.get()
        A_ia_b2a += result[0]
        A_ia_a2b += result[1]
        B_ia_b2a += result[2]
        B_ia_a2b += result[3]
         
    # The orbital energy difference is added here
    A_ia_b2a += numpy.einsum('ia,ian->ian', e_ia_b2a.reshape(nocc_b,nvir_a), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
    A_ia_a2b += numpy.einsum('ia,ian->ian', e_ia_a2b.reshape(nocc_a,nvir_b), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_b2a.reshape(nocc_b,nvir_a,nstates), Cb_occ.conj(), Ca_vir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Ca_vir.conj(),Cb_occ)
        A_ia_b2a -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_a2b.reshape(nocc_a,nvir_b,nstates), Ca_occ.conj(), Cb_vir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Cb_vir.conj(),Ca_occ)
        A_ia_a2b -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_a2b.reshape(nocc_a,nvir_b,nstates), Ca_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Ca_vir.conj(),Cb_occ)
        B_ia_b2a -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_b2a.reshape(nocc_b,nvir_a,nstates), Cb_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cb_vir.conj(),Ca_occ)
        B_ia_a2b -= erimo

    A_ia_b2a = A_ia_b2a.reshape(-1,nstates)
    A_ia_a2b = A_ia_a2b.reshape(-1,nstates)
    B_ia_b2a = B_ia_b2a.reshape(-1,nstates)
    B_ia_a2b = B_ia_a2b.reshape(-1,nstates)
    
    TD_ia_ApB_T = A_ia_b2a+B_ia_b2a
    TD_ia_ApB_B = B_ia_a2b+A_ia_a2b
    TD_ia_ApB = numpy.concatenate((TD_ia_ApB_T,TD_ia_ApB_B),axis=0)
    return TD_ia_ApB

def spin_flip_AmB_matx_parallel(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,ncpu):
    ngrid = weights.shape[-1]
    fxc, hyec = kernel
    omega, alpha,hyb = hyec
    
    e_ia_b2a,e_ia_a2b = e_ia
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = ais
    nstates = x0.shape[-1]
    
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    ndim_vb2a = nvir_a*nocc_b
    x0_b2a = x0[:ndim_vb2a].conj()
    x0_a2b = x0[ndim_vb2a:].conj()
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    # import pdb
    # pdb.set_trace()
    
    if xctype == 'LDA':
        for para in NX_list:
            idxi,idxf = para
            para_results.append(pool.apply_async(A_ia_spin_flip_AmB,
                                (xctype, x0, (mo_a_vir[idxi:idxf], mo_a_occ[idxi:idxf],
                                mo_b_vir[idxi:idxf], mo_b_occ[idxi:idxf]),(fxc[idxi:idxf],hyec),
                                weights[idxi:idxf])))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA' or xctype == 'MGGA':
        for para in NX_list:
            idxi,idxf = para
            fxc_para = []
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(A_ia_spin_flip_AmB,
                                (xctype,x0,(mo_a_vir[:,idxi:idxf],mo_b_occ[:,idxi:idxf], 
                                mo_b_vir[:,idxi:idxf],mo_b_occ[:,idxi:idxf]), (fxc_para,hyec),
                                weights[idxi:idxf])))
            
        pool.close()
        pool.join()
    

    A_ia_b2a = 0.0
    A_ia_a2b = 0.0
    B_ia_b2a = 0.0
    B_ia_a2b = 0.0
    
    for result_para in para_results:
        result = result_para.get()
        A_ia_b2a += result[0]
        A_ia_a2b += result[1]
        B_ia_b2a += result[2]
        B_ia_a2b += result[3]

    # The orbital energy difference is added here
    A_ia_b2a += numpy.einsum('ia,ian->ian', e_ia_b2a.reshape(nocc_b,nvir_a), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
    A_ia_a2b += numpy.einsum('ia,ian->ian', e_ia_a2b.reshape(nocc_a,nvir_b), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
    
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_b2a.reshape(nocc_b,nvir_a,nstates), Cb_occ.conj(), Ca_vir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Ca_vir.conj(),Cb_occ)
        A_ia_b2a -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_a2b.reshape(nocc_a,nvir_b,nstates), Ca_occ.conj(), Cb_vir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Cb_vir.conj(),Ca_occ)
        A_ia_a2b -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_b2a.reshape(nocc_b,nvir_a,nstates), Cb_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cb_vir.conj(),Ca_occ)
        B_ia_a2b -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_a2b.reshape(nocc_a,nvir_b,nstates), Ca_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Ca_vir.conj(),Cb_occ)
        B_ia_b2a -= erimo

    A_ia_b2a = A_ia_b2a.reshape(-1,nstates)
    A_ia_a2b = A_ia_a2b.reshape(-1,nstates)
    B_ia_b2a = B_ia_b2a.reshape(-1,nstates)
    B_ia_a2b = B_ia_a2b.reshape(-1,nstates)    
    
    TD_ia_AmB_T = A_ia_b2a-B_ia_b2a
    TD_ia_AmB_B = A_ia_a2b-B_ia_a2b
    TD_ia_AmB = numpy.concatenate((TD_ia_AmB_T,TD_ia_AmB_B),axis=0)
   
    return TD_ia_AmB

def spin_flip_ApB_matx(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
    """Get the matrix vector product for spin-flip calculations
        \sum_{bj} K_{aibj}U_{jb}

    Args:
        e_ia (numpy.array): (nocc,nvir) of Delta_{ia}
        kernel (tuple): ((fxc), eri_mo_aibj) # 这里 eri_mo_aibj 顺带可用来计算 hartree potential
        ao (numpy.array): basis
        C (numpy.array): coefficient
        x0 (numpy.array): guess vector
            x0 will be in shape (nocc, nvir, nstates), if xo.reshape(nocc, nvir, nstates)
        xctype (string): xctype
        weights (numpy.array): numerical integration weights
        ais (tuple): tuple of basis or molecule orbital products

    Returns:
        The produced vector
            ! NOTE: the output dimension is (nocc, nvir, nstates)
    """
    # import pdb
    # pdb.set_trace()
    e_ia_b2a,e_ia_a2b = e_ia
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = ais
    nstates = x0.shape[-1]
    
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    ndim_vb2a = nvir_a*nocc_b
    x0_b2a = x0[:ndim_vb2a]
    x0_a2b = x0[ndim_vb2a:]
    
    if xctype == 'LDA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_a_occ,optimize=True)
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s = fxc
        # The pseudo-density is calculated!
        rho1_b2a = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(),x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(),x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
      
        # *2 for xx,yy parts. 
        A_ia_b2a = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a, optimize=True)*2.0
        A_ia_a2b = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b, optimize=True)*2.0
        
        B_ia_b2a = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b.conj(), optimize=True)*2.0
        B_ia_a2b = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a.conj(), optimize=True)*2.0
        
        # The orbital energy difference is added here
        A_ia_b2a += numpy.einsum('ia,ian->ian', e_ia_b2a.reshape(nocc_b,nvir_a), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        A_ia_a2b += numpy.einsum('ia,ian->ian', e_ia_a2b.reshape(nocc_a,nvir_b), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
    elif xctype == 'GGA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        # \nabla 
        nabla_ai_b2a = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        nabla_ai_a2b = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s, s_Ns, Ns_Ns = fxc
        ngrid = s_s.shape[-1]

        # The pseudo-density is calculated!
        rho1_b2a = numpy.zeros((4, ngrid, nstates))
        rho1_a2b = numpy.zeros((4, ngrid, nstates))
        rho1_b2a[0] = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b[0] = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
        A_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_b2a[1:], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_b2a[1:], optimize=True)
        
        B_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_a2b[1:].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_a2b[1:].conj(), optimize=True)
        
        A_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_a2b[1:], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_a2b[1:], optimize=True)
        
        B_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_b2a[1:].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_b2a[1:].conj(), optimize=True)
        
        # *2 for xx,yy parts. 
        A_ia_b2a *= 2.0
        B_ia_b2a *= 2.0
        A_ia_a2b *= 2.0
        B_ia_a2b *= 2.0
        
        # The orbital energy difference is added here
        A_ia_b2a += numpy.einsum('ia,ian->ian', e_ia_b2a.reshape(nocc_b,nvir_a), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        A_ia_a2b += numpy.einsum('ia,ian->ian', e_ia_a2b.reshape(nocc_a,nvir_b), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)

    elif xctype == 'MGGA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        nabla_ai_b2a = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        nabla_ai_a2b = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        tau_ai_b2a = 0.5*numpy.einsum('xpa,xpi->pai',mo_a_vir[1:4].conj(),mo_b_occ[1:4],optimize=True)   
        tau_ai_a2b = 0.5*numpy.einsum('xpa,xpi->pai',mo_b_vir[1:4].conj(),mo_a_occ[1:4],optimize=True) 
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s, s_Ns, Ns_Ns, u_u, s_u, Ns_u = fxc
        ngrid = s_s.shape[-1]
    
        # The pseudo-density is calculated!
        rho1_b2a = numpy.zeros((5, ngrid, nstates))
        rho1_a2b = numpy.zeros((5, ngrid, nstates))
        
        rho1_b2a[0] = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[4] = numpy.einsum('pbj,jbn->pn', tau_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b[0] = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[4] = numpy.einsum('pbj,jbn->pn', tau_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)

        A_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_b2a[1:4], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_b2a[1:4], optimize=True)
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, u_u*weights, rho1_b2a[4], optimize=True) #u_u
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', ai_b2a, s_u*weights, rho1_b2a[4], optimize=True) #s_u
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, s_u*weights, rho1_b2a[0], optimize=True) #u_s
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', tau_ai_b2a, Ns_u*weights, rho1_b2a[1:4], optimize=True) #u_Ns
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, Ns_u*weights, rho1_b2a[4], optimize=True) #u_Ns
        
        A_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_a2b[1:4], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_a2b[1:4], optimize=True)
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, u_u*weights, rho1_a2b[4], optimize=True) #u_u
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', ai_a2b, s_u*weights, rho1_a2b[4], optimize=True) #s_u
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, s_u*weights, rho1_a2b[0], optimize=True) #u_s
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', tau_ai_a2b, Ns_u*weights, rho1_a2b[1:4], optimize=True) #u_Ns
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, Ns_u*weights, rho1_a2b[4], optimize=True) #u_Ns
        
        B_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_a2b[1:4].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_a2b[1:4].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, u_u*weights, rho1_a2b[4].conj(), optimize=True) #u_u
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', ai_b2a, s_u*weights, rho1_a2b[4].conj(), optimize=True) #s_u
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, s_u*weights, rho1_a2b[0].conj(), optimize=True) #u_s
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', tau_ai_b2a, Ns_u*weights, rho1_a2b[1:4].conj(), optimize=True) #u_Ns
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, Ns_u*weights, rho1_a2b[4].conj(), optimize=True) #u_Ns
        
        B_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_b2a[1:4].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_b2a[1:4].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, u_u*weights, rho1_b2a[4].conj(), optimize=True) #u_u
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', ai_a2b, s_u*weights, rho1_b2a[4].conj(), optimize=True) #s_u
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, s_u*weights, rho1_b2a[0].conj(), optimize=True) #u_s
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', tau_ai_a2b, Ns_u*weights, rho1_b2a[1:4].conj(), optimize=True) #u_Ns
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, Ns_u*weights, rho1_b2a[4].conj(), optimize=True) #u_Ns
        
        A_ia_b2a *= 2.0
        A_ia_a2b *= 2.0
        B_ia_b2a *= 2.0
        B_ia_a2b *= 2.0
        
        # The orbital energy difference is added here
        A_ia_b2a += numpy.einsum('ia,ian->ian', e_ia_b2a.reshape(nocc_b,nvir_a), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        A_ia_a2b += numpy.einsum('ia,ian->ian', e_ia_a2b.reshape(nocc_a,nvir_b), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_b2a.reshape(nocc_b,nvir_a,nstates), Cb_occ.conj(), Ca_vir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Ca_vir.conj(),Cb_occ)
        A_ia_b2a -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_a2b.reshape(nocc_a,nvir_b,nstates), Ca_occ.conj(), Cb_vir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Cb_vir.conj(),Ca_occ)
        A_ia_a2b -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_a2b.reshape(nocc_a,nvir_b,nstates), Ca_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Ca_vir.conj(),Cb_occ)
        B_ia_b2a -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_b2a.reshape(nocc_b,nvir_a,nstates), Cb_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cb_vir.conj(),Ca_occ)
        B_ia_a2b -= erimo

    A_ia_b2a = A_ia_b2a.reshape(-1,nstates)
    A_ia_a2b = A_ia_a2b.reshape(-1,nstates)
    B_ia_b2a = B_ia_b2a.reshape(-1,nstates)
    B_ia_a2b = B_ia_a2b.reshape(-1,nstates)
    
    TD_ia_ApB_T = A_ia_b2a+B_ia_b2a
    TD_ia_ApB_B = B_ia_a2b+A_ia_a2b
    TD_ia_ApB = numpy.concatenate((TD_ia_ApB_T,TD_ia_ApB_B),axis=0)
    return TD_ia_ApB

def spin_flip_AmB_matx(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
    """Get the matrix vector product for spin-flip calculations
        \sum_{bj} K_{aibj}U_{jb}

    Args:
        e_ia (numpy.array): (nocc,nvir) of Delta_{ia}
        kernel (tuple): ((fxc), eri_mo_aibj) # 这里 eri_mo_aibj 顺带可用来计算 hartree potential
        ao (numpy.array): basis
        C (numpy.array): coefficient
        x0 (numpy.array): guess vector
            x0 will be in shape (nocc, nvir, nstates), if xo.reshape(nocc, nvir, nstates)
        xctype (string): xctype
        weights (numpy.array): numerical integration weights
        ais (tuple): tuple of basis or molecule orbital products

    Returns:
        The produced vector
            ! NOTE: the output dimension is (nocc, nvir, nstates)
    """
    # import pdb
    # pdb.set_trace()
    e_ia_b2a,e_ia_a2b = e_ia
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = ais
    nstates = x0.shape[-1]
    
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    ndim_vb2a = nvir_a*nocc_b
    x0_b2a = x0[:ndim_vb2a].conj()
    x0_a2b = x0[ndim_vb2a:].conj()
    
    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_a_occ,optimize=True)
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s = fxc
        # The pseudo-density is calculated!
        rho1_b2a = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(),x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(),x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
        # *2 for xx,yy parts. 
        A_ia_b2a = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a, optimize=True)*2.0
        A_ia_a2b = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b, optimize=True)*2.0
        
        # Pay attention this point
        B_ia_a2b = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a.conj(), optimize=True)*2.0
        B_ia_b2a = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b.conj(), optimize=True)*2.0
        
        # The orbital energy difference is added here
        A_ia_b2a += numpy.einsum('ia,ian->ian', e_ia_b2a.reshape(nocc_b,nvir_a), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        A_ia_a2b += numpy.einsum('ia,ian->ian', e_ia_a2b.reshape(nocc_a,nvir_b), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
    elif xctype == 'GGA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        # \nabla 
        nabla_ai_b2a = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        nabla_ai_a2b = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s, s_Ns, Ns_Ns = fxc
        ngrid = s_s.shape[-1]

        # The pseudo-density is calculated!
        rho1_b2a = numpy.zeros((4, ngrid, nstates))
        rho1_a2b = numpy.zeros((4, ngrid, nstates))
        rho1_b2a[0] = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b[0] = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
        A_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_b2a[1:], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_b2a[1:], optimize=True)
        
        B_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_b2a[1:].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_b2a[1:].conj(), optimize=True)
        
        A_ia_a2b  =     numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b +=   numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_a2b[1:], optimize=True)
        A_ia_a2b +=   numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_a2b[1:], optimize=True)
        
        B_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_a2b[1:].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_a2b[1:].conj(), optimize=True)
        
        # *2 for xx,yy parts. 
        A_ia_b2a *= 2.0
        A_ia_a2b *= 2.0
        B_ia_b2a *= 2.0
        B_ia_a2b *= 2.0
        
        # The orbital energy difference is added here
        A_ia_b2a += numpy.einsum('ia,ian->ian', e_ia_b2a.reshape(nocc_b,nvir_a), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        A_ia_a2b += numpy.einsum('ia,ian->ian', e_ia_a2b.reshape(nocc_a,nvir_b), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)

    elif xctype == 'MGGA':
        ai_b2a = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_a2b = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        nabla_ai_b2a = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        nabla_ai_a2b = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        tau_ai_b2a = 0.5*numpy.einsum('xpa,xpi->pai',mo_a_vir[1:4].conj(),mo_b_occ[1:4],optimize=True)   
        tau_ai_a2b = 0.5*numpy.einsum('xpa,xpi->pai',mo_b_vir[1:4].conj(),mo_a_occ[1:4],optimize=True) 
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s, s_Ns, Ns_Ns, u_u, s_u, Ns_u = fxc
        ngrid = s_s.shape[-1]
    
        # The pseudo-density is calculated!
        rho1_b2a = numpy.zeros((5, ngrid, nstates))
        rho1_a2b = numpy.zeros((5, ngrid, nstates))
        
        rho1_b2a[0] = numpy.einsum('pbj,jbn->pn', ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_b2a[4] = numpy.einsum('pbj,jbn->pn', tau_ai_b2a.conj(), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        rho1_a2b[0] = numpy.einsum('pbj,jbn->pn', ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        rho1_a2b[4] = numpy.einsum('pbj,jbn->pn', tau_ai_a2b.conj(), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)

        A_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_b2a[0], optimize=True)
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_b2a[1:4], optimize=True)
        A_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_b2a[1:4], optimize=True)
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, u_u*weights, rho1_b2a[4], optimize=True) #u_u
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', ai_b2a, s_u*weights, rho1_b2a[4], optimize=True) #s_u
        A_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, s_u*weights, rho1_b2a[0], optimize=True) #u_s
        A_ia_b2a += numpy.einsum('pai,xp,xpn->ian', tau_ai_b2a, Ns_u*weights, rho1_b2a[1:4], optimize=True) #u_Ns
        A_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, Ns_u*weights, rho1_b2a[4], optimize=True) #u_Ns
        
        A_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_a2b[0], optimize=True)
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_a2b[1:4], optimize=True)
        A_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_a2b[1:4], optimize=True)
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, u_u*weights, rho1_a2b[4], optimize=True) #u_u
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', ai_a2b, s_u*weights, rho1_a2b[4], optimize=True) #s_u
        A_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, s_u*weights, rho1_a2b[0], optimize=True) #u_s
        A_ia_a2b += numpy.einsum('pai,xp,xpn->ian', tau_ai_a2b, Ns_u*weights, rho1_a2b[1:4], optimize=True) #u_Ns
        A_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, Ns_u*weights, rho1_a2b[4], optimize=True) #u_Ns
        
        B_ia_b2a  = numpy.einsum('pai,p,pn->ian', ai_b2a, s_s*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, s_Ns*weights, rho1_a2b[0].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', ai_b2a, s_Ns*weights, rho1_a2b[1:4].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_b2a, Ns_Ns*weights, rho1_a2b[1:4].conj(), optimize=True)
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, u_u*weights, rho1_a2b[4].conj(), optimize=True) #u_u
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', ai_b2a, s_u*weights, rho1_a2b[4].conj(), optimize=True) #s_u
        B_ia_b2a += numpy.einsum('pai,p,pn->ian', tau_ai_b2a, s_u*weights, rho1_a2b[0].conj(), optimize=True) #u_s
        B_ia_b2a += numpy.einsum('pai,xp,xpn->ian', tau_ai_b2a, Ns_u*weights, rho1_a2b[1:4].conj(), optimize=True) #u_Ns
        B_ia_b2a += numpy.einsum('xpai,xp,pn->ian', nabla_ai_b2a, Ns_u*weights, rho1_a2b[4].conj(), optimize=True) #u_Ns
        
        B_ia_a2b  = numpy.einsum('pai,p,pn->ian', ai_a2b, s_s*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, s_Ns*weights, rho1_b2a[0].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', ai_a2b, s_Ns*weights, rho1_b2a[1:4].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('xpai,xyp,ypn->ian', nabla_ai_a2b, Ns_Ns*weights, rho1_b2a[1:4].conj(), optimize=True)
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, u_u*weights, rho1_b2a[4].conj(), optimize=True) #u_u
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', ai_a2b, s_u*weights, rho1_b2a[4].conj(), optimize=True) #s_u
        B_ia_a2b += numpy.einsum('pai,p,pn->ian', tau_ai_a2b, s_u*weights, rho1_b2a[0].conj(), optimize=True) #u_s
        B_ia_a2b += numpy.einsum('pai,xp,xpn->ian', tau_ai_a2b, Ns_u*weights, rho1_b2a[1:4].conj(), optimize=True) #u_Ns
        B_ia_a2b += numpy.einsum('xpai,xp,pn->ian', nabla_ai_a2b, Ns_u*weights, rho1_b2a[4].conj(), optimize=True) #u_Ns
        
        A_ia_b2a *= 2.0
        A_ia_a2b *= 2.0
        B_ia_b2a *= 2.0
        B_ia_a2b *= 2.0
        
        # The orbital energy difference is added here
        A_ia_b2a += numpy.einsum('ia,ian->ian', e_ia_b2a.reshape(nocc_b,nvir_a), x0_b2a.reshape(nocc_b,nvir_a,nstates), optimize=True)
        A_ia_a2b += numpy.einsum('ia,ian->ian', e_ia_a2b.reshape(nocc_a,nvir_b), x0_a2b.reshape(nocc_a,nvir_b,nstates), optimize=True)
        
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_b2a.reshape(nocc_b,nvir_a,nstates), Cb_occ.conj(), Ca_vir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Ca_vir.conj(),Cb_occ)
        A_ia_b2a -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_a2b.reshape(nocc_a,nvir_b,nstates), Ca_occ.conj(), Cb_vir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Cb_vir.conj(),Ca_occ)
        A_ia_a2b -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_b2a.reshape(nocc_b,nvir_a,nstates), Cb_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cb_vir.conj(),Ca_occ)
        B_ia_a2b -= erimo
        
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_a2b.reshape(nocc_a,nvir_b,nstates), Ca_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Ca_vir.conj(),Cb_occ)
        B_ia_b2a -= erimo

    A_ia_b2a = A_ia_b2a.reshape(-1,nstates)
    A_ia_a2b = A_ia_a2b.reshape(-1,nstates)
    B_ia_b2a = B_ia_b2a.reshape(-1,nstates)
    B_ia_a2b = B_ia_a2b.reshape(-1,nstates)    
    
    TD_ia_AmB_T = A_ia_b2a-B_ia_b2a
    TD_ia_AmB_B = A_ia_a2b-B_ia_a2b
    TD_ia_AmB = numpy.concatenate((TD_ia_AmB_T,TD_ia_AmB_B),axis=0)
   
    return TD_ia_AmB

def A_ia_spin_conserving_AmB(xctype,z0,mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ,fxc,weights):
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    nov_a = nocc_a*nvir_a
    nstates = z0.shape[-1]
    x0_aa = z0[:nov_a].conj()
    x0_bb = z0[nov_a:].conj()
    
    A_ia_aaaa = 0.0
    A_ia_aabb = 0.0
    A_ia_bbaa = 0.0
    A_ia_bbbb = 0.0
    
    B_ia_aaaa = 0.0
    B_ia_aabb = 0.0
    B_ia_bbaa = 0.0
    B_ia_bbbb = 0.0
    
    if xctype == 'LDA':
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
     
        n_n,n_s,s_s = fxc
   
        # pseduo density
        rho1_aa = numpy.einsum('pai,ian->pn', ai_aa,x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pai,ian->pn', ai_bb,x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # n_n
        A_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa.conj(),rho1_aa,optimize=True)  
        A_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa.conj(),rho1_bb,optimize=True)
        A_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb.conj(),rho1_aa,optimize=True)
        A_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb.conj(),rho1_bb,optimize=True)
        
        # n_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_aa,optimize=True)  
        A_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_bb,optimize=True)
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_aa,optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_bb,optimize=True)
        
        # s_n
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_aa,optimize=True)  
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_bb,optimize=True)
        A_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_aa,optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_bb,optimize=True)
        
        # s_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa.conj(),rho1_aa,optimize=True)  
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa.conj(),rho1_bb,optimize=True)
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb.conj(),rho1_aa,optimize=True)
        A_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb.conj(),rho1_bb,optimize=True)
        
        # pseduo density
        rho1_aa = numpy.einsum('pbj,jbn->pn', ai_aa,x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pbj,jbn->pn', ai_bb,x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # n_n
        B_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb,optimize=True)
        
        # n_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_n
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb,optimize=True)
        
    elif xctype == 'GGA':
        # import pdb
        # pdb.set_trace()
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        # \nabla 
        nabla_ai_aa = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        nabla_ai_bb = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)    
        
        n_n,n_s,n_Nn,n_Ns,s_s,s_Nn,s_Ns,Nn_Nn,Nn_Ns,Ns_Ns = fxc

        ngrid = s_s.shape[-1]
        nstates = z0.shape[-1]
        
        # The pseudo-density is calculated!
        rho1_aa = numpy.zeros((4, ngrid, nstates))
        rho1_bb = numpy.zeros((4, ngrid, nstates))
        rho1_aa[0] = numpy.einsum('pbj,jbn->pn', ai_aa, x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_aa[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_aa, x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_bb[0] = numpy.einsum('pbj,jbn->pn', ai_bb, x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        rho1_bb[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_bb, x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        
        # n_n
        A_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa.conj(),rho1_aa[0],optimize=True)  
        A_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # n_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_aa[0],optimize=True)  
        A_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # s_n
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_aa[0],optimize=True)  
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # s_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa.conj(),rho1_aa[0],optimize=True)  
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # n_Nn
        A_ia_aaaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Nn_n
        A_ia_aaaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa.conj(),rho1_aa[0],optimize=True)
        A_ia_bbaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # n_Ns
        A_ia_aaaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Ns_n
        A_ia_aaaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa.conj(),rho1_aa[0],optimize=True)
        A_ia_bbaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb.conj(),rho1_bb[0],optimize=True)

        # s_Nn
        A_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Nn_s
        A_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa.conj(),rho1_aa[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb.conj(),rho1_bb[0],optimize=True)

        # s_Ns
        A_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Ns_s
        A_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa.conj(),rho1_aa[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # Nn_Nn part
        A_ia_aaaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb.conj(),rho1_bb[1:4],optimize=True)

        # Nn_Ns part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Ns_Nn part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb.conj(),rho1_bb[1:4],optimize=True)

        # Ns_Ns part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Calculate B Part.
        # n_n
        B_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa[0],optimize=True)  
        B_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # n_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        B_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # s_n
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # s_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # n_Nn
        B_ia_aaaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_n
        B_ia_aaaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        B_ia_bbaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # n_Ns
        B_ia_aaaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_n
        B_ia_aaaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        B_ia_bbaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Nn
        B_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_s
        B_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Ns
        B_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_s
        B_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # Nn_Nn part
        B_ia_aaaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Nn_Ns part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_Nn part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Ns_Ns part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        
    return ((A_ia_aaaa,A_ia_aabb,A_ia_bbaa,A_ia_bbbb),
            (B_ia_aaaa,B_ia_aabb,B_ia_bbaa,B_ia_bbbb))

def A_ia_spin_conserving_ApB(xctype,z0,mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ,fxc,weights):
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    nov_a = nocc_a*nvir_a
    nstates = z0.shape[-1]
    x0_aa = z0[:nov_a]
    x0_bb = z0[nov_a:]
    
    A_ia_aaaa = 0.0
    A_ia_aabb = 0.0
    A_ia_bbaa = 0.0
    A_ia_bbbb = 0.0
    
    B_ia_aaaa = 0.0
    B_ia_aabb = 0.0
    B_ia_bbaa = 0.0
    B_ia_bbbb = 0.0
    
    if xctype == 'LDA':
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        n_n,n_s,s_s = fxc
        # pseduo density
        rho1_aa = numpy.einsum('pbj,jbn->pn', ai_aa.conj(),x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pbj,jbn->pn', ai_bb.conj(),x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # n_n
        A_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa,optimize=True)  
        A_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb,optimize=True)
        A_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa,optimize=True)
        A_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb,optimize=True)
        
        # n_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        A_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_n
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        A_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa,optimize=True)  
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb,optimize=True)
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa,optimize=True)
        A_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # pseduo density
        rho1_aa = numpy.einsum('pbj,jbn->pn', ai_aa,x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pbj,jbn->pn', ai_bb,x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # n_n
        B_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb,optimize=True)
        
        # n_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_n
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb,optimize=True)
        
    elif xctype == 'GGA':
        # import pdb
        # pdb.set_trace()
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        # \nabla 
        nabla_ai_aa = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        nabla_ai_bb = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)    
        
        # eri_ao has hybrid factor produced!
        n_n,n_s,n_Nn,n_Ns,s_s,s_Nn,s_Ns,Nn_Nn,Nn_Ns,Ns_Ns = fxc
        
        ngrid = s_s.shape[-1]
        nstates = z0.shape[-1]
        
        # The pseudo-density is calculated!
        rho1_aa = numpy.zeros((4, ngrid, nstates))
        rho1_bb = numpy.zeros((4, ngrid, nstates))
        rho1_aa[0] = numpy.einsum('pbj,jbn->pn', ai_aa.conj(), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_aa[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_aa.conj(), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_bb[0] = numpy.einsum('pbj,jbn->pn', ai_bb.conj(), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        rho1_bb[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_bb.conj(), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        
        # n_n
        A_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # n_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # s_n
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # s_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        
        # n_Nn
        A_ia_aaaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_n
        A_ia_aaaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ia_aabb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # n_Ns
        A_ia_aaaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_n
        A_ia_aaaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ia_aabb      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Nn
        A_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_s
        A_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Ns
        A_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_s
        A_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # Nn_Nn part
        A_ia_aaaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Nn_Ns part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_Nn part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Ns_Ns part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # Calculate B part.
        # n_n
        B_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa[0].conj(),optimize=True)  
        B_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # n_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0].conj(),optimize=True)  
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # s_n
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0].conj(),optimize=True)  
        B_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # s_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa[0].conj(),optimize=True)  
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb[0].conj(),optimize=True)
        
        
        # n_Nn
        B_ia_aaaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Nn_n
        B_ia_aaaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_aa[0].conj(),optimize=True)
        B_ia_aabb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # n_Ns
        B_ia_aaaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Ns_n
        B_ia_aaaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_aa[0].conj(),optimize=True)
        B_ia_aabb      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_bb[0].conj(),optimize=True)

        # s_Nn
        B_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Nn_s
        B_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_aa[0].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_bb[0].conj(),optimize=True)

        # s_Ns
        B_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Ns_s
        B_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_aa[0].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # Nn_Nn part
        B_ia_aaaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_bb[1:4].conj(),optimize=True)

        # Nn_Ns part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Ns_Nn part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4].conj(),optimize=True)

        # Ns_Ns part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_bb[1:4].conj(),optimize=True)
    
    return ((A_ia_aaaa,A_ia_aabb,A_ia_bbaa,A_ia_bbbb),
            (B_ia_aaaa,B_ia_aabb,B_ia_bbaa,B_ia_bbbb))

def spin_conserving_AmB_matx_parallel(e_ia, kernel, z0, xctype, weights, ais, uvs, mf,ncpu):
    e_ia_aa,e_ia_bb = e_ia
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    ngrid = weights.shape[-1]
    
    nstates = z0.shape[-1]
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    nov_aa = nocc_a*nvir_a
    x0_aa = z0[:nov_aa].conj()
    x0_bb = z0[nov_aa:].conj()
    
    fxc, hyec = kernel
    omega, alpha,hyb = hyec
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    if xctype == 'LDA':
        for para in NX_list:
            idxi,idxf = para
            fxc_para = []
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][idxi:idxf])
            para_results.append(pool.apply_async(A_ia_spin_conserving_AmB,(xctype, z0, 
                                mo_a_vir[idxi:idxf], mo_a_occ[idxi:idxf], 
                                mo_b_vir[idxi:idxf], mo_b_occ[idxi:idxf],
                                fxc_para, weights[idxi:idxf])))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA':
        for para in NX_list:
            idxi,idxf = para
            fxc_para = []
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(A_ia_spin_conserving_AmB,(xctype,z0,
                                mo_a_vir[:,idxi:idxf], mo_a_occ[:,idxi:idxf], 
                                mo_b_vir[:,idxi:idxf], mo_b_occ[:,idxi:idxf],
                                fxc_para, weights[idxi:idxf])))
            
        pool.close()
        pool.join()

    elif xctype == 'MGGA':
        raise NotImplementedError("Spin-conserved scheme isn't implemented in Meta-GGA")
    
    A_ia_aaaa = 0.0
    A_ia_aabb = 0.0
    A_ia_bbaa = 0.0
    A_ia_bbbb = 0.0
    
    B_ia_aaaa = 0.0
    B_ia_aabb = 0.0
    B_ia_bbaa = 0.0
    B_ia_bbbb = 0.0
    
    for result_para in para_results:
        result = result_para.get()
        A_ia_aaaa += result[0][0]
        A_ia_aabb += result[0][1]
        A_ia_bbaa += result[0][2]
        A_ia_bbbb += result[0][3]
        
        B_ia_aaaa += result[1][0]
        B_ia_aabb += result[1][1]
        B_ia_bbaa += result[1][2]
        B_ia_bbbb += result[1][3]
    
    # The orbital energy difference is added here
    A_ia_aaaa += numpy.einsum('ia,ian->ian', e_ia_aa.reshape(nocc_a,nvir_a), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
    A_ia_bbbb += numpy.einsum('ia,ian->ian', e_ia_bb.reshape(nocc_b,nvir_b), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
    
    # import pdb
    # pdb.set_trace()
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
    # The hartree potential term.
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir, Ca_occ.conj(), optimize=True)
    A_ia_aaaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir, Ca_occ.conj(), optimize=True)
    A_ia_bbaa += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir, Cb_occ.conj(), optimize=True)
    A_ia_aabb += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir, Cb_occ.conj(), optimize=True)
    A_ia_bbbb += erimo
    
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir, Ca_occ.conj(), optimize=True)
    B_ia_aaaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir, Ca_occ.conj(), optimize=True)
    B_ia_bbaa += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir, Cb_occ.conj(), optimize=True)
    B_ia_aabb += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir, Cb_occ.conj(), optimize=True)
    B_ia_bbbb += erimo
    
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Ca_vir,Ca_occ.conj())
        A_ia_aaaa -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ub,vj->jbn',eri,Cb_vir,Cb_occ.conj())
        A_ia_bbbb -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Ca_vir.conj(),Ca_occ)
        B_ia_aaaa -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,vb,uj->jbn',eri,Cb_vir.conj(),Cb_occ)
        B_ia_bbbb -= erimo

    A_ia_aaaa = A_ia_aaaa.reshape(-1,nstates)
    A_ia_aabb = A_ia_aabb.reshape(-1,nstates)
    A_ia_bbaa = A_ia_bbaa.reshape(-1,nstates)
    A_ia_bbbb = A_ia_bbbb.reshape(-1,nstates)
    
    B_ia_aaaa = B_ia_aaaa.reshape(-1,nstates)
    B_ia_aabb = B_ia_aabb.reshape(-1,nstates)
    B_ia_bbaa = B_ia_bbaa.reshape(-1,nstates)
    B_ia_bbbb = B_ia_bbbb.reshape(-1,nstates)
    
    TD_ia_AmB_L = A_ia_aaaa-B_ia_aaaa+A_ia_bbaa-B_ia_bbaa
    TD_ia_AmB_R = A_ia_aabb-B_ia_aabb+A_ia_bbbb-B_ia_bbbb
    TD_ia_AmB = numpy.concatenate((TD_ia_AmB_L,TD_ia_AmB_R),axis=0)
    return TD_ia_AmB

def spin_conserving_ApB_matx_parallel(e_ia, kernel, z0, xctype, weights, ais, uvs, mf,ncpu):
    e_ia_aa,e_ia_bb = e_ia
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    ngrid = weights.shape[-1]
    
    nstates = z0.shape[-1]
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    nov_aa = nocc_a*nvir_a
    x0_aa = z0[:nov_aa]
    x0_bb = z0[nov_aa:]
    
    fxc, hyec = kernel
    omega, alpha,hyb = hyec
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    if xctype == 'LDA':
        for para in NX_list:
            idxi,idxf = para
            fxc_para = []
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][idxi:idxf])
            para_results.append(pool.apply_async(A_ia_spin_conserving_ApB,(xctype, z0, 
                                mo_a_vir[idxi:idxf], mo_a_occ[idxi:idxf], 
                                mo_b_vir[idxi:idxf], mo_b_occ[idxi:idxf],
                                fxc_para, weights[idxi:idxf])))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA':
        for para in NX_list:
            idxi,idxf = para
            fxc_para = []
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(A_ia_spin_conserving_ApB,(xctype,z0,
                                mo_a_vir[:,idxi:idxf], mo_a_occ[:,idxi:idxf], 
                                mo_b_vir[:,idxi:idxf], mo_b_occ[:,idxi:idxf],
                                fxc_para, weights[idxi:idxf])))
            
        pool.close()
        pool.join()

    elif xctype == 'MGGA':
        raise NotImplementedError("Spin-conserved scheme isn't implemented in Meta-GGA")
    
    A_ia_aaaa = 0.0
    A_ia_aabb = 0.0
    A_ia_bbaa = 0.0
    A_ia_bbbb = 0.0
    
    B_ia_aaaa = 0.0
    B_ia_aabb = 0.0
    B_ia_bbaa = 0.0
    B_ia_bbbb = 0.0
    
    for result_para in para_results:
        result = result_para.get()
        A_ia_aaaa += result[0][0]
        A_ia_aabb += result[0][1]
        A_ia_bbaa += result[0][2]
        A_ia_bbbb += result[0][3]
        
        B_ia_aaaa += result[1][0]
        B_ia_aabb += result[1][1]
        B_ia_bbaa += result[1][2]
        B_ia_bbbb += result[1][3]
    
    # The orbital energy difference is added here
    A_ia_aaaa += numpy.einsum('ia,ian->ian', e_ia_aa.reshape(nocc_a,nvir_a), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
    A_ia_bbbb += numpy.einsum('ia,ian->ian', e_ia_bb.reshape(nocc_b,nvir_b), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
    
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
    
    # The hartree potential term.
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    A_ia_aaaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    A_ia_aabb += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    A_ia_bbaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    A_ia_bbbb += erimo
    
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    B_ia_aaaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    B_ia_aabb += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    B_ia_bbaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    B_ia_bbbb += erimo
    
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Ca_vir.conj(),Ca_occ)
        A_ia_aaaa -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ub,vj->jbn',eri,Cb_vir.conj(),Cb_occ)
        A_ia_bbbb -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Ca_vir.conj(),Ca_occ)
        B_ia_aaaa -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cb_vir.conj(),Cb_occ)
        B_ia_bbbb -= erimo

    A_ia_aaaa = A_ia_aaaa.reshape(-1,nstates)
    A_ia_aabb = A_ia_aabb.reshape(-1,nstates)
    A_ia_bbaa = A_ia_bbaa.reshape(-1,nstates)
    A_ia_bbbb = A_ia_bbbb.reshape(-1,nstates)
    
    B_ia_aaaa = B_ia_aaaa.reshape(-1,nstates)
    B_ia_aabb = B_ia_aabb.reshape(-1,nstates)
    B_ia_bbaa = B_ia_bbaa.reshape(-1,nstates)
    B_ia_bbbb = B_ia_bbbb.reshape(-1,nstates)
        
    TD_ia_ApB_T = A_ia_aaaa+B_ia_aaaa+A_ia_aabb+B_ia_aabb
    TD_ia_ApB_B = A_ia_bbaa+B_ia_bbaa+A_ia_bbbb+B_ia_bbbb
    TD_ia_ApB = numpy.concatenate((TD_ia_ApB_T,TD_ia_ApB_B),axis=0)
    return TD_ia_ApB

def spin_conserving_ApB_matx(e_ia, kernel, z0, xctype, weights, ais, uvs, mf,*args):
    # import pdb
    # pdb.set_trace()
    e_ia_aa,e_ia_bb = e_ia
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    
    nstates = z0.shape[-1]
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    nov_aa = nocc_a*nvir_a
    x0_aa = z0[:nov_aa]
    x0_bb = z0[nov_aa:]
    # y0_aa = z0[nov_aa+nov_bb:nov_aa+nov_bb+nov_aa]
    # y0_bb = z0[nov_aa+nov_bb+nov_aa:]
    
    # (A-B)(A+B) -> z^T (A-B)  and (A+B) z
    # Initial.
    A_ia_aaaa = 0.0
    A_ia_aabb = 0.0
    A_ia_bbaa = 0.0
    A_ia_bbbb = 0.0
    
    B_ia_aaaa = 0.0
    B_ia_aabb = 0.0
    B_ia_bbaa = 0.0
    B_ia_bbbb = 0.0
    
    if xctype == 'LDA':
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
     
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        n_n,n_s,s_s = fxc
        
        # import pdb
        # pdb.set_trace()
        
        # pseduo density
        rho1_aa = numpy.einsum('pbj,jbn->pn', ai_aa.conj(),x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pbj,jbn->pn', ai_bb.conj(),x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # n_n
        A_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa,optimize=True)  
        A_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb,optimize=True)
        A_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa,optimize=True)
        A_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb,optimize=True)
        
        # n_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        A_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_n
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        A_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa,optimize=True)  
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb,optimize=True)
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa,optimize=True)
        A_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # pseduo density
        rho1_aa = numpy.einsum('pbj,jbn->pn', ai_aa,x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pbj,jbn->pn', ai_bb,x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # n_n
        B_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb,optimize=True)
        
        # n_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_n
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # The orbital energy difference is added here
        A_ia_aaaa += numpy.einsum('ia,ian->ian', e_ia_aa.reshape(nocc_a,nvir_a), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        A_ia_bbbb += numpy.einsum('ia,ian->ian', e_ia_bb.reshape(nocc_b,nvir_b), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
    
    elif xctype == 'GGA':
        # import pdb
        # pdb.set_trace()
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        # \nabla 
        nabla_ai_aa = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        nabla_ai_bb = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)    
        
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        n_n,n_s,n_Nn,n_Ns,s_s,s_Nn,s_Ns,Nn_Nn,Nn_Ns,Ns_Ns = fxc
        
        omega, alpha,hyb = hyec
        ngrid = s_s.shape[-1]
        nstates = z0.shape[-1]
        
        # The pseudo-density is calculated!
        rho1_aa = numpy.zeros((4, ngrid, nstates))
        rho1_bb = numpy.zeros((4, ngrid, nstates))
        rho1_aa[0] = numpy.einsum('pbj,jbn->pn', ai_aa.conj(), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_aa[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_aa.conj(), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_bb[0] = numpy.einsum('pbj,jbn->pn', ai_bb.conj(), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        rho1_bb[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_bb.conj(), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        
        # n_n
        A_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # n_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # s_n
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # s_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        
        # n_Nn
        A_ia_aaaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_n
        A_ia_aaaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ia_aabb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # n_Ns
        A_ia_aaaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_n
        A_ia_aaaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ia_aabb      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Nn
        A_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_s
        A_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Ns
        A_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_s
        A_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ia_bbbb    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # Nn_Nn part
        A_ia_aaaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Nn_Ns part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_Nn part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Ns_Ns part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ia_bbbb    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # Calculate B part.
        # n_n
        B_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa[0].conj(),optimize=True)  
        B_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # n_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0].conj(),optimize=True)  
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # s_n
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0].conj(),optimize=True)  
        B_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # s_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa[0].conj(),optimize=True)  
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb[0].conj(),optimize=True)
        
        
        # n_Nn
        B_ia_aaaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Nn_n
        B_ia_aaaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_aa[0].conj(),optimize=True)
        B_ia_aabb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # n_Ns
        B_ia_aaaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Ns_n
        B_ia_aaaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_aa[0].conj(),optimize=True)
        B_ia_aabb      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_bb[0].conj(),optimize=True)

        # s_Nn
        B_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Nn_s
        B_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_aa[0].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_bb[0].conj(),optimize=True)

        # s_Ns
        B_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Ns_s
        B_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_aa[0].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_bb[0].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_aa[0].conj(),optimize=True)
        B_ia_bbbb    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_bb[0].conj(),optimize=True)
        
        # Nn_Nn part
        B_ia_aaaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_bb[1:4].conj(),optimize=True)

        # Nn_Ns part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # Ns_Nn part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4].conj(),optimize=True)

        # Ns_Ns part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_aa[1:4].conj(),optimize=True)
        B_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_bb[1:4].conj(),optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_aa[1:4].conj(),optimize=True)
        B_ia_bbbb    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_bb[1:4].conj(),optimize=True)
        
        # The orbital energy difference is added here
        A_ia_aaaa += numpy.einsum('ia,ian->ian', e_ia_aa.reshape(nocc_a,nvir_a), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        A_ia_bbbb += numpy.einsum('ia,ian->ian', e_ia_bb.reshape(nocc_b,nvir_b), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)

    elif xctype == 'MGGA':
        raise NotImplementedError("Spin-conserved scheme isn't implemented in Meta-GGA")
    
    # import pdb
    # pdb.set_trace()
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
    
    # The hartree potential term.
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    A_ia_aaaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    A_ia_aabb += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    A_ia_bbaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    A_ia_bbbb += erimo
    
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    B_ia_aaaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    B_ia_aabb += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    B_ia_bbaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    B_ia_bbbb += erimo
    
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Ca_vir.conj(),Ca_occ)
        A_ia_aaaa -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ub,vj->jbn',eri,Cb_vir.conj(),Cb_occ)
        A_ia_bbbb -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Ca_vir.conj(),Ca_occ)
        B_ia_aaaa -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cb_vir.conj(),Cb_occ)
        B_ia_bbbb -= erimo

    A_ia_aaaa = A_ia_aaaa.reshape(-1,nstates)
    A_ia_aabb = A_ia_aabb.reshape(-1,nstates)
    A_ia_bbaa = A_ia_bbaa.reshape(-1,nstates)
    A_ia_bbbb = A_ia_bbbb.reshape(-1,nstates)
    
    B_ia_aaaa = B_ia_aaaa.reshape(-1,nstates)
    B_ia_aabb = B_ia_aabb.reshape(-1,nstates)
    B_ia_bbaa = B_ia_bbaa.reshape(-1,nstates)
    B_ia_bbbb = B_ia_bbbb.reshape(-1,nstates)
    
    TD_ia_ApB_T = A_ia_aaaa+B_ia_aaaa+A_ia_aabb+B_ia_aabb
    TD_ia_ApB_B = A_ia_bbaa+B_ia_bbaa+A_ia_bbbb+B_ia_bbbb
    TD_ia_ApB = numpy.concatenate((TD_ia_ApB_T,TD_ia_ApB_B),axis=0)
    return TD_ia_ApB

def spin_conserving_AmB_matx(e_ia, kernel, z0, xctype, weights, ais, uvs, mf,*args):
    e_ia_aa,e_ia_bb = e_ia
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    
    nstates = z0.shape[-1]
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    nov_aa = nocc_a*nvir_a
    nov_bb = nocc_b*nvir_b
    
    x0_aa = z0[:nov_aa].conj()
    x0_bb = z0[nov_aa:].conj()
    
    # (A+B) (A-B) -> z^T (A+B) and (A-B) z
    # Initial.
    A_ia_aaaa = 0.0
    A_ia_aabb = 0.0
    A_ia_bbaa = 0.0
    A_ia_bbbb = 0.0
    
    B_ia_aaaa = 0.0
    B_ia_aabb = 0.0
    B_ia_bbaa = 0.0
    B_ia_bbbb = 0.0
    
    if xctype == 'LDA':
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
     
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        n_n,n_s,s_s = fxc
        
        # import pdb
        # pdb.set_trace()
        
        # pseduo density
        rho1_aa = numpy.einsum('pai,ian->pn', ai_aa,x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pai,ian->pn', ai_bb,x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # n_n
        A_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa.conj(),rho1_aa,optimize=True)  
        A_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa.conj(),rho1_bb,optimize=True)
        A_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb.conj(),rho1_aa,optimize=True)
        A_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb.conj(),rho1_bb,optimize=True)
        
        # n_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_aa,optimize=True)  
        A_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_bb,optimize=True)
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_aa,optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_bb,optimize=True)
        
        # s_n
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_aa,optimize=True)  
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_bb,optimize=True)
        A_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_aa,optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_bb,optimize=True)
        
        # s_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa.conj(),rho1_aa,optimize=True)  
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa.conj(),rho1_bb,optimize=True)
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb.conj(),rho1_aa,optimize=True)
        A_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb.conj(),rho1_bb,optimize=True)
        
        # pseduo density
        rho1_aa = numpy.einsum('pbj,jbn->pn', ai_aa,x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pbj,jbn->pn', ai_bb,x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # n_n
        B_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb,optimize=True)
        
        # n_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_n
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # s_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa,optimize=True)  
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb,optimize=True)
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa,optimize=True)
        B_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb,optimize=True)
        
        # The orbital energy difference is added here
        A_ia_aaaa += numpy.einsum('ia,ian->ian', e_ia_aa.reshape(nocc_a,nvir_a), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        A_ia_bbbb += numpy.einsum('ia,ian->ian', e_ia_bb.reshape(nocc_b,nvir_b), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
    
    elif xctype == 'GGA':
        # import pdb
        # pdb.set_trace()
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        # \nabla 
        nabla_ai_aa = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        nabla_ai_bb = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)    
        
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        n_n,n_s,n_Nn,n_Ns,s_s,s_Nn,s_Ns,Nn_Nn,Nn_Ns,Ns_Ns = fxc
        
        omega, alpha,hyb = hyec
        ngrid = s_s.shape[-1]
        nstates = z0.shape[-1]
        
        # The pseudo-density is calculated!
        rho1_aa = numpy.zeros((4, ngrid, nstates))
        rho1_bb = numpy.zeros((4, ngrid, nstates))
        rho1_aa[0] = numpy.einsum('pbj,jbn->pn', ai_aa, x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_aa[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_aa, x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_bb[0] = numpy.einsum('pbj,jbn->pn', ai_bb, x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        rho1_bb[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_bb, x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        
        # n_n
        A_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa.conj(),rho1_aa[0],optimize=True)  
        A_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # n_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_aa[0],optimize=True)  
        A_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # s_n
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_aa[0],optimize=True)  
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # s_s
        A_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa.conj(),rho1_aa[0],optimize=True)  
        A_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # n_Nn
        A_ia_aaaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Nn_n
        A_ia_aaaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa.conj(),rho1_aa[0],optimize=True)
        A_ia_bbaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # n_Ns
        A_ia_aaaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Ns_n
        A_ia_aaaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa.conj(),rho1_aa[0],optimize=True)
        A_ia_bbaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb.conj(),rho1_bb[0],optimize=True)

        # s_Nn
        A_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Nn_s
        A_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa.conj(),rho1_aa[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb.conj(),rho1_bb[0],optimize=True)

        # s_Ns
        A_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Ns_s
        A_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa.conj(),rho1_aa[0],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa.conj(),rho1_bb[0],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb.conj(),rho1_aa[0],optimize=True)
        A_ia_bbbb    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb.conj(),rho1_bb[0],optimize=True)
        
        # Nn_Nn part
        A_ia_aaaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb.conj(),rho1_bb[1:4],optimize=True)

        # Nn_Ns part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Ns_Nn part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb.conj(),rho1_bb[1:4],optimize=True)

        # Ns_Ns part
        A_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa.conj(),rho1_bb[1:4],optimize=True)
        A_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb.conj(),rho1_aa[1:4],optimize=True)
        A_ia_bbbb    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb.conj(),rho1_bb[1:4],optimize=True)
        
        # Calculate B Part.
        # n_n
        B_ia_aaaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa[0],optimize=True)  
        B_ia_bbaa += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # n_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        B_ia_bbaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # s_n
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb +=    numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # s_s
        B_ia_aaaa +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        B_ia_bbaa += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb +=    numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # n_Nn
        B_ia_aaaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_n
        B_ia_aaaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        B_ia_bbaa += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # n_Ns
        B_ia_aaaa      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_n
        B_ia_aaaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        B_ia_bbaa      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Nn
        B_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_s
        B_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Ns
        B_ia_aaaa    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_s
        B_ia_aaaa    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        B_ia_bbbb    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # Nn_Nn part
        B_ia_aaaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Nn_Ns part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_Nn part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Ns_Ns part
        B_ia_aaaa    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        B_ia_bbaa += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        B_ia_aabb += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        B_ia_bbbb    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # The orbital energy difference is added here
        A_ia_aaaa += numpy.einsum('ia,ian->ian', e_ia_aa.reshape(nocc_a,nvir_a), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        A_ia_bbbb += numpy.einsum('ia,ian->ian', e_ia_bb.reshape(nocc_b,nvir_b), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)

    elif xctype == 'MGGA':
        raise NotImplementedError("Spin-conserved scheme isn't implemented in Meta-GGA")
    
    # import pdb
    # pdb.set_trace()
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
    # The hartree potential term.
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir, Ca_occ.conj(), optimize=True)
    A_ia_aaaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir, Ca_occ.conj(), optimize=True)
    A_ia_bbaa += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir, Cb_occ.conj(), optimize=True)
    A_ia_aabb += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir, Cb_occ.conj(), optimize=True)
    A_ia_bbbb += erimo
    
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir, Ca_occ.conj(), optimize=True)
    B_ia_aaaa += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir, Ca_occ.conj(), optimize=True)
    B_ia_bbaa += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir, Cb_occ.conj(), optimize=True)
    B_ia_aabb += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir, Cb_occ.conj(), optimize=True)
    B_ia_bbbb += erimo
    
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Ca_vir,Ca_occ.conj())
        A_ia_aaaa -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ub,vj->jbn',eri,Cb_vir,Cb_occ.conj())
        A_ia_bbbb -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ, Ca_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Ca_vir.conj(),Ca_occ)
        B_ia_aaaa -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ, Cb_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,vb,uj->jbn',eri,Cb_vir.conj(),Cb_occ)
        B_ia_bbbb -= erimo

    A_ia_aaaa = A_ia_aaaa.reshape(-1,nstates)
    A_ia_aabb = A_ia_aabb.reshape(-1,nstates)
    A_ia_bbaa = A_ia_bbaa.reshape(-1,nstates)
    A_ia_bbbb = A_ia_bbbb.reshape(-1,nstates)
    
    B_ia_aaaa = B_ia_aaaa.reshape(-1,nstates)
    B_ia_aabb = B_ia_aabb.reshape(-1,nstates)
    B_ia_bbaa = B_ia_bbaa.reshape(-1,nstates)
    B_ia_bbbb = B_ia_bbbb.reshape(-1,nstates)
    
    TD_ia_AmB_L = A_ia_aaaa-B_ia_aaaa+A_ia_bbaa-B_ia_bbaa
    TD_ia_AmB_R = A_ia_aabb-B_ia_aabb+A_ia_bbbb-B_ia_bbbb
    TD_ia_AmB = numpy.concatenate((TD_ia_AmB_L,TD_ia_AmB_R),axis=0)
    return TD_ia_AmB

def A_ia_non_collinear_ApB(xctype,x0,ais,fxc):
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = ais
    nocc = mo_a_occ.shape[-1] 
    nvir = mo_a_vir.shape[-1] 
    nstates = x0.shape[-1]
    
    if xctype == 'LDA':
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_ab = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_ba = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        n_n, n_s, s_s = fxc
        
        # import pdb
        # pdb.set_trace()
        # Note: weights has been multiplied here.
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        A_ia  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        A_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        A_ia += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        A_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
    
        B_ia  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1.conj(), optimize=True).astype(numpy.complex128) # n_n
        B_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1.conj(), optimize=True) # n_s
        B_ia += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1.conj(), optimize=True) # s_n
        B_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1.conj(), optimize=True) # s_s
    
    elif xctype == 'GGA':
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        # construct gradient terms
        ai_na_a = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_a_na = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        ai_nb_a = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_b_na = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_nb_b = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_b_nb = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        
        ai_nrho = ai_na_a + ai_a_na + ai_nb_b + ai_b_nb
        ai_nMx = ai_na_b + ai_a_nb + ai_nb_a + ai_b_na
        ai_nMy = -1.0j*(ai_na_b + ai_a_nb) + 1.0j*(ai_nb_a + ai_b_na)
        ai_nMz = ai_na_a + ai_a_na - ai_nb_b - ai_b_nb
        ai_ns = numpy.array([ai_nMx,ai_nMy,ai_nMz])
        
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        ngrid = Nn_Nntmp.shape[-1]
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_nrho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_ns.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        A_ia = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_Nn, nrho1, optimize=True) # n_Nn
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_nrho, n_Nn, rho1, optimize=True) # Nn_n
        
        A_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho, n_Ns, nM1, optimize=True) # n_Ns
        A_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_ns, n_Ns, rho1, optimize=True) # Ns_n
        
        A_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s, s_Nn, nrho1, optimize=True) # s_Nn
        A_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_nrho, s_Nn, M1, optimize=True) # Nn_s
        
        A_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s, s_Ns, nM1, optimize=True) # s_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_nrho, Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_nrho, Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_ns, Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        B_ia = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1.conj(), optimize=True).astype(numpy.complex128) # n_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1.conj(), optimize=True) # n_s
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1.conj(), optimize=True) # s_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_Nn, nrho1.conj(), optimize=True) # n_Nn
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_nrho, n_Nn, rho1.conj(), optimize=True) # Nn_n
        
        B_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho, n_Ns, nM1.conj(), optimize=True) # n_Ns
        B_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_ns, n_Ns, rho1.conj(), optimize=True) # Ns_n
        
        B_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1.conj(), optimize=True) # s_s
        
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s, s_Nn, nrho1.conj(), optimize=True) # s_Nn
        B_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_nrho, s_Nn, M1.conj(), optimize=True) # Nn_s
        
        B_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s, s_Ns, nM1.conj(), optimize=True) # s_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, s_Ns, M1.conj(), optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_nrho, Nn_Nn, nrho1.conj(), optimize=True) # Nn_Nn
        
        B_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_nrho, Nn_Ns, nM1.conj(), optimize=True) # Nn_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, Nn_Ns, nrho1.conj(), optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        B_ia += numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_ns, Ns_Ns, nM1.conj(), optimize=True) # Ns_Ns
        
    return A_ia, B_ia  

def A_ia_non_collinear_AmB(xctype,x0,ais,fxc):
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = ais
    nocc = mo_a_occ.shape[-1] 
    nvir = mo_a_vir.shape[-1] 
    nstates = x0.shape[-1]
    
    if xctype == 'LDA':
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_ab = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_ba = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        n_n, n_s, s_s = fxc
        
        # import pdb
        # pdb.set_trace()
        # Note: weights has been multiplied here.
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho, x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s, x0.reshape(nocc,nvir,nstates), optimize=True)
        A_ia  = numpy.einsum('pai,p,pn->ian', ai_rho.conj(), n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        A_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho.conj(), n_s, M1, optimize=True) # n_s
        A_ia += numpy.einsum('xpai,xp,pn->ian', ai_s.conj(), n_s, rho1, optimize=True) # s_n
        A_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s.conj(), s_s, M1, optimize=True) # s_s
        
        B_ia  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        B_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        B_ia += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        B_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        
    elif xctype == 'GGA':
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        # construct gradient terms
        ai_na_a = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_a_na = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        ai_nb_a = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_b_na = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_nb_b = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_b_nb = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        
        ai_nrho = ai_na_a + ai_a_na + ai_nb_b + ai_b_nb
        ai_nMx = ai_na_b + ai_a_nb + ai_nb_a + ai_b_na
        ai_nMy = -1.0j*(ai_na_b + ai_a_nb) + 1.0j*(ai_nb_a + ai_b_na)
        ai_nMz = ai_na_a + ai_a_na - ai_nb_b - ai_b_nb
        ai_ns = numpy.array([ai_nMx,ai_nMy,ai_nMz])
        
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        ngrid = Nn_Nntmp.shape[-1]
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho, x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s, x0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_nrho, x0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_ns, x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        A_ia = numpy.einsum('pai,p,pn->ian', ai_rho.conj(), n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho.conj(), n_s, M1, optimize=True) # n_s
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s.conj(), n_s, rho1, optimize=True) # s_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho.conj(), n_Nn, nrho1, optimize=True) # n_Nn
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_nrho.conj(), n_Nn, rho1, optimize=True) # Nn_n
        
        A_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho.conj(), n_Ns, nM1, optimize=True) # n_Ns
        A_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_ns.conj(), n_Ns, rho1, optimize=True) # Ns_n
        
        A_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s.conj(), s_s, M1, optimize=True) # s_s
        
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s.conj(), s_Nn, nrho1, optimize=True) # s_Nn
        A_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_nrho.conj(), s_Nn, M1, optimize=True) # Nn_s
        
        A_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s.conj(), s_Ns, nM1, optimize=True) # s_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns.conj(), s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_nrho.conj(), Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_nrho.conj(), Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns.conj(), Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_ns.conj(), Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        B_ia = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_Nn, nrho1, optimize=True) # n_Nn
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_nrho, n_Nn, rho1, optimize=True) # Nn_n
        
        B_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho, n_Ns, nM1, optimize=True) # n_Ns
        B_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_ns, n_Ns, rho1, optimize=True) # Ns_n
        
        B_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s, s_Nn, nrho1, optimize=True) # s_Nn
        B_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_nrho, s_Nn, M1, optimize=True) # Nn_s
        
        B_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s, s_Ns, nM1, optimize=True) # s_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_nrho, Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        B_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_nrho, Nn_Ns, nM1, optimize=True) # Nn_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        B_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_ns, Ns_Ns, nM1, optimize=True) # Ns_Ns
        
    return A_ia,B_ia

def non_collinear_AmB_matx_parallel(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,ncpu):
    fxc, hyec = kernel
    omega, alpha,hyb = hyec
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    Ca_vir, Ca_occ, Cb_vir, Cb_occ = uvs
    x0 = x0.conj()
    
    nstates = x0.shape[-1]
    nocc = mo_a_occ.shape[-1]
    nvir = mo_a_vir.shape[-1]
    ngrid = weights.shape[-1]
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        for para in NX_list:
            idxi,idxf = para
            ais_para = []
            fxc_para = []
            for i in range(len(ais)):
                ais_para.append(ais[i][idxi:idxf])
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])    
            para_results.append(pool.apply_async(A_ia_non_collinear_AmB,
                                (xctype, x0, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA':
        for para in NX_list:
            idxi,idxf = para
            ais_para = []
            fxc_para = []
            for i in range(len(ais)):
                ais_para.append(ais[i][:,idxi:idxf])
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(A_ia_non_collinear_AmB,
                                (xctype, x0, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    
    A_ia,B_ia = (0.0,0.0)
    for result_para in para_results:
        result = result_para.get()
        A_ia += result[0]
        B_ia += result[1]
        
    # The orbital energy difference is calculated here
    A_ia+= numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
    
    # import pdb
    # pdb.set_trace()
    Cvir = numpy.concatenate((Ca_vir, Cb_vir),axis=0)
    Cocc = numpy.concatenate((Ca_occ, Cb_occ),axis=0)
    # The hartree potential term.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Cvir, Cocc.conj(), optimize=True)
    A_ia += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Cvir.conj(), Cocc, optimize=True)
    B_ia += erimo
    
    # The excat exchange is calculated
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc.conj(), Cvir, optimize=True)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Cvir.conj(),Cocc, optimize=True)
        A_ia -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cvir.conj(),Cocc, optimize=True)
        B_ia -= erimo

    A_ia = A_ia.reshape(-1,nstates)
    B_ia = B_ia.reshape(-1,nstates)
    TD_ia_AmB = A_ia - B_ia
    return TD_ia_AmB

def non_collinear_ApB_matx_parallel(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,ncpu):
    fxc, hyec = kernel
    omega, alpha,hyb = hyec
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    Ca_vir, Ca_occ, Cb_vir, Cb_occ = uvs
    
    nstates = x0.shape[-1]
    nocc = mo_a_occ.shape[-1]
    nvir = mo_a_vir.shape[-1]
    ngrid = weights.shape[-1]
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        for para in NX_list:
            idxi,idxf = para
            ais_para = []
            fxc_para = []
            for i in range(len(ais)):
                ais_para.append(ais[i][idxi:idxf])
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])    
            para_results.append(pool.apply_async(A_ia_non_collinear_ApB,
                                (xctype, x0, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA':
        for para in NX_list:
            idxi,idxf = para
            ais_para = []
            fxc_para = []
            for i in range(len(ais)):
                ais_para.append(ais[i][:,idxi:idxf])
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(A_ia_non_collinear_ApB,
                                (xctype, x0, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    
    A_ia,B_ia = (0.0,0.0)
    for result_para in para_results:
        result = result_para.get()
        A_ia += result[0]
        B_ia += result[1]
        
    # The orbital energy difference is calculated here
    A_ia += numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
    
    # import pdb
    # pdb.set_trace()
    Cvir = numpy.concatenate((Ca_vir, Cb_vir),axis=0)
    Cocc = numpy.concatenate((Ca_occ, Cb_occ),axis=0)
    # The hartree potential term.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc.conj(), Cvir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Cvir.conj(), Cocc, optimize=True)
    A_ia += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Cvir.conj(), Cocc, optimize=True)
    B_ia += erimo
    
    # The excat exchange is calculated
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc.conj(), Cvir, optimize=True)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Cvir.conj(),Cocc, optimize=True)
        A_ia -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cvir.conj(),Cocc, optimize=True)
        B_ia -= erimo

    A_ia = A_ia.reshape(-1,nstates)
    B_ia = B_ia.reshape(-1,nstates)
    TD_ia_ApB = A_ia + B_ia
    return TD_ia_ApB

def non_collinear_ApB_matx(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
    fxc, hyec = kernel
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    Ca_vir, Ca_occ, Cb_vir, Cb_occ = uvs
    nstates = x0.shape[-1]
    nocc = mo_a_occ.shape[-1]
    nvir = mo_a_vir.shape[-1]
    if xctype == 'LDA':
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_ab = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_ba = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        n_n, n_s, s_s = fxc
        
        # import pdb
        # pdb.set_trace()
        # Note: weights has been multiplied here.
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        A_ia  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        A_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        A_ia += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        A_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
    
        B_ia  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1.conj(), optimize=True).astype(numpy.complex128) # n_n
        B_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1.conj(), optimize=True) # n_s
        B_ia += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1.conj(), optimize=True) # s_n
        B_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1.conj(), optimize=True) # s_s
        
        # The orbital energy difference is calculated here
        A_ia += numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    elif xctype == 'GGA':
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        # construct gradient terms
        ai_na_a = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_a_na = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        ai_nb_a = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_b_na = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_nb_b = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_b_nb = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        
        ai_nrho = ai_na_a + ai_a_na + ai_nb_b + ai_b_nb
        ai_nMx = ai_na_b + ai_a_nb + ai_nb_a + ai_b_na
        ai_nMy = -1.0j*(ai_na_b + ai_a_nb) + 1.0j*(ai_nb_a + ai_b_na)
        ai_nMz = ai_na_a + ai_a_na - ai_nb_b - ai_b_nb
        ai_ns = numpy.array([ai_nMx,ai_nMy,ai_nMz])
        
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        ngrid = Nn_Nntmp.shape[-1]
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_nrho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_ns.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        A_ia = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_Nn, nrho1, optimize=True) # n_Nn
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_nrho, n_Nn, rho1, optimize=True) # Nn_n
        
        A_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho, n_Ns, nM1, optimize=True) # n_Ns
        A_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_ns, n_Ns, rho1, optimize=True) # Ns_n
        
        A_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s, s_Nn, nrho1, optimize=True) # s_Nn
        A_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_nrho, s_Nn, M1, optimize=True) # Nn_s
        
        A_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s, s_Ns, nM1, optimize=True) # s_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_nrho, Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_nrho, Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_ns, Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        B_ia = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1.conj(), optimize=True).astype(numpy.complex128) # n_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1.conj(), optimize=True) # n_s
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1.conj(), optimize=True) # s_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_Nn, nrho1.conj(), optimize=True) # n_Nn
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_nrho, n_Nn, rho1.conj(), optimize=True) # Nn_n
        
        B_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho, n_Ns, nM1.conj(), optimize=True) # n_Ns
        B_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_ns, n_Ns, rho1.conj(), optimize=True) # Ns_n
        
        B_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1.conj(), optimize=True) # s_s
        
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s, s_Nn, nrho1.conj(), optimize=True) # s_Nn
        B_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_nrho, s_Nn, M1.conj(), optimize=True) # Nn_s
        
        B_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s, s_Ns, nM1.conj(), optimize=True) # s_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, s_Ns, M1.conj(), optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_nrho, Nn_Nn, nrho1.conj(), optimize=True) # Nn_Nn
        
        B_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_nrho, Nn_Ns, nM1.conj(), optimize=True) # Nn_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, Nn_Ns, nrho1.conj(), optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        B_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_ns, Ns_Ns, nM1.conj(), optimize=True) # Ns_Ns
        
        # The orbital energy difference is calculated here
        A_ia+= numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    
    # import pdb
    # pdb.set_trace()
    Cvir = numpy.concatenate((Ca_vir, Cb_vir),axis=0)
    Cocc = numpy.concatenate((Ca_occ, Cb_occ),axis=0)
    # The hartree potential term.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc.conj(), Cvir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Cvir.conj(), Cocc, optimize=True)
    A_ia += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Cvir.conj(), Cocc, optimize=True)
    B_ia += erimo
    
    # The excat exchange is calculated
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc.conj(), Cvir, optimize=True)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Cvir.conj(),Cocc, optimize=True)
        A_ia -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cvir.conj(),Cocc, optimize=True)
        B_ia -= erimo

    A_ia = A_ia.reshape(-1,nstates)
    B_ia = B_ia.reshape(-1,nstates)
    TD_ia_ApB = A_ia + B_ia
    return TD_ia_ApB

def non_collinear_AmB_matx(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
    fxc, hyec = kernel
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    Ca_vir, Ca_occ, Cb_vir, Cb_occ = uvs
    nstates = x0.shape[-1]
    nocc = mo_a_occ.shape[-1]
    nvir = mo_a_vir.shape[-1]
    x0 = x0.conj()
    if xctype == 'LDA':
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_ab = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_ba = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        n_n, n_s, s_s = fxc
        
        # import pdb
        # pdb.set_trace()
        # Note: weights has been multiplied here.
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho, x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s, x0.reshape(nocc,nvir,nstates), optimize=True)
        A_ia  = numpy.einsum('pai,p,pn->ian', ai_rho.conj(), n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        A_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho.conj(), n_s, M1, optimize=True) # n_s
        A_ia += numpy.einsum('xpai,xp,pn->ian', ai_s.conj(), n_s, rho1, optimize=True) # s_n
        A_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s.conj(), s_s, M1, optimize=True) # s_s
        
        B_ia  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        B_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        B_ia += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        B_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        
        # The orbital energy difference is calculated here
        A_ia += numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    elif xctype == 'GGA':
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        # construct gradient terms
        ai_na_a = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_a_na = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        ai_nb_a = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_b_na = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_nb_b = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_b_nb = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        
        ai_nrho = ai_na_a + ai_a_na + ai_nb_b + ai_b_nb
        ai_nMx = ai_na_b + ai_a_nb + ai_nb_a + ai_b_na
        ai_nMy = -1.0j*(ai_na_b + ai_a_nb) + 1.0j*(ai_nb_a + ai_b_na)
        ai_nMz = ai_na_a + ai_a_na - ai_nb_b - ai_b_nb
        ai_ns = numpy.array([ai_nMx,ai_nMy,ai_nMz])
        
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        ngrid = Nn_Nntmp.shape[-1]
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho, x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s, x0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_nrho, x0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_ns, x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        A_ia = numpy.einsum('pai,p,pn->ian', ai_rho.conj(), n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho.conj(), n_s, M1, optimize=True) # n_s
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s.conj(), n_s, rho1, optimize=True) # s_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho.conj(), n_Nn, nrho1, optimize=True) # n_Nn
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_nrho.conj(), n_Nn, rho1, optimize=True) # Nn_n
        
        A_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho.conj(), n_Ns, nM1, optimize=True) # n_Ns
        A_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_ns.conj(), n_Ns, rho1, optimize=True) # Ns_n
        
        A_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s.conj(), s_s, M1, optimize=True) # s_s
        
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s.conj(), s_Nn, nrho1, optimize=True) # s_Nn
        A_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_nrho.conj(), s_Nn, M1, optimize=True) # Nn_s
        
        A_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s.conj(), s_Ns, nM1, optimize=True) # s_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns.conj(), s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_nrho.conj(), Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_nrho.conj(), Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns.conj(), Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_ns.conj(), Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        B_ia = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_Nn, nrho1, optimize=True) # n_Nn
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_nrho, n_Nn, rho1, optimize=True) # Nn_n
        
        B_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho, n_Ns, nM1, optimize=True) # n_Ns
        B_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_ns, n_Ns, rho1, optimize=True) # Ns_n
        
        B_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s, s_Nn, nrho1, optimize=True) # s_Nn
        B_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_nrho, s_Nn, M1, optimize=True) # Nn_s
        
        B_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s, s_Ns, nM1, optimize=True) # s_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_nrho, Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        B_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_nrho, Nn_Ns, nM1, optimize=True) # Nn_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_ns, Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        B_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_ns, Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        # The orbital energy difference is calculated here
        A_ia+= numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    
    # import pdb
    # pdb.set_trace()
    Cvir = numpy.concatenate((Ca_vir, Cb_vir),axis=0)
    Cocc = numpy.concatenate((Ca_occ, Cb_occ),axis=0)
    # The hartree potential term.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Cvir, Cocc.conj(), optimize=True)
    A_ia += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Cvir.conj(), Cocc, optimize=True)
    B_ia += erimo
    
    # The excat exchange is calculated
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc.conj(), Cvir, optimize=True)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,Cvir.conj(),Cocc, optimize=True)
        A_ia -= erimo
        
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir.conj(), optimize=True)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,va,ui->ian',eri,Cvir.conj(),Cocc, optimize=True)
        B_ia -= erimo

    A_ia = A_ia.reshape(-1,nstates)
    B_ia = B_ia.reshape(-1,nstates)
    TD_ia_AmB = A_ia - B_ia
    return TD_ia_AmB

def A_ia_non_collinear_r(xctype,z0,ais,fxc):
    mo_vir_L,mo_vir_S,mo_occ_L,mo_occ_S = ais
    nstates = z0.shape[-1]
    nocc = mo_occ_L.shape[-1]
    nvir = mo_vir_L.shape[-1]
    
    x0 = z0[:z0.shape[0]//2]
    y0 = z0[z0.shape[0]//2:]
    
    betasigma_x = numpy.array(
                 [[0,1,0,0],
                  [1,0,0,0],
                  [0,0,0,-1],
                  [0,0,-1,0]]
                 )
    betasigma_y = numpy.array(
                 [[0,-1.0j,0,0],
                  [1.0j,0,0,0],
                  [0,0,0,1.0j],
                  [0,0,-1.0j,0]]
                  )
    betasigma_z = numpy.array(
                 [[1,0,0,0],
                  [0,-1,0,0],
                  [0,0,-1,0],
                  [0,0,0,1]]
                  )
        
    if xctype == 'LDA':
        n_n,n_s,s_s = fxc 
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i       
        ai_rho = numpy.einsum('cxpa,cxpi->pai', mo_vir_L.conj(), mo_occ_L, optimize=True)
        ai_rho+= numpy.einsum('cxpa,cxpi->pai', mo_vir_S.conj(), mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        
        ai_Mx = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_x[:2,:2], mo_occ_L, optimize=True)
        ai_Mx+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_x[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_y[:2,:2], mo_occ_L, optimize=True)
        ai_My+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_y[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_z[:2,:2], mo_occ_L, optimize=True)
        ai_Mz+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_z[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 =   numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_top  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True) # n_n
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        TD_ia_top += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), y0.reshape(nocc,nvir,nstates), optimize=True)
        M1   = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), y0.reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_top += numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1.conj(), optimize=True) # n_n B_ia_top Part.
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1.conj(), optimize=True) # n_s B_ia_top Part.
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1.conj(), optimize=True) # s_n B_ia_top Part.
        TD_ia_top += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1.conj(), optimize=True) # s_s B_ia_top Part.
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        M1 =   numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.conj().reshape(nocc,nvir,nstates), optimize=True)

        TD_ia_bom  =-numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1.conj(), optimize=True) # n_n
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1.conj(), optimize=True) # n_s
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1.conj(), optimize=True) # s_n
        TD_ia_bom -= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1.conj(), optimize=True) # s_s
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        M1   = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_bom -= numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True) # n_n B_ia_top Part.
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s B_ia_top Part.
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n B_ia_top Part.
        TD_ia_bom -= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s B_ia_top Part.
        
    elif xctype == 'GGA':
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        
        ai_rho = numpy.einsum('cxpa,cpi->xpai', mo_vir_L.conj(), mo_occ_L[:,0], optimize=True)
        ai_rho+= numpy.einsum('cpa,cxpi->xpai', mo_vir_L[:,0].conj(), mo_occ_L, optimize=True)
        ai_rho+= numpy.einsum('cxpa,cpi->xpai', mo_vir_S.conj(), mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_rho+= numpy.einsum('cpa,cxpi->xpai', mo_vir_S[:,0].conj(), mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_rho[0]*= 0.5
        
        ai_Mx = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_x[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_Mx+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_x[:2,:2], mo_occ_L, optimize=True)
        ai_Mx+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_x[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mx+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_x[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_y[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_My+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_y[:2,:2], mo_occ_L, optimize=True)
        ai_My+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_y[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_y[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_z[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_Mz+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_z[:2,:2], mo_occ_L, optimize=True)
        ai_Mz+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_z[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_z[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        
        ai_Mx[0]*=0.5
        ai_My[0]*=0.5
        ai_Mz[0]*=0.5
        
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        ngrid = Nn_Nntmp.shape[-1]
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        TD_ia_top  = numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        TD_ia_top += numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        TD_ia_top += numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        TD_ia_top += numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        TD_ia_top += numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        TD_ia_top += numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        TD_ia_top += numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        TD_ia_top += numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        TD_ia_top += numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        TD_ia_top += numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        TD_ia_top += numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        TD_ia_top += numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0], y0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0], y0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:], y0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:], y0.reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_top += numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        TD_ia_top +=   numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        TD_ia_top +=   numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        TD_ia_top += numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        TD_ia_top += numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        TD_ia_top += numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        TD_ia_top +=   numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        TD_ia_top += numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        TD_ia_top += numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        TD_ia_top += numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        TD_ia_top +=   numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        TD_ia_top += numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        TD_ia_top += numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0].conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0].conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:].conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:].conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_bom =- numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        TD_ia_bom -= numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        TD_ia_bom -= numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        TD_ia_bom -= numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        TD_ia_bom -= numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        TD_ia_bom -= numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        TD_ia_bom -= numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        TD_ia_bom -= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        TD_ia_bom -= numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        TD_ia_bom -= numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        TD_ia_bom -= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        TD_ia_bom -= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0], x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0], x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:], x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:], x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_bom -= numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        TD_ia_bom -=   numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        TD_ia_bom -= numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        TD_ia_bom -= numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        TD_ia_bom -= numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        TD_ia_bom -=   numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        TD_ia_bom -= numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        TD_ia_bom -= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        TD_ia_bom -= numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        TD_ia_bom -=   numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        TD_ia_bom -= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        TD_ia_bom -= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
    return (TD_ia_top,TD_ia_bom)

def non_collinear_Amat_r_parallel(e_ia, kernel, z0, xctype, weights, ais, uvs, mf,ncpu):
    fxc,hyec = kernel 
    nstates = z0.shape[-1]
    C_vir, C_occ = uvs
    nocc = C_occ.shape[-1]
    nvir = C_vir.shape[-1]
    
    x0 = z0[:z0.shape[0]//2]
    y0 = z0[z0.shape[0]//2:]
    
    ngrid = fxc[0].shape[-1]
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    # import pdb
    # pdb.set_trace()
    
    if xctype == 'LDA':
        for para in NX_list:
            idxi,idxf = para
            ais_para = []
            fxc_para = []
            for i in range(len(ais)):
                ais_para.append(ais[i][:,:,idxi:idxf])
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])    
            para_results.append(pool.apply_async(A_ia_non_collinear_r,
                                (xctype, z0, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA':
        for para in NX_list:
            idxi,idxf = para
            ais_para = []
            fxc_para = []
            for i in range(len(ais)):
                ais_para.append(ais[i][:,:,idxi:idxf])
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(A_ia_non_collinear_r,
                                (xctype, z0, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    # import pdb
    # pdb.set_trace()
    
    TD_ia_top,TD_ia_bom = (0.0, 0.0)
    for result_para in para_results:
        result = result_para.get()
        TD_ia_top += result[0]
        TD_ia_bom += result[1]

    # The orbital energy difference is calculated here
    TD_ia_top += numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
    TD_ia_bom -= numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
    
    # The Coloumb Part of A.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    TD_ia_top += erimo
    
    # The Coloumb Part of B.
    dm1 = numpy.einsum('jbn,vj,ub->vun', y0.reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    TD_ia_top += erimo
    
    # The Coloumb Part of B*.
    dm1 = numpy.einsum('jbn,vj,ub->vun', x0.conj().reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    TD_ia_bom -= erimo
    
    # The Coloumb Part of A*.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', y0.conj().reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    TD_ia_bom -= erimo

    # The excat exchange is calculated.
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
        # import pdb
        # pdb.set_trace()
        # A Part.
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) >= 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,C_vir.conj(),C_occ, optimize=True)
        TD_ia_top -= erimo
         
        # B Part.
        dm2 = numpy.einsum('jbn,vj,ub->vun', y0.reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) > 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,C_vir.conj(),C_occ, optimize=True)
        TD_ia_top -= erimo
        
        # -B*Part.
        dm2 = numpy.einsum('jbn,vj,ub->vun', x0.conj().reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) >= 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,C_vir.conj(),C_occ, optimize=True)
        TD_ia_bom += erimo
        
        # -A*Part.
        dm2 = numpy.einsum('jbn,vj,ub->uvn', y0.conj().reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) >= 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,C_vir.conj(),C_occ, optimize=True)
        TD_ia_bom += erimo

    TD_ia_top = TD_ia_top.reshape(-1,nstates)
    TD_ia_bom = TD_ia_bom.reshape(-1,nstates).conj()
    TD_ia = numpy.concatenate((TD_ia_top,TD_ia_bom),axis=0)
    return TD_ia

def non_collinear_Amat_r(e_ia, kernel, z0, xctype, weights, ais, uvs, mf,*args):
    fxc,hyec = kernel 
    mo_vir_L, mo_vir_S, mo_occ_L , mo_occ_S = ais
    nstates = z0.shape[-1]
    C_vir, C_occ = uvs
    nocc = C_occ.shape[-1]
    nvir = C_vir.shape[-1]
    
    x0 = z0[:z0.shape[0]//2]
    y0 = z0[z0.shape[0]//2:]
    
    if xctype == 'LDA':
        n_n,n_s,s_s = fxc 
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i       
        ai_rho = numpy.einsum('cxpa,cxpi->pai', mo_vir_L.conj(), mo_occ_L, optimize=True)
        ai_rho+= numpy.einsum('cxpa,cxpi->pai', mo_vir_S.conj(), mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        betasigma_x = numpy.array(
            [[0,1,0,0],
             [1,0,0,0],
             [0,0,0,-1],
             [0,0,-1,0]]
        )
        betasigma_y = numpy.array(
            [[0,-1.0j,0,0],
             [1.0j,0,0,0],
             [0,0,0,1.0j],
             [0,0,-1.0j,0]]
        )
        betasigma_z = numpy.array(
            [[1,0,0,0],
             [0,-1,0,0],
             [0,0,-1,0],
             [0,0,0,1]]
        )
        ai_Mx = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_x[:2,:2], mo_occ_L, optimize=True)
        ai_Mx+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_x[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_y[:2,:2], mo_occ_L, optimize=True)
        ai_My+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_y[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_z[:2,:2], mo_occ_L, optimize=True)
        ai_Mz+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_z[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 =   numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)

        TD_ia_top  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True) # n_n
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        TD_ia_top += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        # The orbital energy difference is calculated here
        TD_ia_top += numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), y0.reshape(nocc,nvir,nstates), optimize=True)
        M1   = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), y0.reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_top += numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1.conj(), optimize=True) # n_n B_ia_top Part.
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1.conj(), optimize=True) # n_s B_ia_top Part.
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1.conj(), optimize=True) # s_n B_ia_top Part.
        TD_ia_top += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1.conj(), optimize=True) # s_s B_ia_top Part.
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        M1 =   numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.conj().reshape(nocc,nvir,nstates), optimize=True)

        TD_ia_bom  =-numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1.conj(), optimize=True) # n_n
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1.conj(), optimize=True) # n_s
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1.conj(), optimize=True) # s_n
        TD_ia_bom -= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1.conj(), optimize=True) # s_s
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        M1   = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_bom -= numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True) # n_n B_ia_top Part.
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s B_ia_top Part.
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n B_ia_top Part.
        TD_ia_bom -= numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s B_ia_top Part.
        
        # The orbital energy difference is calculated here
        TD_ia_bom -= numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        
    elif xctype == 'GGA':
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        
        ai_rho = numpy.einsum('cxpa,cpi->xpai', mo_vir_L.conj(), mo_occ_L[:,0], optimize=True)
        ai_rho+= numpy.einsum('cpa,cxpi->xpai', mo_vir_L[:,0].conj(), mo_occ_L, optimize=True)
        ai_rho+= numpy.einsum('cxpa,cpi->xpai', mo_vir_S.conj(), mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_rho+= numpy.einsum('cpa,cxpi->xpai', mo_vir_S[:,0].conj(), mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_rho[0]*= 0.5
        
        betasigma_x = numpy.array(
            [[0,1,0,0],
             [1,0,0,0],
             [0,0,0,-1],
             [0,0,-1,0]]
        )
        betasigma_y = numpy.array(
            [[0,-1.0j,0,0],
             [1.0j,0,0,0],
             [0,0,0,1.0j],
             [0,0,-1.0j,0]]
        )
        betasigma_z = numpy.array(
            [[1,0,0,0],
             [0,-1,0,0],
             [0,0,-1,0],
             [0,0,0,1]]
        )
        ai_Mx = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_x[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_Mx+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_x[:2,:2], mo_occ_L, optimize=True)
        ai_Mx+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_x[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mx+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_x[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_y[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_My+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_y[:2,:2], mo_occ_L, optimize=True)
        ai_My+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_y[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_y[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_z[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_Mz+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_z[:2,:2], mo_occ_L, optimize=True)
        ai_Mz+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_z[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_z[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        
        ai_Mx[0]*=0.5
        ai_My[0]*=0.5
        ai_Mz[0]*=0.5
        
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        ngrid = Nn_Nntmp.shape[-1]
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        TD_ia_top = numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        TD_ia_top+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        TD_ia_top+= numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        TD_ia_top += numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        TD_ia_top += numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        TD_ia_top += numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        TD_ia_top += numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        TD_ia_top += numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        TD_ia_top += numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        TD_ia_top += numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        TD_ia_top += numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        TD_ia_top += numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        # The orbital energy difference is calculated here
        TD_ia_top += numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True) 
        
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0], y0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0], y0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:], y0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:], y0.reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_top += numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        TD_ia_top +=   numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        TD_ia_top += numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        TD_ia_top += numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        TD_ia_top +=   numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        TD_ia_top += numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        TD_ia_top += numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        TD_ia_top += numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        TD_ia_top +=   numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        TD_ia_top += numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        TD_ia_top += numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        TD_ia_top += numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        TD_ia_top +=   numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        TD_ia_top += numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        TD_ia_top += numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0].conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0].conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:].conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:].conj(), y0.conj().reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        TD_ia_bom =- numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        TD_ia_bom -= numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        TD_ia_bom -= numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        TD_ia_bom -= numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        TD_ia_bom -= numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        TD_ia_bom -= numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        TD_ia_bom -= numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        TD_ia_bom -= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        TD_ia_bom -= numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        TD_ia_bom -= numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        TD_ia_bom -= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        TD_ia_bom -= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        # The orbital energy difference is calculated here
        TD_ia_bom -= numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), y0.conj().reshape(nocc,nvir,nstates), optimize=True) 
        
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0], x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0], x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:], x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:], x0.conj().reshape(nocc,nvir,nstates), optimize=True)
        
        TD_ia_bom -= numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        TD_ia_bom -= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        TD_ia_bom -= numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        TD_ia_bom -=   numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        TD_ia_bom -= numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        TD_ia_bom -= numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        TD_ia_bom -= numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        TD_ia_bom -=   numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        TD_ia_bom -= numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        TD_ia_bom -= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        TD_ia_bom -= numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        TD_ia_bom -=   numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        TD_ia_bom -= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        TD_ia_bom -= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
    # The Coloumb Part of A.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
    
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    TD_ia_top += erimo
    
    # The Coloumb Part of B.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', y0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
    
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    TD_ia_top += erimo
    
    # The Coloumb Part of -B*.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.conj().reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
    
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    TD_ia_bom -= erimo
    
    # The Coloumb Part of -A*.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', y0.conj().reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
    
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    TD_ia_bom -= erimo
    
    # The excat exchange is calculated
    omega, alpha, hyb = hyec
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        # A Part.
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) > 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,C_vir.conj(),C_occ, optimize=True)
        TD_ia_top -= erimo
        
        # B Part.
        dm2 = numpy.einsum('jbn,vj,ub->uvn', y0.reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) > 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,va,ui->ian',eri,C_vir.conj(),C_occ, optimize=True)
        TD_ia_top -= erimo
        
        # Approach 2
        # n2c = C_vir.shape[0]//2
        # eri_LL = mf.mol.intor('int2e_spinor')*hyb
        # eri_LS = mf.mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2*hyb
        # eri_SS = mf.mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4*hyb
        # # transform the eri to mo space.
        # if abs(omega) >= 1e-10:
        #     with mf.mol.with_range_coulomb(omega=omega):
        #         eri_LL += eri_LL*alpha/hyb
        #         eri_LS += eri_LS*alpha/hyb
        #         eri_SS += eri_SS*alpha/hyb
        
        # eri_LL_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LL,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],y0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_LS_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LS,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],y0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SL_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LS.transpose(2,3,0,1),
        #                       C_vir[:n2c].conj(),C_occ[:n2c],y0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SS_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_SS,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],y0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_LL_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_LS_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_SL_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_SS_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True)  
        
        
        # -B*Part.
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.conj().reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) > 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,va,ui->ian',eri,C_vir.conj(),C_occ, optimize=True)
        TD_ia_bom += erimo
        
        # eri_LL_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LL,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],x0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_LS_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LS,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],x0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SL_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LS.transpose(2,3,0,1),
        #                       C_vir[:n2c].conj(),C_occ[:n2c],x0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SS_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_SS,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],x0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_LL_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_LS_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_SL_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_SS_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True)
        
        # -A*Part.
        dm2 = numpy.einsum('jbn,vj,ub->uvn', y0.conj().reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) > 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,ua,vi->ian',eri,C_vir.conj(),C_occ, optimize=True)
        TD_ia_bom += erimo
    
    TD_ia_top = TD_ia_top.reshape(-1,nstates)
    TD_ia_bom = TD_ia_bom.reshape(-1,nstates).conj()
    
    TD_ia = numpy.concatenate((TD_ia_top,TD_ia_bom),axis=0)
    return TD_ia


class Solver_TDDFT():
    def __init__(self, mf,mf2, Extype, kernel, nstates=3, init_guess=None, scheme='DAVIDSON',
                 max_cycle = 50, conv_tol = 1.0E-8, cutoff = 25, diff = 1e-4, parallel=False, 
                 ncpu=None,Whkerl=False):
        
        if init_guess is None:
            init_guess = init_guess_naive
        
        self.init_guess = init_guess
        # Excitation type.
        self.Extype = Extype
        # Meanfield object
        self.mf = mf
        self.mf2 = mf2
        # ((fxc), eri_ao) fxc is the 2nd functional derivatives
        self.kernel = kernel
        # max iteration cycles
        self.max_cycle = max_cycle
        # convergence tolerance
        self.conv_tol = conv_tol
        # number of intrested states
        self.nstates = nstates
        # Diagnostic method LANCOV or DAVIDSON
        self.scheme = scheme
        # Number of cutoff cycles
        self.cutoff = cutoff
        # uks_to_gks: diff value. 
        self.diff = 1e-4
        # Whether parallel or not.
        self.parallel = parallel
        self.ncpu = None
        # Whether to use thw whole kernel or not in DAVIDSON or not.
        self.Whkerl = Whkerl
        
        if Extype == 'SPIN_CONSERVED':
            if parallel:
                self.AmB_Matx = spin_conserving_AmB_matx_parallel
                self.ApB_Matx = spin_conserving_ApB_matx_parallel
            else:
                self.AmB_Matx = spin_conserving_AmB_matx
                self.ApB_Matx = spin_conserving_ApB_matx
                
            self.get_Diagelmet_of_A = get_Diagelemt_of_O_sc_simply
            self.get_e_ia_and_mo = get_e_ia_and_mo
        
        elif 'SPIN' in Extype: # == 'SPIN_FLIP_UP' or Extype == 'SPIN_FLIP_DOWN'
            if parallel:
                self.AmB_Matx = spin_flip_AmB_matx_parallel
                self.ApB_Matx = spin_flip_ApB_matx_parallel
            else:
                self.AmB_Matx = spin_flip_AmB_matx
                self.ApB_Matx = spin_flip_ApB_matx
                
            self.get_Diagelmet_of_A = get_Diagelemt_of_O_sf_simply
            self.get_e_ia_and_mo = get_e_ia_and_mo
        
        elif Extype == 'GKS' or Extype == None:
            if self.mf2 is not None:
                self.get_e_ia_and_mo = get_nc_utg_e_ia_and_mo  
            else:
                self.get_e_ia_and_mo = get_nc_e_ia_and_mo 
            if parallel:
                self.AmB_Matx = non_collinear_AmB_matx_parallel
                self.ApB_Matx = non_collinear_ApB_matx_parallel
            else:
                self.AmB_Matx = non_collinear_AmB_matx
                self.ApB_Matx = non_collinear_ApB_matx
            self.get_Diagelmet_of_A = get_Diagelemt_of_O_nc_simply
                
        elif Extype == 'DKS':
            if parallel:
                self.Amatx = non_collinear_Amat_r_parallel
            else:
                self.Amatx = non_collinear_Amat_r
                
            self.get_e_ia_and_mo = get_nc_r_e_ia_and_mo
            self.get_Diagelmet_of_A = get_Diagelemt_of_O_nc_simply_r
                                
    def solver(self, ao, xctype):
        # Main solver
        # Pre-calculations
        weights = self.mf.grids.weights
        # ais means molecular orbitals with single orbital form rather than pair form.
        if self.Extype == 'GKS' or self.Extype == None:
            e_ia, ais, uvs = self.get_e_ia_and_mo(self.mf,self.mf2,xctype,ao,self.diff,self.Extype)[:3]
            if self.mf2 is not None:
                self.mf = self.mf2
        else:
            e_ia, ais, uvs = self.get_e_ia_and_mo(self.mf,xctype,ao,self.Extype)[:3]
        
        z0 = self.init_guess(self.Extype,self.nstates, e_ia)
            
        # Note that self.nstates <= nstate_new
        # Because nstate_new may contains more states that degenerated
        nstate_new = z0.shape[-1]
        omega0 = numpy.zeros((nstate_new))
        # hyec means hybrid functional coffiecients.
        fxc,hyec = self.kernel
       
        # Lancov
        if self.scheme.upper() == 'LAN':
            raise RuntimeError("Lancov hasn't been implemented.")
            
        elif self.scheme.upper() == 'DAVIDSON':
            # (A-B)(A+B)(X+Y) = w^2(X+Y) -> OZ = w^2 Z for non-relativistic cases.
            # [[ A   B ],
            #  [-B*,-A*]] for relativistic case.
            D = self.get_Diagelmet_of_A(e_ia)
           
            for icycle in range(self.max_cycle):
                # CUTOFF length
                if icycle%self.cutoff == 0:
                    if self.Extype != 'DKS':
                        z0 = z0[:,:nstate_new]
                    else:
                        z0 = z0[:,:nstate_new]
                  
                if self.Extype != 'DKS':
                    # Project to the sub-space
                    ApBz0  = self.ApB_Matx(e_ia, (fxc,hyec), z0, xctype, weights, ais, uvs, self.mf,self.ncpu)
                    z0tAmB = self.AmB_Matx(e_ia, (fxc,hyec), z0, xctype, weights, ais, uvs, self.mf,self.ncpu)
                    z0TDz0 = numpy.einsum('vm,vn->mn',z0tAmB,ApBz0,optimize=True)

                    omega_tmp, c_small = numpy.linalg.eig(z0TDz0)
                    idx_eigen = numpy.argsort(omega_tmp)
                    omega_tmp=omega_tmp[idx_eigen]
                    c_small = c_small[:,idx_eigen]
                    
                    omega = numpy.emath.sqrt(omega_tmp)
                    idx_pov = numpy.where(omega.real>0)[0]
                    c_small = c_small[:,idx_pov]
                    omega_p = omega[idx_pov]
                else:
                    Az0 = self.Amatx(e_ia, (fxc,hyec), z0, xctype, weights, ais, uvs, self.mf,self.ncpu)
                    z0TDz0 = z0.conj().T@Az0
                 
                    omega_tmp, c_small = numpy.linalg.eig(z0TDz0)
                    idx_pov = numpy.where(omega_tmp.real>0)[0]
                    omega_tmp = omega_tmp[idx_pov]
                    c_small = c_small[:,idx_pov]
                    
                    idx_eigen = numpy.argsort(omega_tmp)
                    omega_tmp = omega_tmp[idx_eigen]
                    c_small = c_small[:,idx_eigen]
                    omega_p = omega_tmp*1.0
                    
                if omega_p.size < z0.shape[-1]:
                    er_value = z0.shape[-1]- omega_p.size
                    if omega_p.size >= 2:
                        w_step = omega_p[-1] - omega_p[-2]
                        for i in range(er_value):
                            omega_p = numpy.append(omega_p,w_step+omega_p[-1])
                    else:
                        for i in range(er_value):
                            omega_p = numpy.append(omega_p,omega_p[-1]+0.05)
                
    
                omega_diff = numpy.abs(omega_p[:nstate_new]-omega0)
                print("Difference")
                print(omega_diff)
                omega0 = omega_p[:nstate_new]
                print(omega_p[:self.nstates].real*27.21138386)
                print(f"In circle {icycle}, {(omega_diff<self.conv_tol).sum()}/{nstate_new} states converged!")
                if (omega_diff<self.conv_tol).sum() == nstate_new:
                    break
                
                # compute the residual -> rk = omega I - A x_tot c_small
                if self.Extype != 'DKS':
                    z1 = -numpy.einsum('n,tn->tn',omega_tmp[idx_pov],z0@c_small, optimize = True)
                    # This is a bug.
                    TDz0 = numpy.einsum('vm,mn->vn',z0,z0TDz0,optimize = True)
                    z1 += TDz0@c_small
                    
                    for i in range(z1.shape[-1]):
                        Dinv = numpy.diag(1/(omega_tmp[idx_pov][i]-D+1e-99))
                        z1[:,i] = numpy.einsum('ts,s->t',Dinv,z1[:,i], optimize = True)
                    z0 = numpy.linalg.qr(numpy.concatenate((z0, z1[:,:nstate_new]),axis=1))[0]
                    
                    # For calculating oscillator_strength.
                    self.ApBz0 = self.ApB_Matx(e_ia, (fxc,hyec), z0, xctype, weights, ais, uvs, self.mf,self.ncpu)
                else:
                    # import pdb
                    # pdb.set_trace()
                    z1 = -numpy.einsum('n,tn->tn',omega_tmp,z0@c_small, optimize = True)
                    TDz0 = Az0
                    z1 += TDz0@c_small
                    
                    for i in range(z1.shape[-1]):
                        Dinv = numpy.diag(1/(omega_tmp[i]-D+1e-99))
                        z1[:,i] = numpy.einsum('ts,s->t',Dinv,z1[:,i], optimize = True)
                    z0 = numpy.linalg.qr(numpy.concatenate((z0, z1[:,:nstate_new].real),axis=1))[0]
                    
            if icycle == self.max_cycle-1:
                raise RuntimeError("Not convergence!")
            
        else:
            raise NotImplementedError("Only Davidson is implemented!")
        # import pdb
        # pdb.set_trace()
        return omega_p[:self.nstates].real*27.21138386, z0[:,:self.nstates]
    