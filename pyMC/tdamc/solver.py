#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2022-07-01 15:27:46
LastEditTime: 2022-10-28 05:48:21
LastEditors: Li Hao
Description: 
FilePath: /pyMC/tdamc/solver.py

 May the force be with you!
'''

from re import M
import numpy
import time
from pyscf.lib import logger
from pyscf import lib
from pyMC.tdamc import tdamc_uks 
from pyMC.tdamc import tdamc_gks 
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
    if Extype == 'SPIN_CONSERVED' or len(e_ia)==2:
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
    else:
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
    return x0
    
def get_e_ia_and_mo(mf,xctype,ao,Extype='SPIN_FLIP_UP'):
    if Extype == 'SPIN_FLIP_UP' or Extype == 'SPIN_FLIP_DOWN':
        e_ai,ais,uvs = tdamc_uks.get_iAmat_and_mo_tda(mf,xctype,ao,Extype)
        # From e_ai to e_ia
        nvir = ais[0].shape[-1]
        e_ia = e_ai.reshape(nvir,-1).T.ravel()
    elif Extype == 'SPIN_CONSERVED':
        # import pdb
        # pdb.set_trace()
        e_ai,ais,uvs = tdamc_uks.get_iAmat_and_mo_tda(mf,xctype,ao,Extype)
        e_ai_aa,e_ai_bb = e_ai
        nvir_a = ais[0].shape[-1]
        nvir_b = ais[2].shape[-1]
        e_ia_aa = e_ai_aa.reshape(nvir_a,-1).T.ravel()
        e_ia_bb = e_ai_bb.reshape(nvir_b,-1).T.ravel()
        e_ia = (e_ia_aa,e_ia_bb)
    return e_ia,ais,uvs
    
def get_nc_e_ia_and_mo(mf,mf2,xctype,ao,diff,Extype='GKS'):
    e_ai,ais,uvs = tdamc_gks.get_iAmat_and_mo_tda(mf,xctype,ao)
    # From e_ai to e_ia
    nvir = ais[0].shape[-1]
    e_ia = e_ai.reshape(nvir,-1).T.ravel()
    return e_ia,ais,uvs


def get_nc_utg_e_ia_and_mo(mf1,mf2,xctype,ao,diff,Extype='GKS'):
    e_ai,ais,uvs = tdamc_gks.uks_to_gks_iAamt_and_mo_tda(mf1,mf2,xctype,ao,diff)
    # From e_ai to e_ia
    nvir = ais[0].shape[-1]
    e_ia = e_ai.reshape(nvir,-1).T.ravel()
    return e_ia,ais,uvs

def get_nc_r_e_ia_and_mo(mf,xctype,ao,Extype='DKS'):
    e_ai,ais,uvs = tdamc_gks.get_iAmat_and_mo_tda_r(mf,xctype,ao)
    # From e_ai to e_ia
    nvir = uvs[0].shape[-1]
    e_ia = e_ai.reshape(nvir,-1).T.ravel()
    return e_ia,ais,uvs

def D_ia_spin_flip(xctype,mo_vir,mo_occ,fxc,weights):
    if xctype == 'LDA':
        ai = numpy.einsum('pa,pi->pai',mo_vir.conj(),mo_occ,optimize=True)
        s_s = fxc
        # *2 for xx,yy parts.
        D_ia = numpy.einsum('pai,p,pai->ia', ai, s_s*weights, ai.conj(), optimize=True)*2.0
        
    elif xctype == 'GGA':
        ai = numpy.einsum('pa,pi->pai',mo_vir[0].conj(),mo_occ[0],optimize=True)
        # \nabla 
        nabla_ai = numpy.einsum('xpa,pi->xpai',mo_vir[1:4].conj(),mo_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_vir[0].conj(),mo_occ[1:4],optimize=True)
        
        s_s, s_Ns, Ns_Ns = fxc
        D_ia = numpy.einsum('pai,p,pai->ia', ai, s_s*weights, ai.conj(), optimize=True)
        D_ia += numpy.einsum('pai,xp,xpai->ia', ai, s_Ns*weights, nabla_ai.conj(), optimize=True)
        D_ia += numpy.einsum('xpai,xp,pai->ia', nabla_ai, s_Ns*weights, ai.conj(), optimize=True)
        D_ia += numpy.einsum('xpai,xyp,ypai->ia', nabla_ai, Ns_Ns*weights, nabla_ai.conj(), optimize=True)
        D_ia *= 2.0
        
    elif xctype == 'MGGA':
        ai = numpy.einsum('pa,pi->pai',mo_vir[0].conj(),mo_occ[0],optimize=True)
        nabla_ai = numpy.einsum('xpa,pi->xpai',mo_vir[1:4].conj(),mo_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_vir[0].conj(),mo_occ[1:4],optimize=True)
        tau_ai = 0.5*numpy.einsum('xpa,xpi->pai',mo_vir[1:4].conj(),mo_occ[1:4],optimize=True)    
     
        s_s, s_Ns, Ns_Ns, u_u, s_u, Ns_u = fxc
    
        D_ia = numpy.einsum('pai,p,pai->ia', ai, s_s*weights,ai.conj(), optimize=True)
        D_ia+= numpy.einsum('xpai,xp,pai->ia', nabla_ai, s_Ns*weights, ai.conj(), optimize=True)
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai, s_Ns*weights, nabla_ai.conj(), optimize=True)
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', nabla_ai, Ns_Ns*weights, nabla_ai.conj(), optimize=True)
        D_ia+= numpy.einsum('pai,p,pai->ia', tau_ai, u_u*weights, tau_ai.conj(), optimize=True) #u_u
        D_ia+= numpy.einsum('pai,p,pai->ia', ai, s_u*weights, tau_ai.conj(), optimize=True) #s_u
        D_ia+= numpy.einsum('pai,p,pai->ia', tau_ai, s_u*weights, ai.conj(), optimize=True) #u_s
        D_ia+= numpy.einsum('pai,xp,xpai->ia', tau_ai, Ns_u*weights, nabla_ai.conj(), optimize=True) #u_Ns
        D_ia+= numpy.einsum('xpai,xp,pai->ia', nabla_ai, Ns_u*weights, tau_ai.conj(), optimize=True) #u_Ns
        D_ia *= 2.0    
        
    return D_ia

def get_Diagelemt_of_A_parallel(e_ia, kernel,xctype, weights, ais, uvs, mf,ncpu,Whkerl=False):
    ''' Get the Diagelemt of A matrix for spin-flip matrix in parellel. '''
    
    mo_vir, mo_occ = ais
    fxc, hyec = kernel
    omega, alpha,hyb = hyec
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
    
    if xctype == 'LDA':
        for para in NX_list:
            idxi,idxf = para
            para_results.append(pool.apply_async(D_ia_spin_flip,
                                (xctype, mo_vir[idxi:idxf], mo_occ[idxi:idxf], 
                                fxc[idxi:idxf], weights[idxi:idxf])))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA' or xctype == 'MGGA':
        for para in NX_list:
            idxi,idxf = para
            fxc_para = []
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(D_ia_spin_flip,
                                (xctype,mo_vir[:,idxi:idxf], mo_occ[:,idxi:idxf], 
                                fxc_para, weights[idxi:idxf])))
            
        pool.close()
        pool.join()
    
    D_ia = 0.0
    for result_para in para_results:
        result = result_para.get()
        D_ia += result
        
    # Whkerl determines whether use the whole kernel to calculate the diagonal element or not, which performances
    # a little effcet in the DAVIDSON scheme.
    if Whkerl:
    # import pdb
    # pdb.set_trace()
    # Exact HF exchange of hybrid functionals. 
        if numpy.abs(hyb) >= 1e-10:
            Cvir,Cocc = uvs
            nvir = Cvir.shape[-1]
            nocc = Cocc.shape[-1]
        
            eri_mo = numpy.empty((nocc,nvir))
            for i in range(nocc):
                for a in range(nvir):
                    dm1 = numpy.einsum('vi,ua->uv', Cocc[:,i].reshape(-1,1), Cvir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm1, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm1, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cvir[:,a],Cocc[:,i])  
            D_ia -= eri_mo
     
    # The orbital energy difference is added here
    D_ia = D_ia.ravel() 
    D_ia += e_ia
    
    return D_ia

def get_Diagelemt_of_A(e_ia, kernel,xctype, weights, ais, uvs, mf,*args,Whkerl=False):
    '''Get the diagelement of A matrix for spin-flip TDA.
    
    Parameters
    ----------
    Args:
        e_ia : numpy.narray
            The difference value between occupied (i) and virtual orbital (a) energies.
        kernel : tuple
            Spin-flip kernels.
        xctype : str
            The type of XC functionals, LDA, GGA, MGGA 
        weights : numpy.array
            The weights of solid angles in spin space
        ais : numpy.array
            Mo_coeff of virtual (a) and occupied (i) orbitals based mo.
        uvs : numpy.array
            Coeff of virtual (a) and occupied (i) orbitals based ao, mainly for the calculation of the hartree potential
            and the exact exchange.
        mf : SCF object
    
    Kwargs:
        Whkerl : bool
            The Kwarg determines the Diagelement whether includes the hartree potentia and the exact exchange or not, w-
            hich plays almost no effects in the solvation of the final matrix. 
               
    Returns:
    ----------
        D_ia: numpy.array
    '''
    
    mo_vir, mo_occ = ais
    fxc, hyec = kernel
    omega, alpha,hyb = hyec
    
    if xctype == 'LDA':
        ai = numpy.einsum('pa,pi->pai',mo_vir.conj(),mo_occ,optimize=True)
        s_s = fxc
        # *2 for xx,yy parts.
        D_ia = numpy.einsum('pai,p,pai->ia', ai, s_s*weights, ai.conj(), optimize=True)*2.0
        
    elif xctype == 'GGA':
        ai = numpy.einsum('pa,pi->pai',mo_vir[0].conj(),mo_occ[0],optimize=True)
        # \nabla 
        nabla_ai = numpy.einsum('xpa,pi->xpai',mo_vir[1:4].conj(),mo_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_vir[0].conj(),mo_occ[1:4],optimize=True)
        
        s_s, s_Ns, Ns_Ns = fxc
        D_ia = numpy.einsum('pai,p,pai->ia', ai, s_s*weights, ai.conj(), optimize=True)
        D_ia += numpy.einsum('pai,xp,xpai->ia', ai, s_Ns*weights, nabla_ai.conj(), optimize=True)
        D_ia += numpy.einsum('xpai,xp,pai->ia', nabla_ai, s_Ns*weights, ai.conj(), optimize=True)
        D_ia += numpy.einsum('xpai,xyp,ypai->ia', nabla_ai, Ns_Ns*weights, nabla_ai.conj(), optimize=True)
        D_ia *= 2.0
        
    elif xctype == 'MGGA':
        ai = numpy.einsum('pa,pi->pai',mo_vir[0].conj(),mo_occ[0],optimize=True)
        nabla_ai = numpy.einsum('xpa,pi->xpai',mo_vir[1:4].conj(),mo_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_vir[0].conj(),mo_occ[1:4],optimize=True)
        tau_ai = 0.5*numpy.einsum('xpa,xpi->pai',mo_vir[1:4].conj(),mo_occ[1:4],optimize=True)    
     
        s_s, s_Ns, Ns_Ns, u_u, s_u, Ns_u = fxc
    
        D_ia = numpy.einsum('pai,p,pai->ia', ai, s_s*weights,ai.conj(), optimize=True)
        D_ia+= numpy.einsum('xpai,xp,pai->ia', nabla_ai, s_Ns*weights, ai.conj(), optimize=True)
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai, s_Ns*weights, nabla_ai.conj(), optimize=True)
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', nabla_ai, Ns_Ns*weights, nabla_ai.conj(), optimize=True)
        D_ia+= numpy.einsum('pai,p,pai->ia', tau_ai, u_u*weights, tau_ai.conj(), optimize=True) #u_u
        D_ia+= numpy.einsum('pai,p,pai->ia', ai, s_u*weights, tau_ai.conj(), optimize=True) #s_u
        D_ia+= numpy.einsum('pai,p,pai->ia', tau_ai, s_u*weights, ai.conj(), optimize=True) #u_s
        D_ia+= numpy.einsum('pai,xp,xpai->ia', tau_ai, Ns_u*weights, nabla_ai.conj(), optimize=True) #u_Ns
        D_ia+= numpy.einsum('xpai,xp,pai->ia', nabla_ai, Ns_u*weights, tau_ai.conj(), optimize=True) #u_Ns
        D_ia *= 2.0    
    
    # Whkerl determines whether use the whole kernel to calculate the diagonal element or not, which performances
    # a little effcet in the DAVIDSON scheme.
    if Whkerl:
    # import pdb
    # pdb.set_trace()
    # Exact HF exchange of hybrid functionals. 
        if numpy.abs(hyb) >= 1e-10:
            Cvir,Cocc = uvs
            nvir = Cvir.shape[-1]
            nocc = Cocc.shape[-1]
        
            eri_mo = numpy.empty((nocc,nvir))
            for i in range(nocc):
                for a in range(nvir):
                    dm1 = numpy.einsum('vi,ua->uv', Cocc[:,i].reshape(-1,1), Cvir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm1, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm1, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cvir[:,a],Cocc[:,i])  
            D_ia -= eri_mo
     
    # The orbital energy difference is added here
    D_ia = D_ia.ravel() 
    D_ia += e_ia
    
    return D_ia

def D_ia_spin_conserving(xctype,mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ,fxc,weights):
    if xctype == 'LDA':
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
    
        n_n,n_s,s_s = fxc
       
        # import pdb
        # pdb.set_trace()
        D_ia_top  = numpy.einsum('pai,p,pai->ia', ai_aa, n_n*weights, ai_aa.conj(), optimize=True) # n_n
        D_ia_top += numpy.einsum('pai,p,pai->ia', ai_aa, n_s*weights, ai_aa.conj(), optimize=True) # n_s
        D_ia_top += numpy.einsum('pai,p,pai->ia', ai_aa, n_s*weights, ai_aa.conj(), optimize=True) # s_n
        D_ia_top += numpy.einsum('pai,p,pai->ia', ai_aa, s_s*weights, ai_aa.conj(), optimize=True) # s_s
         
        D_ia_bom  = numpy.einsum('pai,p,pai->ia', ai_bb, n_n*weights, ai_bb.conj(), optimize=True) # n_n
        D_ia_bom -= numpy.einsum('pai,p,pai->ia', ai_bb, n_s*weights, ai_bb.conj(), optimize=True) # n_s
        D_ia_bom -= numpy.einsum('pai,p,pai->ia', ai_bb, n_s*weights, ai_bb.conj(), optimize=True) # s_n
        D_ia_bom += numpy.einsum('pai,p,pai->ia', ai_bb, s_s*weights, ai_bb.conj(), optimize=True) # s_s
        
    elif xctype == 'GGA':
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        # \nabla 
        nabla_ai_aa = numpy.einsum('xpa,pi->xpai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        nabla_ai_bb = numpy.einsum('xpa,pi->xpai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)    
        
        n_n,n_s,n_Nn,n_Ns,s_s,s_Nn,s_Ns,Nn_Nn,Nn_Ns,Ns_Ns = fxc
       
        # nn
        D_ia_top = numpy.einsum('p,pai,pai->ia',n_n*weights,ai_aa,ai_aa.conj(),optimize=True)  
        D_ia_bom = numpy.einsum('p,pai,pai->ia',n_n*weights,ai_bb,ai_bb.conj(),optimize=True)
        
        # ns
        D_ia_top    += numpy.einsum('p,pai,pai->ia',n_s*weights,ai_aa,ai_aa.conj(),optimize=True)  
        D_ia_bom += -1*numpy.einsum('p,pai,pai->ia',n_s*weights,ai_bb,ai_bb.conj(),optimize=True)
        
        # sn
        D_ia_top    += numpy.einsum('p,pai,pai->ia',n_s*weights,ai_aa,ai_aa.conj(),optimize=True)  
        D_ia_bom += -1*numpy.einsum('p,pai,pai->ia',n_s*weights,ai_bb,ai_bb.conj(),optimize=True)
 
        # n_Nn
        D_ia_top += numpy.einsum('xp,pai,xpai->ia',n_Nn*weights,ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += numpy.einsum('xp,pai,xpai->ia',n_Nn*weights,ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Nn_n
        D_ia_top += numpy.einsum('xp,xpai,pai->ia',n_Nn*weights,nabla_ai_aa,ai_aa.conj(),optimize=True)
        D_ia_bom += numpy.einsum('xp,xpai,pai->ia',n_Nn*weights,nabla_ai_bb,ai_bb.conj(),optimize=True)
        
        # n_Ns
        D_ia_top      += numpy.einsum('xp,pai,xpai->ia',n_Ns*weights,ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += -1.0*numpy.einsum('xp,pai,xpai->ia',n_Ns*weights,ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Ns_n
        D_ia_top      += numpy.einsum('xp,xpai,pai->ia',n_Ns*weights,nabla_ai_aa,ai_aa.conj(),optimize=True)
        D_ia_bom += -1.0*numpy.einsum('xp,xpai,pai->ia',n_Ns*weights,nabla_ai_bb,ai_bb.conj(),optimize=True)

        # ss
        D_ia_top    += numpy.einsum('p,pai,pai->ia',s_s*weights,ai_aa,ai_aa.conj(),optimize=True)  
        D_ia_bom    += numpy.einsum('p,pai,pai->ia',s_s*weights,ai_bb,ai_bb.conj(),optimize=True)

        # s_Nn
        D_ia_top    += numpy.einsum('xp,pai,xpai->ia',s_Nn*weights,ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += -1*numpy.einsum('xp,pai,xpai->ia',s_Nn*weights,ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Nn_s
        D_ia_top    += numpy.einsum('xp,xpai,pai->ia',s_Nn*weights,nabla_ai_aa,ai_aa.conj(),optimize=True)
        D_ia_bom += -1*numpy.einsum('xp,xpai,pai->ia',s_Nn*weights,nabla_ai_bb,ai_bb.conj(),optimize=True)

        # s_Ns
        D_ia_top    += numpy.einsum('xp,pai,xpai->ia',s_Ns*weights,ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom    += numpy.einsum('xp,pai,xpai->ia',s_Ns*weights,ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Ns_s
        D_ia_top    += numpy.einsum('xp,xpai,pai->ia',s_Ns*weights,nabla_ai_aa,ai_aa.conj(),optimize=True)
        D_ia_bom    += numpy.einsum('xp,xpai,pai->ia',s_Ns*weights,nabla_ai_bb,ai_bb.conj(),optimize=True)
        
        # Nn_Nn part
        D_ia_top += numpy.einsum('xyp,xpai,ypai->ia',Nn_Nn*weights,nabla_ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += numpy.einsum('xyp,xpai,ypai->ia',Nn_Nn*weights,nabla_ai_bb,nabla_ai_bb.conj(),optimize=True)

        # Nn_Ns part
        D_ia_top    += numpy.einsum('xyp,xpai,ypai->ia',Nn_Ns*weights,nabla_ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += -1*numpy.einsum('xyp,xpai,ypai->ia',Nn_Ns*weights,nabla_ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Ns_Nn part
        D_ia_top    += numpy.einsum('xyp,xpai,ypai->ia',Nn_Ns*weights,nabla_ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += -1*numpy.einsum('xyp,xpai,ypai->ia',Nn_Ns*weights,nabla_ai_bb,nabla_ai_bb.conj(),optimize=True)

        # Ns_Ns part
        D_ia_top    += numpy.einsum('xyp,xpai,ypai->ia',Ns_Ns*weights,nabla_ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom    += numpy.einsum('xyp,xpai,ypai->ia',Ns_Ns*weights,nabla_ai_bb,nabla_ai_bb.conj(),optimize=True)
    
    return D_ia_top,D_ia_bom
    
def get_Diagelemt_of_A_sc_parallel(e_ia, kernel, xctype, weights, ais, uvs, mf,ncpu,Whkerl=False):
    ''' Get the Diagelemt of A matrix for spin-conserving matrix in parellel. '''
    e_ia_aa,e_ia_bb = e_ia
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    ngrid = weights.shape[-1]
    
    fxc,hyec = kernel
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
            para_results.append(pool.apply_async(D_ia_spin_conserving,(xctype, 
                                mo_a_vir[idxi:idxf], mo_a_occ[idxi:idxf], 
                                mo_b_vir[idxi:idxf], mo_b_occ[idxi:idxf],
                                fxc_para, weights[idxi:idxf])))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA' or xctype == 'MGGA':
        for para in NX_list:
            idxi,idxf = para
            fxc_para = []
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(D_ia_spin_conserving,(xctype,
                                mo_a_vir[:,idxi:idxf], mo_a_occ[:,idxi:idxf], 
                                mo_b_vir[:,idxi:idxf], mo_b_occ[:,idxi:idxf],
                                fxc_para, weights[idxi:idxf])))
            
        pool.close()
        pool.join()
    
    D_ia_top = 0.0
    D_ia_bom = 0.0
    for result_para in para_results:
        result = result_para.get()
        D_ia_top += result[0]
        D_ia_bom += result[1]
        
    # import pdb
    # pdb.set_trace()
    
    if Whkerl:
        Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
        # The hartree potential term.
        nvir_a = Ca_vir.shape[-1]
        nocc_a = Ca_occ.shape[-1]
        nvir_b = Cb_vir.shape[-1]
        nocc_b = Cb_occ.shape[-1]
        
        eri_mo = numpy.empty((nocc_a,nvir_a))
        for i in range(nocc_a):
            for a in range(nvir_a):
                dm1 = numpy.einsum('vi,ua->uv', Ca_occ[:,i].conj().reshape(-1,1), Ca_vir[:,a].reshape(-1,1), optimize=True)
                eri_ia = mf.get_j(mf.mol, dm1, hermi=0)
                eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Ca_vir[:,a].conj(),Ca_occ[:,i])  
        D_ia_top += eri_mo
        
        
        eri_mo = numpy.empty((nocc_b,nvir_b))
        for i in range(nocc_b):
            for a in range(nvir_b):
                dm1 = numpy.einsum('vi,ua->uv', Cb_occ[:,i].conj().reshape(-1,1), Cb_vir[:,a].reshape(-1,1), optimize=True)
                eri_ia = mf.get_j(mf.mol, dm1, hermi=0)
                eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cb_vir[:,a].conj(),Cb_occ[:,i])  
        D_ia_bom += eri_mo
        
        # The excat exchange is calculated
        if numpy.abs(hyb) >= 1e-10:
            eri_mo = numpy.empty((nocc_a,nvir_a))
            for i in range(nocc_a):
                for a in range(nvir_a):
                    dm1 = numpy.einsum('vi,ua->uv', Ca_occ[:,i].reshape(-1,1), Ca_vir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm1, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm1, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Ca_vir[:,a],Ca_occ[:,i])  
            D_ia_top -= eri_mo
            
            eri_mo = numpy.empty((nocc_b,nvir_b))
            for i in range(nocc_b):
                for a in range(nvir_b):
                    dm1 = numpy.einsum('vi,ua->uv', Cb_occ[:,i].reshape(-1,1), Cb_vir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm1, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm1, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cb_vir[:,a],Cb_occ[:,i])  
            D_ia_bom -= eri_mo
    # import pdb
    # pdb.set_trace()
    
    # The orbital energy difference is added here.
    D_ia_top = D_ia_top.ravel() 
    D_ia_top += e_ia_aa
    
    D_ia_bom = D_ia_bom.ravel() 
    D_ia_bom += e_ia_bb
    # import pdb
    # pdb.set_trace()
    D_ia = numpy.concatenate([D_ia_top,D_ia_bom],axis=0)
    return D_ia

def get_Diagelemt_of_A_sc(e_ia, kernel, xctype, weights, ais, uvs, mf,*args,Whkerl=False):
    '''Get the diagelement of A matrix for spin-conserving TDA
    
    Parameters
    ----------
    Args:
        e_ia : numpy.narray
            The difference value between occupied (i) and virtual orbital (a) energies.
        kernel : tuple
            Spin-flip kernels.
        xctype : str
            The type of XC functionals, LDA, GGA, MGGA 
        weights : numpy.array
            The weights of solid angles in spin space
        ais : numpy.array
            Mo_coeff of virtual (a) and occupied (i) orbitals based mo.
        uvs : numpy.array
            Coeff of virtual (a) and occupied (i) orbitals based ao, mainly for the calculation of the hartree potential
            and the exact exchange.
        mf : SCF object
    
    Kwargs:
        Whkerl : bool
            The Kwarg determines the Diagelement whether includes the hartree potentia and the exact exchange or not, w-
            hich plays almost no effects in the solvation of the final matrix. 
               
    Returns:
    ----------
        D_ia: numpy.array
    '''
    e_ia_aa,e_ia_bb = e_ia
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        n_n,n_s,s_s = fxc
       
        # import pdb
        # pdb.set_trace()
        D_ia_top  = numpy.einsum('pai,p,pai->ia', ai_aa, n_n*weights, ai_aa.conj(), optimize=True) # n_n
        D_ia_top += numpy.einsum('pai,p,pai->ia', ai_aa, n_s*weights, ai_aa.conj(), optimize=True) # n_s
        D_ia_top += numpy.einsum('pai,p,pai->ia', ai_aa, n_s*weights, ai_aa.conj(), optimize=True) # s_n
        D_ia_top += numpy.einsum('pai,p,pai->ia', ai_aa, s_s*weights, ai_aa.conj(), optimize=True) # s_s
         
        D_ia_bom  = numpy.einsum('pai,p,pai->ia', ai_bb, n_n*weights, ai_bb.conj(), optimize=True) # n_n
        D_ia_bom -= numpy.einsum('pai,p,pai->ia', ai_bb, n_s*weights, ai_bb.conj(), optimize=True) # n_s
        D_ia_bom -= numpy.einsum('pai,p,pai->ia', ai_bb, n_s*weights, ai_bb.conj(), optimize=True) # s_n
        D_ia_bom += numpy.einsum('pai,p,pai->ia', ai_bb, s_s*weights, ai_bb.conj(), optimize=True) # s_s
        
    elif xctype == 'GGA':
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
        # nn
        D_ia_top = numpy.einsum('p,pai,pai->ia',n_n*weights,ai_aa,ai_aa.conj(),optimize=True)  
        D_ia_bom = numpy.einsum('p,pai,pai->ia',n_n*weights,ai_bb,ai_bb.conj(),optimize=True)
        
        # ns
        D_ia_top    += numpy.einsum('p,pai,pai->ia',n_s*weights,ai_aa,ai_aa.conj(),optimize=True)  
        D_ia_bom += -1*numpy.einsum('p,pai,pai->ia',n_s*weights,ai_bb,ai_bb.conj(),optimize=True)
        
        # sn
        D_ia_top    += numpy.einsum('p,pai,pai->ia',n_s*weights,ai_aa,ai_aa.conj(),optimize=True)  
        D_ia_bom += -1*numpy.einsum('p,pai,pai->ia',n_s*weights,ai_bb,ai_bb.conj(),optimize=True)
 
        # n_Nn
        D_ia_top += numpy.einsum('xp,pai,xpai->ia',n_Nn*weights,ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += numpy.einsum('xp,pai,xpai->ia',n_Nn*weights,ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Nn_n
        D_ia_top += numpy.einsum('xp,xpai,pai->ia',n_Nn*weights,nabla_ai_aa,ai_aa.conj(),optimize=True)
        D_ia_bom += numpy.einsum('xp,xpai,pai->ia',n_Nn*weights,nabla_ai_bb,ai_bb.conj(),optimize=True)
        
        # n_Ns
        D_ia_top      += numpy.einsum('xp,pai,xpai->ia',n_Ns*weights,ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += -1.0*numpy.einsum('xp,pai,xpai->ia',n_Ns*weights,ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Ns_n
        D_ia_top      += numpy.einsum('xp,xpai,pai->ia',n_Ns*weights,nabla_ai_aa,ai_aa.conj(),optimize=True)
        D_ia_bom += -1.0*numpy.einsum('xp,xpai,pai->ia',n_Ns*weights,nabla_ai_bb,ai_bb.conj(),optimize=True)

        # ss
        D_ia_top    += numpy.einsum('p,pai,pai->ia',s_s*weights,ai_aa,ai_aa.conj(),optimize=True)  
        D_ia_bom    += numpy.einsum('p,pai,pai->ia',s_s*weights,ai_bb,ai_bb.conj(),optimize=True)

        # s_Nn
        D_ia_top    += numpy.einsum('xp,pai,xpai->ia',s_Nn*weights,ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += -1*numpy.einsum('xp,pai,xpai->ia',s_Nn*weights,ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Nn_s
        D_ia_top    += numpy.einsum('xp,xpai,pai->ia',s_Nn*weights,nabla_ai_aa,ai_aa.conj(),optimize=True)
        D_ia_bom += -1*numpy.einsum('xp,xpai,pai->ia',s_Nn*weights,nabla_ai_bb,ai_bb.conj(),optimize=True)

        # s_Ns
        D_ia_top    += numpy.einsum('xp,pai,xpai->ia',s_Ns*weights,ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom    += numpy.einsum('xp,pai,xpai->ia',s_Ns*weights,ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Ns_s
        D_ia_top    += numpy.einsum('xp,xpai,pai->ia',s_Ns*weights,nabla_ai_aa,ai_aa.conj(),optimize=True)
        D_ia_bom    += numpy.einsum('xp,xpai,pai->ia',s_Ns*weights,nabla_ai_bb,ai_bb.conj(),optimize=True)
        
        # Nn_Nn part
        D_ia_top += numpy.einsum('xyp,xpai,ypai->ia',Nn_Nn*weights,nabla_ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += numpy.einsum('xyp,xpai,ypai->ia',Nn_Nn*weights,nabla_ai_bb,nabla_ai_bb.conj(),optimize=True)

        # Nn_Ns part
        D_ia_top    += numpy.einsum('xyp,xpai,ypai->ia',Nn_Ns*weights,nabla_ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += -1*numpy.einsum('xyp,xpai,ypai->ia',Nn_Ns*weights,nabla_ai_bb,nabla_ai_bb.conj(),optimize=True)
        
        # Ns_Nn part
        D_ia_top    += numpy.einsum('xyp,xpai,ypai->ia',Nn_Ns*weights,nabla_ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom += -1*numpy.einsum('xyp,xpai,ypai->ia',Nn_Ns*weights,nabla_ai_bb,nabla_ai_bb.conj(),optimize=True)

        # Ns_Ns part
        D_ia_top    += numpy.einsum('xyp,xpai,ypai->ia',Ns_Ns*weights,nabla_ai_aa,nabla_ai_aa.conj(),optimize=True)
        D_ia_bom    += numpy.einsum('xyp,xpai,ypai->ia',Ns_Ns*weights,nabla_ai_bb,nabla_ai_bb.conj(),optimize=True)
    # import pdb
    # pdb.set_trace()
    
    if Whkerl:
        Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
        # The hartree potential term.
        nvir_a = Ca_vir.shape[-1]
        nocc_a = Ca_occ.shape[-1]
        nvir_b = Cb_vir.shape[-1]
        nocc_b = Cb_occ.shape[-1]
        
        eri_mo = numpy.empty((nocc_a,nvir_a))
        for i in range(nocc_a):
            for a in range(nvir_a):
                dm1 = numpy.einsum('vi,ua->uv', Ca_occ[:,i].conj().reshape(-1,1), Ca_vir[:,a].reshape(-1,1), optimize=True)
                eri_ia = mf.get_j(mf.mol, dm1, hermi=0)
                eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Ca_vir[:,a].conj(),Ca_occ[:,i])  
        D_ia_top += eri_mo
        
        eri_mo = numpy.empty((nocc_b,nvir_b))
        for i in range(nocc_b):
            for a in range(nvir_b):
                dm1 = numpy.einsum('ui,va->vu', Cb_occ[:,i].conj().reshape(-1,1), Cb_vir[:,a].reshape(-1,1), optimize=True)
                eri_ia = mf.get_j(mf.mol, dm1, hermi=0)
                eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cb_vir[:,a].conj(),Cb_occ[:,i])  
        D_ia_bom += eri_mo
        
        # The excat exchange is calculated
        if numpy.abs(hyb) >= 1e-10:
            eri_mo = numpy.empty((nocc_a,nvir_a))
            for i in range(nocc_a):
                for a in range(nvir_a):
                    dm1 = numpy.einsum('ui,va->vu', Ca_occ[:,i].reshape(-1,1), Ca_vir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm1, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm1, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Ca_vir[:,a],Ca_occ[:,i])  
            D_ia_top -= eri_mo
            
            eri_mo = numpy.empty((nocc_b,nvir_b))
            for i in range(nocc_b):
                for a in range(nvir_b):
                    dm1 = numpy.einsum('vi,ua->uv', Cb_occ[:,i].reshape(-1,1), Cb_vir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm1, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm1, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cb_vir[:,a],Cb_occ[:,i])  
            D_ia_bom -= eri_mo
    # import pdb
    # pdb.set_trace()
    
    # The orbital energy difference is added here.
    D_ia_top = D_ia_top.ravel() 
    D_ia_top += e_ia_aa
    
    D_ia_bom = D_ia_bom.ravel() 
    D_ia_bom += e_ia_bb
    # import pdb
    # pdb.set_trace()
    D_ia = numpy.concatenate([D_ia_top,D_ia_bom],axis=0)
    return D_ia

def D_ia_nc(xctype,ais,fxc):
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = ais
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
        D_ia = numpy.einsum('pai,p,pai->ia', ai_rho, n_n, ai_rho.conj(), optimize=True).astype(numpy.complex128) # n_n
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho, n_s, ai_s.conj(), optimize=True) # n_s
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_s, n_s, ai_rho.conj(), optimize=True) # s_n
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', ai_s, s_s, ai_s.conj(), optimize=True) # s_s
        
        
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
        # g means nabla_i, h means m_i
        ai_ns = numpy.einsum('xynai->yxnai',ai_ns)
        
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        # import pdb
        # pdb.set_trace()
        D_ia = numpy.einsum('pai,p,pai->ia', ai_rho, n_n, ai_rho.conj(), optimize=True).astype(numpy.complex128) # n_n
        
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho, n_s, ai_s.conj(), optimize=True) # n_s
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_s, n_s, ai_rho.conj(), optimize=True) # s_n
        
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho, n_Nn, ai_nrho.conj(), optimize=True) # n_Nn
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_nrho, n_Nn, ai_rho.conj(), optimize=True) # Nn_n
        
        D_ia+= numpy.einsum('pai,yxp,yxpai->ia', ai_rho, n_Ns, ai_ns.conj(), optimize=True) # n_Ns
        D_ia+= numpy.einsum('yxpai,yxp,pai->ia', ai_ns, n_Ns, ai_rho.conj(), optimize=True) # Ns_n
        
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', ai_s, s_s, ai_s.conj(), optimize=True) # s_s
        
        D_ia+= numpy.einsum('xpai,yxp,ypai->ia', ai_s, s_Nn, ai_nrho.conj(), optimize=True) # s_Nn
        D_ia+= numpy.einsum('ypai,yxp,xpai->ia', ai_nrho, s_Nn, ai_s.conj(), optimize=True) # Nn_s
        
        D_ia+= numpy.einsum('xpai,zyxp,zypai->ia', ai_s, s_Ns, ai_ns.conj(), optimize=True) # s_Ns
        D_ia+= numpy.einsum('zxpai,zyxp,ypai->ia', ai_ns, s_Ns, ai_s.conj(), optimize=True) # Ns_s
        
        ngrid = s_s.shape[-1]
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        D_ia += numpy.einsum('xpai,yxp,ypai->ia', ai_nrho, Nn_Nn, ai_nrho.conj(), optimize=True) # Nn_Nn
        
        D_ia += numpy.einsum('zpai,zyxp,yxpai->ia', ai_nrho, Nn_Ns, ai_ns.conj(), optimize=True) # Nn_Ns
        D_ia += numpy.einsum('zxpai,zxyp,ypai->ia', ai_ns, Nn_Ns, ai_nrho.conj(), optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        D_ia+= numpy.einsum('wzpai,wyzxp,yxpai->ia', ai_ns, Ns_Ns, ai_ns.conj(), optimize=True) # Ns_Ns
        
    elif xctype == 'MGGA':
        pass
    
    return D_ia
    
def get_nc_Diagelemt_of_A_parallel(e_ia, kernel,xctype, weights, ais, uvs, mf,ncpu,Whkerl=False,):
    # import pdb
    # pdb.set_trace()
    fxc, hyec = kernel
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    Ca_vir, Ca_occ, Cb_vir, Cb_occ = uvs
    ngrid = mo_a_vir.shape[-1]
    
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
            ais_para = []
            fxc_para = []
            for i in range(len(ais)):
                ais_para.append(ais[i][idxi:idxf])
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])    
            para_results.append(pool.apply_async(D_ia_nc,
                                (xctype, ais_para, fxc_para)))
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
            para_results.append(pool.apply_async(D_ia_nc,
                                (xctype, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    
    D_ia = 0.0
    for result_para in para_results:
        result = result_para.get()
        D_ia += result
    
    # Whkerl determines whether use the whole kernel to calculate the diagonal element or not, which performances
    # a little effcet in the DAVIDSON scheme.
    if Whkerl:
        Cvir = numpy.concatenate((Ca_vir, Cb_vir),axis=0)
        Cocc = numpy.concatenate((Ca_occ, Cb_occ),axis=0) 
        # Calculation of hartree potential. 
        nvir = Cvir.shape[-1]
        nocc = Cocc.shape[-1]
        eri_mo = numpy.empty((nocc,nvir))
        for i in range(nocc):
            for a in range(nvir):
                dm1 = numpy.einsum('vi,ua->uv', Cocc[:,i].conj().reshape(-1,1), Cvir[:,a].reshape(-1,1), optimize=True)
                eri_ia = mf.get_j(mf.mol, dm1, hermi=0)
                eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cvir[:,a].conj(),Cocc[:,i])  
        D_ia += eri_mo
    
        # Exact HF exchange of hybrid functionals.
        omega, alpha,hyb = hyec
        if numpy.abs(hyb) >= 1e-10:
            eri_mo = numpy.empty((nocc,nvir)) 
            for i in range(nocc):
                for a in range(nvir):
                    dm2 = numpy.einsum('vi,ua->uv', Cocc[:,i].conj().reshape(-1,1), Cvir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm2, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm2, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cvir[:,a].conj(),Cocc[:,i])  
            D_ia -= eri_mo

    # The orbital energy difference is calculated here
    D_ia = D_ia.ravel()
    D_ia = e_ia
  
    return D_ia

def D_ia_nc_r(xctype,ais,fxc):
    mo_vir_L,mo_vir_S,mo_occ_L,mo_occ_S = ais
    
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
        
        D_ia = numpy.einsum('pai,p,pai->ia', ai_rho, n_n, ai_rho.conj(), optimize=True).astype(numpy.complex128) # n_n
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho, n_s, ai_s.conj(), optimize=True) # n_s
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_s, n_s, ai_rho.conj(), optimize=True) # s_n
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', ai_s, s_s, ai_s.conj(), optimize=True) # s_s
        
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
        # y means nabla_i, x means m_i
        ai_s = numpy.einsum('xypai->yxpai',ai_s)
        
        ngrid = Nn_Nntmp.shape[-1]
        
        # import pdb
        # pdb.set_trace()
        D_ia = numpy.einsum('pai,p,pai->ia', ai_rho[0], n_n, ai_rho[0].conj(), optimize=True).astype(numpy.complex128) # n_n
        
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho[0], n_s, ai_s[0].conj(), optimize=True) # n_s
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_s[0], n_s, ai_rho[0].conj(), optimize=True) # s_n
        
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho[0], n_Nn, ai_rho[1:].conj(), optimize=True) # n_Nn
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_rho[1:], n_Nn, ai_rho[0].conj(), optimize=True) # Nn_n
        
        D_ia+= numpy.einsum('pai,yxp,yxpai->ia', ai_rho[0], n_Ns, ai_s[1:].conj(), optimize=True) # n_Ns
        D_ia+= numpy.einsum('yxpai,yxp,pai->ia', ai_s[1:], n_Ns, ai_rho[0].conj(), optimize=True) # Ns_n
        
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', ai_s[0], s_s, ai_s[0].conj(), optimize=True) # s_s
        
        D_ia+= numpy.einsum('xpai,yxp,ypai->ia', ai_s[0], s_Nn, ai_rho[1:].conj(), optimize=True) # s_Nn
        D_ia+= numpy.einsum('ypai,yxp,xpai->ia', ai_rho[1:], s_Nn, ai_s[0].conj(), optimize=True) # Nn_s
        
        D_ia+= numpy.einsum('xpai,zyxp,zypai->ia', ai_s[0], s_Ns, ai_s[1:].conj(), optimize=True) # s_Ns
        D_ia+= numpy.einsum('zxpai,zxyp,ypai->ia', ai_s[1:], s_Ns, ai_s[0].conj(), optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        D_ia+= numpy.einsum('xpai,yxp,ypai->ia', ai_rho[1:], Nn_Nn, ai_rho[1:].conj(), optimize=True) # Nn_Nn
        
        D_ia+= numpy.einsum('zpai,zyxp,yxpai->ia', ai_rho[1:], Nn_Ns, ai_s[1:].conj(), optimize=True) # Nn_Ns
        D_ia+= numpy.einsum('zxpai,zxyp,ypai->ia', ai_s[1:], Nn_Ns, ai_rho[1:].conj(), optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        D_ia+= numpy.einsum('wzpai,wyzxp,yxpai->ia', ai_s[1:], Ns_Ns, ai_s[1:].conj(), optimize=True) # Ns_Ns
    
    return D_ia    
    
def get_nc_r_Diagelemt_of_A_parallel(e_ia, kernel,xctype, weights, ais, uvs, mf,ncpu,Whkerl=False):
    # import pdb
    # pdb.set_trace()
    fxc, hyec = kernel
    C_vir, C_occ = uvs
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
            para_results.append(pool.apply_async(D_ia_nc_r,
                                (xctype, ais_para, fxc_para)))
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
            para_results.append(pool.apply_async(D_ia_nc_r,
                                (xctype, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    
    D_ia = 0.0
    for result_para in para_results:
        result = result_para.get()
        D_ia += result

    # Whkerl determines whether use the whole kernel to calculate the diagonal element or not, which performances
    # a little effcet in the DAVIDSON scheme.
    if Whkerl:
        C_vir, C_occ = uvs

        # Calculation of hartree potential. 
        nvir = C_vir.shape[-1]
        nocc = C_occ.shape[-1]
        eri_mo = numpy.empty((nocc,nvir))
        for i in range(nocc):
            for a in range(nvir):
                dm1 = numpy.einsum('ui,va->uv', C_occ[:,i].conj().reshape(-1,1), C_vir[:,a].reshape(-1,1), optimize=True)
                eri_ia = mf.get_j(mf.mol, dm1, hermi=0)
                eri_mo[i,a] = numpy.einsum('uv,v,u',eri_ia,C_vir[:,a].conj(),C_occ[:,i])  
        D_ia += eri_mo
    
        # Exact HF exchange of hybrid functionals.
        omega, alpha,hyb = hyec
        if numpy.abs(hyb) >= 1e-10:
            eri_mo = numpy.empty((nocc,nvir)) 
            for i in range(nocc):
                for a in range(nvir):
                    dm2 = numpy.einsum('ui,va->uv', C_occ[:,i].conj().reshape(-1,1), C_vir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm2, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm2, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,v,u',eri_ia,C_vir[:,a].conj(),C_occ[:,i])  
            D_ia -= eri_mo

    # The orbital energy difference is calculated here
    D_ia = D_ia.ravel()
    D_ia += e_ia
  
    return D_ia


def get_nc_Diagelemt_of_A(e_ia, kernel,xctype, weights, ais, uvs, mf,*args,Whkerl=False):
    # import pdb
    # pdb.set_trace()
    fxc, hyec = kernel
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    Ca_vir, Ca_occ, Cb_vir, Cb_occ = uvs

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
        D_ia = numpy.einsum('pai,p,pai->ia', ai_rho, n_n, ai_rho.conj(), optimize=True).astype(numpy.complex128) # n_n
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho, n_s, ai_s.conj(), optimize=True) # n_s
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_s, n_s, ai_rho.conj(), optimize=True) # s_n
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', ai_s, s_s, ai_s.conj(), optimize=True) # s_s
        
        
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
        ai_ns = numpy.einsum('xynai->yxnai',ai_ns)
        
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        # import pdb
        # pdb.set_trace()
        D_ia = numpy.einsum('pai,p,pai->ia', ai_rho, n_n, ai_rho.conj(), optimize=True).astype(numpy.complex128) # n_n
        
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho, n_s, ai_s.conj(), optimize=True) # n_s
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_s, n_s, ai_rho.conj(), optimize=True) # s_n
        
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho, n_Nn, ai_nrho.conj(), optimize=True) # n_Nn
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_nrho, n_Nn, ai_rho.conj(), optimize=True) # Nn_n
        
        D_ia+= numpy.einsum('pai,yxp,yxpai->ia', ai_rho, n_Ns, ai_ns.conj(), optimize=True) # n_Ns
        D_ia+= numpy.einsum('yxpai,yxp,pai->ia', ai_ns, n_Ns, ai_rho.conj(), optimize=True) # Ns_n
        
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', ai_s, s_s, ai_s.conj(), optimize=True) # s_s
        
        D_ia+= numpy.einsum('xpai,yxp,ypai->ia', ai_s, s_Nn, ai_nrho.conj(), optimize=True) # s_Nn
        D_ia+= numpy.einsum('ypai,yxp,xpai->ia', ai_nrho, s_Nn, ai_s.conj(), optimize=True) # Nn_s
        
        D_ia+= numpy.einsum('xpai,zyxp,zypai->ia', ai_s, s_Ns, ai_ns.conj(), optimize=True) # s_Ns
        D_ia+= numpy.einsum('zxpai,zxyp,ypai->ia', ai_ns, s_Ns, ai_s.conj(), optimize=True) # Ns_s
        
        ngrid = s_s.shape[-1]
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        D_ia += numpy.einsum('xpai,yxp,ypai->ia', ai_nrho, Nn_Nn, ai_nrho.conj(), optimize=True) # Nn_Nn
        
        D_ia += numpy.einsum('zpai,zyxp,yxpai->ia', ai_nrho, Nn_Ns, ai_ns.conj(), optimize=True) # Nn_Ns
        D_ia += numpy.einsum('zxpai,zxyp,ypai->ia', ai_ns, Nn_Ns, ai_nrho.conj(), optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        D_ia+= numpy.einsum('wzpai,wyzxp,yxpai->ia', ai_ns, Ns_Ns, ai_ns.conj(), optimize=True) # Ns_Ns
        
    elif xctype == 'MGGA':
        pass
    
    # Whkerl determines whether use the whole kernel to calculate the diagonal element or not, which performances
    # a little effcet in the DAVIDSON scheme.
    if Whkerl:
        Cvir = numpy.concatenate((Ca_vir, Cb_vir),axis=0)
        Cocc = numpy.concatenate((Ca_occ, Cb_occ),axis=0) 
        # Calculation of hartree potential. 
        nvir = Cvir.shape[-1]
        nocc = Cocc.shape[-1]
        eri_mo = numpy.empty((nocc,nvir))
        for i in range(nocc):
            for a in range(nvir):
                dm1 = numpy.einsum('vi,ua->uv', Cocc[:,i].conj().reshape(-1,1), Cvir[:,a].reshape(-1,1), optimize=True)
                eri_ia = mf.get_j(mf.mol, dm1, hermi=0)
                eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cvir[:,a].conj(),Cocc[:,i])  
        D_ia += eri_mo
    
        # Exact HF exchange of hybrid functionals.
        omega, alpha,hyb = hyec
        if numpy.abs(hyb) >= 1e-10:
            eri_mo = numpy.empty((nocc,nvir)) 
            for i in range(nocc):
                for a in range(nvir):
                    dm2 = numpy.einsum('vi,ua->uv', Cocc[:,i].conj().reshape(-1,1), Cvir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm2, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm2, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,Cvir[:,a].conj(),Cocc[:,i])  
            D_ia -= eri_mo

    # The orbital energy difference is calculated here
    D_ia = D_ia.ravel()
    D_ia = e_ia
  
    return D_ia

def get_nc_r_Diagelemt_of_A(e_ia, kernel,xctype, weights, ais, uvs, mf,*args,Whkerl=False):
    # import pdb
    # pdb.set_trace()
    fxc, hyec = kernel
    mo_vir_L, mo_vir_S, mo_occ_L , mo_occ_S = ais
    C_vir, C_occ = uvs

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
        
        D_ia = numpy.einsum('pai,p,pai->ia', ai_rho, n_n, ai_rho.conj(), optimize=True).astype(numpy.complex128) # n_n
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho, n_s, ai_s.conj(), optimize=True) # n_s
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_s, n_s, ai_rho.conj(), optimize=True) # s_n
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', ai_s, s_s, ai_s.conj(), optimize=True) # s_s
        
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
        
        # ai_s = numpy.array([ai_Mx[0],ai_My[0],ai_Mz[0]])
        
        # construct gradient terms
        # ai_ns = numpy.array([ai_nMx,ai_nMy,ai_nMz])
        
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        # y means nabla_i, x means m_i
        ai_s = numpy.einsum('xypai->yxpai',ai_s)
        
        ngrid = Nn_Nntmp.shape[-1]
        
        # import pdb
        # pdb.set_trace()
        D_ia = numpy.einsum('pai,p,pai->ia', ai_rho[0], n_n, ai_rho[0].conj(), optimize=True).astype(numpy.complex128) # n_n
        
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho[0], n_s, ai_s[0].conj(), optimize=True) # n_s
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_s[0], n_s, ai_rho[0].conj(), optimize=True) # s_n
        
        D_ia+= numpy.einsum('pai,xp,xpai->ia', ai_rho[0], n_Nn, ai_rho[1:].conj(), optimize=True) # n_Nn
        D_ia+= numpy.einsum('xpai,xp,pai->ia', ai_rho[1:], n_Nn, ai_rho[0].conj(), optimize=True) # Nn_n
        
        D_ia+= numpy.einsum('pai,yxp,yxpai->ia', ai_rho[0], n_Ns, ai_s[1:].conj(), optimize=True) # n_Ns
        D_ia+= numpy.einsum('yxpai,yxp,pai->ia', ai_s[1:], n_Ns, ai_rho[0].conj(), optimize=True) # Ns_n
        
        D_ia+= numpy.einsum('xpai,xyp,ypai->ia', ai_s[0], s_s, ai_s[0].conj(), optimize=True) # s_s
        
        D_ia+= numpy.einsum('xpai,yxp,ypai->ia', ai_s[0], s_Nn, ai_rho[1:].conj(), optimize=True) # s_Nn
        D_ia+= numpy.einsum('ypai,yxp,xpai->ia', ai_rho[1:], s_Nn, ai_s[0].conj(), optimize=True) # Nn_s
        
        D_ia+= numpy.einsum('xpai,zyxp,zypai->ia', ai_s[0], s_Ns, ai_s[1:].conj(), optimize=True) # s_Ns
        D_ia+= numpy.einsum('zxpai,zxyp,ypai->ia', ai_s[1:], s_Ns, ai_s[0].conj(), optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        D_ia+= numpy.einsum('xpai,yxp,ypai->ia', ai_rho[1:], Nn_Nn, ai_rho[1:].conj(), optimize=True) # Nn_Nn
        
        D_ia+= numpy.einsum('zpai,zyxp,yxpai->ia', ai_rho[1:], Nn_Ns, ai_s[1:].conj(), optimize=True) # Nn_Ns
        D_ia+= numpy.einsum('zxpai,zxyp,ypai->ia', ai_s[1:], Nn_Ns, ai_rho[1:].conj(), optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        D_ia+= numpy.einsum('wzpai,wyzxp,yxpai->ia', ai_s[1:], Ns_Ns, ai_s[1:].conj(), optimize=True) # Ns_Ns
        
    elif xctype == 'MGGA':
        pass
    
    # Whkerl determines whether use the whole kernel to calculate the diagonal element or not, which performances
    # a little effcet in the DAVIDSON scheme.
    if Whkerl:
        C_vir, C_occ = uvs

        # Calculation of hartree potential. 
        nvir = C_vir.shape[-1]
        nocc = C_occ.shape[-1]
        eri_mo = numpy.empty((nocc,nvir))
        for i in range(nocc):
            for a in range(nvir):
                dm1 = numpy.einsum('vi,ua->uv', C_occ[:,i].conj().reshape(-1,1), C_vir[:,a].reshape(-1,1), optimize=True)
                eri_ia = mf.get_j(mf.mol, dm1, hermi=0)
                eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,C_vir[:,a].conj(),C_occ[:,i])  
        D_ia += eri_mo
    
        # Exact HF exchange of hybrid functionals.
        omega, alpha,hyb = hyec
        if numpy.abs(hyb) >= 1e-10:
            eri_mo = numpy.empty((nocc,nvir)) 
            for i in range(nocc):
                for a in range(nvir):
                    dm2 = numpy.einsum('vi,ua->uv', C_occ[:,i].conj().reshape(-1,1), C_vir[:,a].reshape(-1,1), optimize=True)
                    eri_ia = mf.get_k(mf.mol, dm2, hermi=0)
                    eri_ia *= hyb
                    if abs(omega) > 1e-10:
                        vklr = mf.get_k(mf.mol, dm2, hermi=0, omega=omega)
                        vklr*= (alpha - hyb)
                        eri_ia += vklr
                    eri_mo[i,a] = numpy.einsum('uv,u,v',eri_ia,C_vir[:,a].conj(),C_occ[:,i])  
            D_ia -= eri_mo

    # The orbital energy difference is calculated here
    D_ia = D_ia.ravel()
    D_ia += e_ia
  
    return D_ia

def A_ai_spin_flip(xctype,x0,mo_vir,mo_occ,fxc,weights):
    nvir = mo_vir.shape[-1]
    nocc = mo_occ.shape[-1]
    nstates = x0.shape[-1]
    if xctype == 'LDA':
        ai = numpy.einsum('pa,pi->pai',mo_vir.conj(),mo_occ,optimize=True)  
        s_s = fxc
        # The pseudo-density is calculated!
        rho1 = numpy.einsum('pbj,jbn->pn', ai.conj(),x0.reshape(nocc,nvir,nstates), optimize=True) 
        # *2 for xx,yy parts. 
        A_ai = numpy.einsum('pai,p,pn->ain', ai, s_s*weights, rho1, optimize=True)*2.0
        # The orbital energy difference is added here
        
    elif xctype == 'GGA':
        ai = numpy.einsum('pa,pi->pai',mo_vir[0].conj(),mo_occ[0],optimize=True)
        # \nabla 
        nabla_ai = numpy.einsum('xpa,pi->xpai',mo_vir[1:4].conj(),mo_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_vir[0].conj(),mo_occ[1:4],optimize=True)

        s_s, s_Ns, Ns_Ns = fxc
        ngrid = s_s.shape[-1]
        # The pseudo-density is calculated!
        rho1 = numpy.zeros((4, ngrid, nstates))
        rho1[0] = numpy.einsum('pbj,jbn->pn', ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        rho1[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        A_ai = numpy.einsum('pai,p,pn->ain', ai, s_s*weights, rho1[0], optimize=True)
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai, s_Ns*weights, rho1[1:], optimize=True)
        A_ai+= numpy.einsum('xpai,xp,pn->ain', nabla_ai, s_Ns*weights, rho1[0], optimize=True)
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', nabla_ai, Ns_Ns*weights, rho1[1:], optimize=True)
        # *2 for xx,yy parts. 
        A_ai *= 2.0
       
    elif xctype == 'MGGA':
        ai = numpy.einsum('pa,pi->pai',mo_vir[0].conj(),mo_occ[0],optimize=True)
        nabla_ai = numpy.einsum('xpa,pi->xpai',mo_vir[1:4].conj(),mo_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_vir[0].conj(),mo_occ[1:4],optimize=True)
        tau_ai = 0.5*numpy.einsum('xpa,xpi->pai',mo_vir[1:4].conj(),mo_occ[1:4],optimize=True)    
     
        s_s, s_Ns, Ns_Ns, u_u, s_u, Ns_u = fxc
        ngrid = s_s.shape[-1]
        # The pseudo-density is calculated!
        rho1 = numpy.zeros((5, ngrid, nstates))
        rho1[0] = numpy.einsum('pbj,jbn->pn', ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        rho1[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        rho1[4] = numpy.einsum('pbj,jbn->pn', tau_ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)

        A_ai = numpy.einsum('pai,p,pn->ain', ai, s_s*weights, rho1[0], optimize=True)
        A_ai+= numpy.einsum('xpai,xp,pn->ain', nabla_ai, s_Ns*weights, rho1[0], optimize=True)
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai, s_Ns*weights, rho1[1:4], optimize=True)
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', nabla_ai, Ns_Ns*weights, rho1[1:4], optimize=True)
        A_ai+= numpy.einsum('pai,p,pn->ain', tau_ai, u_u*weights, rho1[4], optimize=True) #u_u
        A_ai+= numpy.einsum('pai,p,pn->ain', ai, s_u*weights, rho1[4], optimize=True) #s_u
        A_ai+= numpy.einsum('pai,p,pn->ain', tau_ai, s_u*weights, rho1[0], optimize=True) #u_s
        A_ai+= numpy.einsum('pai,xp,xpn->ain', tau_ai, Ns_u*weights, rho1[1:4], optimize=True) #u_Ns
        A_ai+= numpy.einsum('xpai,xp,pn->ain', nabla_ai, Ns_u*weights, rho1[4], optimize=True) #u_Ns
        A_ai *= 2.0
    return A_ai

def spin_flip_Amatx_parallel(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,ncpu):
    ngrid = weights.shape[-1]
    mo_vir, mo_occ = ais
    Cvir,Cocc = uvs
    
    nvir = mo_vir.shape[-1]
    nocc = mo_occ.shape[-1]
    nstates = x0.shape[-1]
    
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
    
    # import pdb
    # pdb.set_trace()
    
    if xctype == 'LDA':
        for para in NX_list:
            idxi,idxf = para
            para_results.append(pool.apply_async(A_ai_spin_flip,
                                (xctype, x0, mo_vir[idxi:idxf], mo_occ[idxi:idxf], 
                                fxc[idxi:idxf], weights[idxi:idxf])))
        pool.close()
        pool.join()
    
    elif xctype == 'GGA' or xctype == 'MGGA':
        for para in NX_list:
            idxi,idxf = para
            fxc_para = []
            for i in range(len(fxc)):
                fxc_para.append(fxc[i][...,idxi:idxf])
            para_results.append(pool.apply_async(A_ai_spin_flip,
                                (xctype,x0,mo_vir[:,idxi:idxf], mo_occ[:,idxi:idxf], 
                                fxc_para, weights[idxi:idxf])))
            
        pool.close()
        pool.join()
    
    # import pdb
    # pdb.set_trace()
    
    A_ai = 0.0
    for result_para in para_results:
        result = result_para.get()
        A_ai += result

    # The orbital energy difference is calculated here
    A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ain',eri,Cvir,Cocc)
        A_ai -= erimo

    A_ai = A_ai.transpose(1,0,2)    
    # A_ai = A_ai.reshape(nvir,nocc,nstates).transpose(1,0,2)    
    return A_ai

def spin_flip_Amatx(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
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
    mo_vir, mo_occ = ais
    nstates = x0.shape[-1]
    nocc = mo_occ.shape[-1]
    nvir = mo_vir.shape[-1]
    if xctype == 'LDA':
        ai = numpy.einsum('pa,pi->pai',mo_vir.conj(),mo_occ,optimize=True)
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s = fxc
        # The pseudo-density is calculated!
        rho1 = numpy.einsum('pbj,jbn->pn', ai.conj(),x0.reshape(nocc,nvir,nstates), optimize=True) 
        # *2 for xx,yy parts. 
        A_ai = numpy.einsum('pai,p,pn->ain', ai, s_s*weights, rho1, optimize=True)*2.0
        # The orbital energy difference is added here
        A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    elif xctype == 'GGA':
        ai = numpy.einsum('pa,pi->pai',mo_vir[0].conj(),mo_occ[0],optimize=True)
        # \nabla 
        nabla_ai = numpy.einsum('xpa,pi->xpai',mo_vir[1:4].conj(),mo_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_vir[0].conj(),mo_occ[1:4],optimize=True)
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s, s_Ns, Ns_Ns = fxc
        ngrid = s_s.shape[-1]
        nstates = x0.shape[-1]
        # The pseudo-density is calculated!
        rho1 = numpy.zeros((4, ngrid, nstates))
        rho1[0] = numpy.einsum('pbj,jbn->pn', ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        rho1[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        A_ai = numpy.einsum('pai,p,pn->ain', ai, s_s*weights, rho1[0], optimize=True)
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai, s_Ns*weights, rho1[1:], optimize=True)
        A_ai+= numpy.einsum('xpai,xp,pn->ain', nabla_ai, s_Ns*weights, rho1[0], optimize=True)
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', nabla_ai, Ns_Ns*weights, rho1[1:], optimize=True)
        # *2 for xx,yy parts. 
        A_ai *= 2.0
        # The orbital energy difference is added here
        A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)

    elif xctype == 'MGGA':
        ai = numpy.einsum('pa,pi->pai',mo_vir[0].conj(),mo_occ[0],optimize=True)
        nabla_ai = numpy.einsum('xpa,pi->xpai',mo_vir[1:4].conj(),mo_occ[0],optimize=True) \
            + numpy.einsum('pa,xpi->xpai',mo_vir[0].conj(),mo_occ[1:4],optimize=True)
        tau_ai = 0.5*numpy.einsum('xpa,xpi->pai',mo_vir[1:4].conj(),mo_occ[1:4],optimize=True)    
        # eri_ao has hybrid factor produced!
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        s_s, s_Ns, Ns_Ns, u_u, s_u, Ns_u = fxc
        ngrid = s_s.shape[-1]
        nstates = x0.shape[-1]
        # The pseudo-density is calculated!
        rho1 = numpy.zeros((5, ngrid, nstates))
        rho1[0] = numpy.einsum('pbj,jbn->pn', ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        rho1[1:4] = numpy.einsum('xpbj,jbn->xpn', nabla_ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        rho1[4] = numpy.einsum('pbj,jbn->pn', tau_ai.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)

        A_ai = numpy.einsum('pai,p,pn->ain', ai, s_s*weights, rho1[0], optimize=True)
        A_ai+= numpy.einsum('xpai,xp,pn->ain', nabla_ai, s_Ns*weights, rho1[0], optimize=True)
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai, s_Ns*weights, rho1[1:4], optimize=True)
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', nabla_ai, Ns_Ns*weights, rho1[1:4], optimize=True)
        A_ai+= numpy.einsum('pai,p,pn->ain', tau_ai, u_u*weights, rho1[4], optimize=True) #u_u
        A_ai+= numpy.einsum('pai,p,pn->ain', ai, s_u*weights, rho1[4], optimize=True) #s_u
        A_ai+= numpy.einsum('pai,p,pn->ain', tau_ai, s_u*weights, rho1[0], optimize=True) #u_s
        A_ai+= numpy.einsum('pai,xp,xpn->ain', tau_ai, Ns_u*weights, rho1[1:4], optimize=True) #u_Ns
        A_ai+= numpy.einsum('xpai,xp,pn->ain', nabla_ai, Ns_u*weights, rho1[4], optimize=True) #u_Ns
        A_ai *= 2.0
        # The orbital energy difference is calculated here
        A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    if numpy.abs(hyb) >= 1e-10:
        Cvir,Cocc = uvs
        dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc, Cvir, optimize=True)
        eri = numpy.empty(dm1.shape)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm1[:,:,i], hermi=0)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm1[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ain',eri,Cvir,Cocc)
        A_ai -= erimo

    A_ai = A_ai.transpose(1,0,2)    
    
    # A_ai = A_ai.reshape(nvir,nocc,nstates).transpose(1,0,2)    
    return A_ai

def A_ai_spin_conserving(xctype,x0,mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ,fxc,weights):
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    nov_a = nocc_a*nvir_a
    nstates = x0.shape[-1]
    x0_aa = x0[:nov_a]
    x0_bb = x0[nov_a:]
    
    if xctype == 'LDA':
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)

        n_n,n_s,s_s = fxc
        
        # The pseudo-density is calculated!
        rho1_aa = numpy.einsum('pbj,jbn->pn', ai_aa.conj(),x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pbj,jbn->pn', ai_bb.conj(),x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # import pdb
        # pdb.set_trace()
        A_ai_top = 0.0
        A_ai_bom = 0.0
        
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_n*weights, rho1_aa, optimize=True) # n_n
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_s*weights, rho1_aa, optimize=True) # n_s
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_s*weights, rho1_aa, optimize=True) # s_n
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, s_s*weights, rho1_aa, optimize=True) # s_s
        
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_n*weights, rho1_bb, optimize=True) # n_n
        A_ai_top += -1*numpy.einsum('pai,p,pn->ian', ai_aa, n_s*weights, rho1_bb, optimize=True) # n_s
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_s*weights, rho1_bb, optimize=True) # s_n
        A_ai_top += -1*numpy.einsum('pai,p,pn->ian', ai_aa, s_s*weights, rho1_bb, optimize=True) # s_s
        
        A_ai_bom += numpy.einsum('pai,p,pn->ian', ai_bb, n_n*weights, rho1_aa, optimize=True) # n_n
        A_ai_bom += numpy.einsum('pai,p,pn->ian', ai_bb, n_s*weights, rho1_aa, optimize=True) # n_s
        A_ai_bom += -1*numpy.einsum('pai,p,pn->ian', ai_bb, n_s*weights, rho1_aa, optimize=True) # s_n
        A_ai_bom += -1*numpy.einsum('pai,p,pn->ian', ai_bb, s_s*weights, rho1_aa, optimize=True) # s_s
        
        A_ai_bom += numpy.einsum('pai,p,pn->ian', ai_bb, n_n*weights, rho1_bb, optimize=True) # n_n
        A_ai_bom += -1*numpy.einsum('pai,p,pn->ian', ai_bb, n_s*weights, rho1_bb, optimize=True) # n_s
        A_ai_bom += -1*numpy.einsum('pai,p,pn->ian', ai_bb, n_s*weights, rho1_bb, optimize=True) # s_n
        A_ai_bom += numpy.einsum('pai,p,pn->ian', ai_bb, s_s*weights, rho1_bb, optimize=True) # s_s
        
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
        # The pseudo-density is calculated!
        rho1_aa = numpy.zeros((4, ngrid, nstates))
        rho1_bb = numpy.zeros((4, ngrid, nstates))
        rho1_aa[0] = numpy.einsum('pbj,jbn->pn', ai_aa.conj(), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_aa[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_aa.conj(), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_bb[0] = numpy.einsum('pbj,jbn->pn', ai_bb.conj(), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        rho1_bb[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_bb.conj(), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        
        A_ai_top = 0.0
        A_ai_bom = 0.0
        
        # nn
        A_ai_top += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ai_top += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # ns
        A_ai_top    += numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ai_top += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom    += numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # sn
        A_ai_top    += numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ai_top    += numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
 
        # n_Nn
        A_ai_top += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_n
        A_ai_top += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ai_top += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # n_Ns
        A_ai_top      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_n
        A_ai_top      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ai_top      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # ss
        A_ai_top    += numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ai_top += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom    += numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb[0],optimize=True)

        # s_Nn
        A_ai_top    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_s
        A_ai_top    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ai_top += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Ns
        A_ai_top    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_s
        A_ai_top    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ai_top += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # Nn_Nn part
        A_ai_top += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Nn_Ns part
        A_ai_top    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # import pdb
        # pdb.set_trace()
        # Ns_Nn part
        A_ai_top    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Ns_Ns part
        A_ai_top    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
    return A_ai_top,A_ai_bom

def spin_conserving_Amatx_parallel(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,ncpu):
    e_ia_aa,e_ia_bb = e_ia
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    ngrid = weights.shape[-1]
    nstates = x0.shape[-1]
    nov_a = nocc_a*nvir_a
    nstates = x0.shape[-1]
    x0_aa = x0[:nov_a]
    x0_bb = x0[nov_a:]
    
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
            para_results.append(pool.apply_async(A_ai_spin_conserving,(xctype, x0, 
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
            para_results.append(pool.apply_async(A_ai_spin_conserving,(xctype,x0,
                                mo_a_vir[:,idxi:idxf], mo_a_occ[:,idxi:idxf], 
                                mo_b_vir[:,idxi:idxf], mo_b_occ[:,idxi:idxf],
                                fxc_para, weights[idxi:idxf])))
            
        pool.close()
        pool.join()

    elif xctype == 'MGGA':
        raise NotImplementedError("Spin-conserved scheme isn't implemented in Meta-GGA")
    
    A_ai_top = 0.0
    A_ai_bom = 0.0
    
    for result_para in para_results:
        result = result_para.get()
        A_ai_top += result[0]
        A_ai_bom += result[1]
    
    # The orbital energy difference is added here
    A_ai_top += numpy.einsum('ia,ian->ian', e_ia_aa.reshape(nocc_a,nvir_a), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
    A_ai_bom += numpy.einsum('ia,ian->ian', e_ia_bb.reshape(nocc_b,nvir_b), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
    # import pdb
    # pdb.set_trace()
    Ca_vir,Ca_occ,Cb_vir,Cb_occ = uvs
    # The hartree potential term.
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    A_ai_top += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    A_ai_top += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    A_ai_bom += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    A_ai_bom += erimo
    
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
        A_ai_top -= erimo
        
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
        A_ai_bom -= erimo

    A_ai_top = A_ai_top.reshape(-1,nstates)
    A_ai_bom = A_ai_bom.reshape(-1,nstates)
    
    A_ai = numpy.concatenate([A_ai_top,A_ai_bom],axis=0)
        
    return A_ai


def spin_conserving_Amatx(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
    e_ia_aa,e_ia_bb = e_ia
    mo_a_vir, mo_a_occ, mo_b_vir, mo_b_occ = ais
    
    nstates = x0.shape[-1]
    nocc_a = mo_a_occ.shape[-1]
    nvir_a = mo_a_vir.shape[-1]
    nocc_b = mo_b_occ.shape[-1]
    nvir_b = mo_b_vir.shape[-1]
    
    nov_a = nocc_a*nvir_a
    
    if xctype == 'LDA':
        ai_aa = numpy.einsum('pa,pi->pai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('pa,pi->pai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        # import pdb
        # pdb.set_trace()
        fxc, hyec = kernel
        omega, alpha,hyb = hyec
        n_n,n_s,s_s = fxc
        
        # import pdb
        # pdb.set_trace()
        x0_aa = x0[:nov_a]
        x0_bb = x0[nov_a:]
        
        # The pseudo-density is calculated!
        rho1_aa = numpy.einsum('pbj,jbn->pn', ai_aa.conj(),x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True) 
        rho1_bb = numpy.einsum('pbj,jbn->pn', ai_bb.conj(),x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True) 
        
        # import pdb
        # pdb.set_trace()
        A_ai_top = 0.0
        A_ai_bom = 0.0
        
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_n*weights, rho1_aa, optimize=True) # n_n
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_s*weights, rho1_aa, optimize=True) # n_s
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_s*weights, rho1_aa, optimize=True) # s_n
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, s_s*weights, rho1_aa, optimize=True) # s_s
        
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_n*weights, rho1_bb, optimize=True) # n_n
        A_ai_top += -1*numpy.einsum('pai,p,pn->ian', ai_aa, n_s*weights, rho1_bb, optimize=True) # n_s
        A_ai_top += numpy.einsum('pai,p,pn->ian', ai_aa, n_s*weights, rho1_bb, optimize=True) # s_n
        A_ai_top += -1*numpy.einsum('pai,p,pn->ian', ai_aa, s_s*weights, rho1_bb, optimize=True) # s_s
        
        A_ai_bom += numpy.einsum('pai,p,pn->ian', ai_bb, n_n*weights, rho1_aa, optimize=True) # n_n
        A_ai_bom += numpy.einsum('pai,p,pn->ian', ai_bb, n_s*weights, rho1_aa, optimize=True) # n_s
        A_ai_bom += -1*numpy.einsum('pai,p,pn->ian', ai_bb, n_s*weights, rho1_aa, optimize=True) # s_n
        A_ai_bom += -1*numpy.einsum('pai,p,pn->ian', ai_bb, s_s*weights, rho1_aa, optimize=True) # s_s
        
        A_ai_bom += numpy.einsum('pai,p,pn->ian', ai_bb, n_n*weights, rho1_bb, optimize=True) # n_n
        A_ai_bom += -1*numpy.einsum('pai,p,pn->ian', ai_bb, n_s*weights, rho1_bb, optimize=True) # n_s
        A_ai_bom += -1*numpy.einsum('pai,p,pn->ian', ai_bb, n_s*weights, rho1_bb, optimize=True) # s_n
        A_ai_bom += numpy.einsum('pai,p,pn->ian', ai_bb, s_s*weights, rho1_bb, optimize=True) # s_s
        
        
        # The orbital energy difference is added here
        A_ai_top += numpy.einsum('ia,ian->ian', e_ia_aa.reshape(nocc_a,nvir_a), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        A_ai_bom += numpy.einsum('ia,ian->ian', e_ia_bb.reshape(nocc_b,nvir_b), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        
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
        nstates = x0.shape[-1]
        x0_aa = x0[:nov_a]
        x0_bb = x0[nov_a:]
        
        # The pseudo-density is calculated!
        rho1_aa = numpy.zeros((4, ngrid, nstates))
        rho1_bb = numpy.zeros((4, ngrid, nstates))
        rho1_aa[0] = numpy.einsum('pbj,jbn->pn', ai_aa.conj(), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_aa[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_aa.conj(), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        rho1_bb[0] = numpy.einsum('pbj,jbn->pn', ai_bb.conj(), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        rho1_bb[1:] = numpy.einsum('xpbj,jbn->xpn', nabla_ai_bb.conj(), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)
        
        A_ai_top = 0.0
        A_ai_bom = 0.0
        
        # nn
        A_ai_top += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ai_top += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += numpy.einsum('p,pai,pn->ian',n_n*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # ns
        A_ai_top    += numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ai_top += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom    += numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
        
        # sn
        A_ai_top    += numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ai_top    += numpy.einsum('p,pai,pn->ian',n_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('p,pai,pn->ian',n_s*weights,ai_bb,rho1_bb[0],optimize=True)
 
        # n_Nn
        A_ai_top += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += numpy.einsum('xp,pai,xpn->ian',n_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_n
        A_ai_top += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ai_top += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += numpy.einsum('xp,xpai,pn->ian',n_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # n_Ns
        A_ai_top      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom      += numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += -1.0*numpy.einsum('xp,pai,xpn->ian',n_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_n
        A_ai_top      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ai_top      += numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += -1.0*numpy.einsum('xp,xpai,pn->ian',n_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # ss
        A_ai_top    += numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_aa[0],optimize=True)  
        A_ai_top += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom    += numpy.einsum('p,pai,pn->ian',s_s*weights,ai_bb,rho1_bb[0],optimize=True)

        # s_Nn
        A_ai_top    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top    += numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,pai,xpn->ian',s_Nn*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Nn_s
        A_ai_top    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ai_top += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom    += numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,xpai,pn->ian',s_Nn*weights,nabla_ai_bb,rho1_bb[0],optimize=True)

        # s_Ns
        A_ai_top    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom    += numpy.einsum('xp,pai,xpn->ian',s_Ns*weights,ai_bb,rho1_bb[1:4],optimize=True)
        
        # Ns_s
        A_ai_top    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_aa[0],optimize=True)
        A_ai_top += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_aa,rho1_bb[0],optimize=True)
        A_ai_bom += -1*numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_aa[0],optimize=True)
        A_ai_bom    += numpy.einsum('xp,xpai,pn->ian',s_Ns*weights,nabla_ai_bb,rho1_bb[0],optimize=True)
        
        # Nn_Nn part
        A_ai_top += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += numpy.einsum('xyp,xpai,ypn->ian',Nn_Nn*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Nn_Ns part
        A_ai_top    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # import pdb
        # pdb.set_trace()
        # Ns_Nn part
        A_ai_top    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top    += numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xyp,xpai,ypn->ian',Nn_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)

        # Ns_Ns part
        A_ai_top    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_aa[1:4],optimize=True)
        A_ai_top += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_aa,rho1_bb[1:4],optimize=True)
        A_ai_bom += -1*numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_aa[1:4],optimize=True)
        A_ai_bom    += numpy.einsum('xyp,xpai,ypn->ian',Ns_Ns*weights,nabla_ai_bb,rho1_bb[1:4],optimize=True)
        
        # The orbital energy difference is added here
        A_ai_top += numpy.einsum('ia,ian->ian', e_ia_aa.reshape(nocc_a,nvir_a), x0_aa.reshape(nocc_a,nvir_a,nstates), optimize=True)
        A_ai_bom += numpy.einsum('ia,ian->ian', e_ia_bb.reshape(nocc_b,nvir_b), x0_bb.reshape(nocc_b,nvir_b,nstates), optimize=True)

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
    A_ai_top += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, Ca_vir.conj(), Ca_occ, optimize=True)
    A_ai_top += erimo
    
    dm1 = numpy.einsum('ian,vi,ua->uvn', x0_aa.reshape(nocc_a,nvir_a,nstates), Ca_occ.conj(), Ca_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    A_ai_bom += erimo
    
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0_bb.reshape(nocc_b,nvir_b,nstates), Cb_occ.conj(), Cb_vir, optimize=True)
    eri = numpy.zeros(dm1.shape)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ub,vj->jbn',eri, Cb_vir.conj(), Cb_occ, optimize=True)
    A_ai_bom += erimo
    
    
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
        A_ai_top -= erimo
        
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
        A_ai_bom -= erimo

    A_ai_top = A_ai_top.reshape(-1,nstates)
    A_ai_bom = A_ai_bom.reshape(-1,nstates)
    
    A_ai = numpy.concatenate([A_ai_top,A_ai_bom],axis=0)
        
    return A_ai

def A_ai_non_collinear(xctype,x0,ais,fxc):
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
        A_ai = numpy.einsum('pai,p,pn->ain', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho, n_s, M1, optimize=True) # n_s
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_s, n_s, rho1, optimize=True) # s_n
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', ai_s, s_s, M1, optimize=True) # s_s
        
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
        igrid = Nn_Nntmp.shape[-1]
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_nrho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_ns.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        A_ai = numpy.einsum('pai,p,pn->ain', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho, n_s, M1, optimize=True) # n_s
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_s, n_s, rho1, optimize=True) # s_n
        
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho, n_Nn, nrho1, optimize=True) # n_Nn
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_nrho, n_Nn, rho1, optimize=True) # Nn_n
        
        A_ai+= numpy.einsum('pai,yxp,yxpn->ain', ai_rho, n_Ns, nM1, optimize=True) # n_Ns
        A_ai+= numpy.einsum('xypai,yxp,pn->ain', ai_ns, n_Ns, rho1, optimize=True) # Ns_n
        
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', ai_s, s_s, M1, optimize=True) # s_s
        
        A_ai+= numpy.einsum('xpai,yxp,ypn->ain', ai_s, s_Nn, nrho1, optimize=True) # s_Nn
        A_ai+= numpy.einsum('ypai,yxp,xpn->ain', ai_nrho, s_Nn, M1, optimize=True) # Nn_s
        
        A_ai+= numpy.einsum('xpai,zyxp,zypn->ain', ai_s, s_Ns, nM1, optimize=True) # s_Ns
        A_ai+= numpy.einsum('xzpai,zyxp,ypn->ain', ai_ns, s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,igrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ai+= numpy.einsum('xpai,yxp,ypn->ain', ai_nrho, Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ai+= numpy.einsum('zpai,zyxp,yxpn->ain', ai_nrho, Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ai+= numpy.einsum('xzpai,zyxp,ypn->ain', ai_ns, Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,igrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ai+= numpy.einsum('wzpai,zyxwp,yxpn->ain', ai_ns, Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        
        ai_aa = ai_ab = ai_ba = ai_bb = None
        ai_na_a = ai_a_na = ai_na_b = ai_a_nb = ai_nb_a = ai_b_na = ai_nb_b = ai_b_nb = None 
        ai_nrho = ai_nMx = ai_nMy = ai_nMz = ai_ns = None
        rho1 = M1 = nrho1 = nM1 = None
        
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    
    return A_ai
        
def non_collinear_Amat_parallel(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,ncpu):
    Ca_vir, Ca_occ, Cb_vir, Cb_occ = uvs
    ngrid = weights.shape[-1]
    
    nstates = x0.shape[-1]
    nocc = Ca_occ.shape[-1]
    nvir = Ca_vir.shape[-1]
    
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
            para_results.append(pool.apply_async(A_ai_non_collinear,
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
            para_results.append(pool.apply_async(A_ai_non_collinear,
                                (xctype, x0, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    
    A_ai = 0.0
    for result_para in para_results:
        result = result_para.get()
        A_ai += result
    
    # The orbital energy difference is calculated here
    A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
    
    # import pdb
    # pdb.set_trace()
    Cvir = numpy.concatenate((Ca_vir, Cb_vir),axis=0)
    Cocc = numpy.concatenate((Ca_occ, Cb_occ),axis=0)
    # The hartree potential term.
    # C is real while eri not.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc.conj(), Cvir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ain',eri, Cvir.conj(), Cocc, optimize=True)
    A_ai+= erimo
    
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc.conj(), Cvir, optimize=True)
        for i in range(dm1.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri = eri.astype(numpy.complex128)
        eri *= hyb
        if abs(omega) > 1e-10:
            for i in range(dm1.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
        erimo = numpy.einsum('uvn,ua,vi->ain',eri,Cvir.conj(),Cocc, optimize=True)
        A_ai -= erimo

    A_ai = A_ai.transpose(1,0,2)    
    return A_ai

def non_collinear_Amat(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
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
        A_ai = numpy.einsum('pai,p,pn->ain', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho, n_s, M1, optimize=True) # n_s
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_s, n_s, rho1, optimize=True) # s_n
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', ai_s, s_s, M1, optimize=True) # s_s
        # The orbital energy difference is calculated here
        A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
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
        A_ai = numpy.einsum('pai,p,pn->ain', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho, n_s, M1, optimize=True) # n_s
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_s, n_s, rho1, optimize=True) # s_n
        
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho, n_Nn, nrho1, optimize=True) # n_Nn
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_nrho, n_Nn, rho1, optimize=True) # Nn_n
        
        A_ai+= numpy.einsum('pai,yxp,yxpn->ain', ai_rho, n_Ns, nM1, optimize=True) # n_Ns
        A_ai+= numpy.einsum('xypai,yxp,pn->ain', ai_ns, n_Ns, rho1, optimize=True) # Ns_n
        
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', ai_s, s_s, M1, optimize=True) # s_s
        
        A_ai+= numpy.einsum('xpai,yxp,ypn->ain', ai_s, s_Nn, nrho1, optimize=True) # s_Nn
        A_ai+= numpy.einsum('ypai,yxp,xpn->ain', ai_nrho, s_Nn, M1, optimize=True) # Nn_s
        
        A_ai+= numpy.einsum('xpai,zyxp,zypn->ain', ai_s, s_Ns, nM1, optimize=True) # s_Ns
        A_ai+= numpy.einsum('xzpai,zyxp,ypn->ain', ai_ns, s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ai+= numpy.einsum('xpai,yxp,ypn->ain', ai_nrho, Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ai+= numpy.einsum('zpai,zyxp,yxpn->ain', ai_nrho, Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ai+= numpy.einsum('xzpai,zyxp,ypn->ain', ai_ns, Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ai+= numpy.einsum('wzpai,zyxwp,yxpn->ain', ai_ns, Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        # The orbital energy difference is calculated here
        A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    elif xctype == 'MGGA':
        pass
    
    # import pdb
    # pdb.set_trace()
    Cvir = numpy.concatenate((Ca_vir, Cb_vir),axis=0)
    Cocc = numpy.concatenate((Ca_occ, Cb_occ),axis=0)
    # The hartree potential term.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), Cocc.conj(), Cvir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] = mf.get_j(mf.mol, dm1[:,:,i], hermi=0)
    erimo = numpy.einsum('uvn,ua,vi->ain',eri, Cvir.conj(), Cocc, optimize=True)
    A_ai+= erimo
    
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
        erimo = numpy.einsum('uvn,ua,vi->ain',eri,Cvir.conj(),Cocc, optimize=True)
        A_ai -= erimo

    A_ai = A_ai.transpose(1,0,2)    
    return A_ai

def A_ai_non_collinear_r(xctype,x0,ais,fxc):
    mo_vir_L,mo_vir_S,mo_occ_L,mo_occ_S = ais
    nstates = x0.shape[-1]
    nocc = mo_occ_L.shape[-1]
    nvir = mo_vir_L.shape[-1]
    
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
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        A_ai = numpy.einsum('pai,p,pn->ain', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho, n_s, M1, optimize=True) # n_s
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_s, n_s, rho1, optimize=True) # s_n
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', ai_s, s_s, M1, optimize=True) # s_s
        
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
        A_ai = numpy.einsum('pai,p,pn->ain', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho[0], n_s, M1, optimize=True) # n_s
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        A_ai+= numpy.einsum('pai,yxp,yxpn->ain', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        A_ai+= numpy.einsum('xypai,yxp,pn->ain', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        A_ai+= numpy.einsum('xpai,yxp,ypn->ain', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        A_ai+= numpy.einsum('ypai,yxp,xpn->ain', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        A_ai+= numpy.einsum('xpai,zyxp,zypn->ain', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        A_ai+= numpy.einsum('xzpai,zyxp,ypn->ain', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ai+= numpy.einsum('xpai,yxp,ypn->ain', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ai+= numpy.einsum('zpai,zyxp,yxpn->ain', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ai+= numpy.einsum('xzpai,zyxp,ypn->ain', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ai+= numpy.einsum('wzpai,zyxwp,yxpn->ain', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
    return A_ai 

def non_collinear_Amat_r_parallel(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,ncpu):
    fxc,hyec = kernel 

    nstates = x0.shape[-1]
    C_vir, C_occ = uvs
    nocc = C_occ.shape[-1]
    nvir = C_vir.shape[-1]
    
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
            para_results.append(pool.apply_async(A_ai_non_collinear_r,
                                (xctype, x0, ais_para, fxc_para)))
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
            para_results.append(pool.apply_async(A_ai_non_collinear_r,
                                (xctype, x0, ais_para, fxc_para)))
        pool.close()
        pool.join()
    
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
    # import pdb
    # pdb.set_trace()
    
    A_ai = 0.0
    for result_para in para_results:
        result = result_para.get()
        A_ai += result

    # The orbital energy difference is calculated here
    A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
    
    # n2c = C_vir.shape[0]//2
    # The hartree potential term.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
     
    erimo = numpy.einsum('uvn,ua,vi->ain',eri, C_vir.conj(), C_occ, optimize=True)
    A_ai += erimo
    # print(erimo[:,:,0])
    
    # Approach 2
    # n2c = C_vir.shape[0]//2
    # # C_vir[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    # # C_occ[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    
    # eri_LL = mf.mol.intor('int2e_spinor')
    # eri_LS = mf.mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2
    # eri_SS = mf.mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4
    # # # # # # transform the eri to mo space.
    # # n2c = C_occ.shape[0]//2
    
    # eri_LL_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LL,
    #                       C_vir[:n2c],C_occ[:n2c].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_LS_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LS,
    #                       C_vir[n2c:],C_occ[n2c:].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_SL_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LS.transpose(2,3,0,1),
    #                       C_vir[:n2c],C_occ[:n2c].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_SS_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_SS,
    #                       C_vir[n2c:],C_occ[n2c:].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    
    # A_ai += numpy.einsum('uvn,ua,vi->ain', eri_LL_ao,
    #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
    
    # A_ai += numpy.einsum('uvn,ua,vi->ain', eri_LS_ao,
    #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
    
    # A_ai += numpy.einsum('uvn,ua,vi->ain', eri_SL_ao,
    #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
    
    # A_ai += numpy.einsum('uvn,ua,vi->ain', eri_SS_ao,
    #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True)   
    
    # The excat exchange is calculated
    # import pdb
    # pdb.set_trace()
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
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
                
        erimo = numpy.einsum('uvn,ua,vi->ain',eri,C_vir.conj(),C_occ, optimize=True)
        A_ai -= erimo

    A_ai = A_ai.transpose(1,0,2)    
    return A_ai

def non_collinear_Amat_r(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
    fxc,hyec = kernel 
    mo_vir_L, mo_vir_S, mo_occ_L , mo_occ_S = ais
    nstates = x0.shape[-1]
    C_vir, C_occ = uvs
    nocc = C_occ.shape[-1]
    nvir = C_vir.shape[-1]
    
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
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        A_ai = numpy.einsum('pai,p,pn->ain', ai_rho, n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho, n_s, M1, optimize=True) # n_s
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_s, n_s, rho1, optimize=True) # s_n
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', ai_s, s_s, M1, optimize=True) # s_s
        # The orbital energy difference is calculated here
        A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
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
        A_ai = numpy.einsum('pai,p,pn->ain', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho[0], n_s, M1, optimize=True) # n_s
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        A_ai+= numpy.einsum('pai,xp,xpn->ain', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        A_ai+= numpy.einsum('xpai,xp,pn->ain', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        A_ai+= numpy.einsum('pai,yxp,yxpn->ain', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        A_ai+= numpy.einsum('xypai,yxp,pn->ain', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        A_ai+= numpy.einsum('xpai,xyp,ypn->ain', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        A_ai+= numpy.einsum('xpai,yxp,ypn->ain', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        A_ai+= numpy.einsum('ypai,yxp,xpn->ain', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        A_ai+= numpy.einsum('xpai,zyxp,zypn->ain', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        A_ai+= numpy.einsum('xzpai,zyxp,ypn->ain', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ai+= numpy.einsum('xpai,yxp,ypn->ain', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ai+= numpy.einsum('zpai,zyxp,yxpn->ain', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ai+= numpy.einsum('xzpai,zyxp,ypn->ain', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ai+= numpy.einsum('wzpai,zyxwp,yxpn->ain', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        # The orbital energy difference is calculated here
        A_ai+= numpy.einsum('ia,ian->ain', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)    
        
    else:
        raise NotImplementedError("Only LDA is implemented.")    
    
    # import pdb
    # pdb.set_trace()
    
    
    # n2c = C_vir.shape[0]//2
    # The hartree potential term.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
     
    erimo = numpy.einsum('uvn,ua,vi->ain',eri, C_vir.conj(), C_occ, optimize=True)
    A_ai += erimo
    # print(erimo[:,:,0])
    
    # Approach 2
    # n2c = C_vir.shape[0]//2
    # # C_vir[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    # # C_occ[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    
    # eri_LL = mf.mol.intor('int2e_spinor')
    # eri_LS = mf.mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2
    # eri_SS = mf.mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4
    # # # # # # transform the eri to mo space.
    # # n2c = C_occ.shape[0]//2
    
    # eri_LL_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LL,
    #                       C_vir[:n2c],C_occ[:n2c].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_LS_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LS,
    #                       C_vir[n2c:],C_occ[n2c:].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_SL_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LS.transpose(2,3,0,1),
    #                       C_vir[:n2c],C_occ[:n2c].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_SS_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_SS,
    #                       C_vir[n2c:],C_occ[n2c:].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    
    # A_ai += numpy.einsum('uvn,ua,vi->ain', eri_LL_ao,
    #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
    
    # A_ai += numpy.einsum('uvn,ua,vi->ain', eri_LS_ao,
    #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
    
    # A_ai += numpy.einsum('uvn,ua,vi->ain', eri_SL_ao,
    #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
    
    # A_ai += numpy.einsum('uvn,ua,vi->ain', eri_SS_ao,
    #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True)   
    
    # The excat exchange is calculated
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
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
                
        erimo = numpy.einsum('uvn,ua,vi->ain',eri,C_vir.conj(),C_occ, optimize=True)
        A_ai -= erimo

    A_ai = A_ai.transpose(1,0,2)    
    return A_ai

def get_Dinv_and_x1_iparts(omega,D,x1):
    if omega.shape[0]>1:
        for i in range(omega.shape[0]):
            Dinv = numpy.linalg.inv(numpy.diag(omega[i]-D+1e-99))
            x1[:,i] = numpy.einsum('ts,s->t',Dinv,x1[:,i], optimize = True)
    else:
        Dinv = numpy.linalg.inv(numpy.diag(omega-D+1e-99))
        # print(Dinv.shape)
        # print(x1.shape)
        x1[:,0] = numpy.einsum('ts,s->t',Dinv,x1[:,0], optimize = True)
    return x1
            
def get_x1_new(omega,D,x1,ncpu=None):
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    ngrid = x1.shape[-1]
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    print(omega.shape[0])
    print(x1.shape[0])
    for para in NX_list:
        idxi,idxf = para
        para_results.append(pool.apply_async(get_Dinv_and_x1_iparts,
                            (omega[idxi:idxf],D,x1[:,idxi:idxf])))   
    pool.close()
    pool.join()
    
    ioff = 0
    for result_para in para_results:
        idxi,idxf = NX_list[ioff]
        result = result_para.get()
        x1[:,idxi:idxf] = result
        ioff += 1
    return x1

class Solver():
    def __init__(self, mf,mf2, Extype, kernel, nstates=3, init_guess=None, scheme='LAN',
                 max_cycle = 50, conv_tol = 1.0E-8, cutoff = 8, diff = 1e-4, parallel=False, ncpu=None,Whkerl=False):
        
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
                self.Amatx = spin_conserving_Amatx_parallel
                self.get_Diagelmet_of_A = get_Diagelemt_of_A_sc_parallel
            else:
                self.Amatx = spin_conserving_Amatx
                self.get_Diagelmet_of_A = get_Diagelemt_of_A_sc
            self.get_e_ia_and_mo = get_e_ia_and_mo
        
        elif Extype == 'SPIN_FLIP_UP' or Extype == 'SPIN_FLIP_DOWN':
            if parallel:
                self.Amatx = spin_flip_Amatx_parallel
                self.get_Diagelmet_of_A = get_Diagelemt_of_A_parallel
            else:
                self.Amatx = spin_flip_Amatx
                self.get_Diagelmet_of_A = get_Diagelemt_of_A
            self.get_e_ia_and_mo = get_e_ia_and_mo
        
        elif Extype == 'GKS' or Extype == None:
            if self.mf2 is not None:
                self.get_e_ia_and_mo = get_nc_utg_e_ia_and_mo  
            else:
                self.get_e_ia_and_mo = get_nc_e_ia_and_mo 
            if parallel:
                self.Amatx = non_collinear_Amat_parallel
                self.get_Diagelmet_of_A = get_nc_Diagelemt_of_A_parallel
            else:
                self.Amatx = non_collinear_Amat
                self.get_Diagelmet_of_A = get_nc_Diagelemt_of_A
                
        elif Extype == 'DKS':
            if parallel:
                self.Amatx = non_collinear_Amat_r_parallel
                self.get_Diagelmet_of_A = get_nc_r_Diagelemt_of_A_parallel
            else:
                self.Amatx = non_collinear_Amat_r
                self.get_Diagelmet_of_A = get_nc_r_Diagelemt_of_A
            self.get_e_ia_and_mo = get_nc_r_e_ia_and_mo 
                                
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
       
        x0 = self.init_guess(self.Extype,self.nstates, e_ia)
            
        if self.Extype == 'GKS' or self.Extype == None:
            x0 = x0.astype(numpy.complex128)
        # Note that self.nstates <= nstate_new
        # Because nstate_new may contains more states that degenerated
        nstate_new = x0.shape[-1]
        # Store the pre-step eigenvalues
        omega0 = numpy.zeros((nstate_new))
        # hyec means hybrid functional coffiecients.
        # import pdb
        # pdb.set_trace()
        fxc,hyec = self.kernel
       
        # Lancov
        if self.scheme.upper() == 'LAN':
            
            for icycle in range(self.max_cycle):
                # CUTOFF length
                if icycle%self.cutoff == 0:
                    x0 = x0[:,:nstate_new]
                # Get the new vector
                # import pdb
                # pdb.set_trace()
                x1 = self.Amatx(e_ia, (fxc,hyec), x0[:,:nstate_new], xctype, weights, ais, uvs, self.mf,self.ncpu)
                x1 = x1.reshape(-1, nstate_new)
                # qr分解
                x_tot = numpy.linalg.qr(numpy.concatenate((x0, x1),axis=1))[0]
                nstate_tot = x_tot.shape[-1]
                Ax = self.Amatx(e_ia, (fxc,hyec), x_tot, xctype, weights, ais, uvs, self.mf,self.ncpu)
                # @ 矩阵乘法
                xtAx = x_tot.conj().T@Ax.reshape(-1,nstate_tot)
                omega, c_small = numpy.linalg.eigh(xtAx)
                x0 = numpy.linalg.qr(x_tot@c_small)[0]
                omega_diff = numpy.abs(omega[:nstate_new]-omega0)
                print("Difference")
                print(omega_diff)
                omega0 = omega[:nstate_new]
                print(omega[:self.nstates]*27.21138386)
                print(f"In circle {icycle}, {(omega_diff<self.conv_tol).sum()}/{nstate_new} states converged!")
                if (omega_diff<self.conv_tol).sum() == nstate_new:
                    break
            if icycle == self.max_cycle-1:
                raise RuntimeError("Not convergence!")
            
        elif self.scheme.upper() == 'DAVIDSON':
            # Compulate the diaongal element of A-matrix.
            t0 = (time.process_time(), time.time())[0]
            D = self.get_Diagelmet_of_A(e_ia, (fxc,hyec),xctype, weights, ais, uvs, self.mf,self.ncpu,Whkerl=self.Whkerl)
            t1 = (time.process_time(), time.time())[0]
            print('Calculation D_init:' + str(t1-t0) + ' seconds.')
            for icycle in range(self.max_cycle):
                # CUTOFF length
                if icycle%self.cutoff == 0:
                    x0 = x0[:,:nstate_new]
                # Project to the sub-space
                t2 = (time.process_time(), time.time())[0]
                Ax0 = self.Amatx(e_ia, (fxc,hyec), x0, xctype, weights, ais, uvs, self.mf,self.ncpu)
                t3 = (time.process_time(), time.time())[0]
                print('Calculation Ax0:' + str(t3-t2) + ' seconds.')
                x0tAx0 = x0.conj().T@Ax0.reshape(-1,x0.shape[-1])
                t4 = (time.process_time(), time.time())[0]
                print('Calculation x0tAx0:' + str(t4-t3) + ' seconds.')
                # Get the new vector
                omega, c_small = numpy.linalg.eigh(x0tAx0)
                
                # Calculate the convergence.
                omega_diff = numpy.abs(omega[:nstate_new]-omega0)
                print("Difference")
                print(omega_diff)
                omega0 = omega[:nstate_new]
                print(omega[:self.nstates]*27.21138386)
                print(f"In circle {icycle}, {(omega_diff<self.conv_tol).sum()}/{nstate_new} states converged!")
                if (omega_diff<self.conv_tol).sum() == nstate_new:
                    break
                
                t5 = (time.process_time(), time.time())[0]
                # compute the residual -> rk = omega I - A x_tot c_small
                x1 = numpy.einsum('n,tn->tn',omega,x0@c_small, optimize = True) 
                U_i = x0@c_small
                x1 -= Ax0.reshape(-1,x0.shape[-1])@c_small
                t6 = (time.process_time(), time.time())[0]
                print('Calculation x1_new:' + str(t6-t5) + ' seconds.')
                    
                for i in range(x1.shape[-1]):
                    t61 = (time.process_time(), time.time())[0]
                    Dinv = numpy.diag(1/(omega[i]-D+1e-99))
                    print('Calculation Dinv_i:' + str(t61-t6) + ' seconds.')
                    x1[:,i] = numpy.einsum('ts,s->t',Dinv,x1[:,i], optimize = True)
                    t62 = (time.process_time(), time.time())[0]
                    print('Calculation Dinv_i:' + str(t62-t61) + ' seconds.')

                t7 = (time.process_time(), time.time())[0]
                print('Calculation Dinv:' + str(t7-t6) + ' seconds.')
                
                x0 = numpy.linalg.qr(numpy.concatenate((x0, x1[:,:nstate_new]),axis=1))[0]
                t8 = (time.process_time(), time.time())[0]
                print('Calculation x0_new:' + str(t8-t7) + ' seconds.')
                print('\n')
                
            if icycle == self.max_cycle-1:
                raise RuntimeError("Not convergence!")
            
        else:
            raise NotImplementedError("Only lancov and davidson is implemented!")
        # import pdb
        # pdb.set_trace()
        return omega[:self.nstates]*27.21138386, U_i[:,:self.nstates]
    
    
        
        
            
        
        
        
        

