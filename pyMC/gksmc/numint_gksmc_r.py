#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-03-10 18:49:20
LastEditTime: 2022-04-12 09:52:50
LastEditors: Li Hao
Description: 
    Numerical integration utils for Dirac4c calculations.
FilePath: \pyMC\gksmc\numint_gksmc_r.py

 May the force be with you!
'''


import numpy
from pyscf import lib
from pyscf.dft import r_numint
from pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, BLKSIZE
from pyMC.gksmc import numint_gksmc
from pyMC.lib import Spoints

# TODO: Codes optimization
# TODO: Multidirections and Tri-directions can be mreged to a same subroutine.


def eval_ao(mol, coords, deriv=0, with_s=True, shls_slice=None,
            non0tab=None, out=None, verbose=None):
    """Getting the ao basis value on each grids.
       It should be noted that this is a overwritten of the original eval_ao subroutine.

    Args:
        mol (gto.Mole() object.  mole_symm.Mole_symm() object): mol object.
        coords (numpy.array): coordinates of the file. [ngrid,3]
        deriv (int, optional): derivative of the ao basis. Defaults to 0.
        with_s (bool, optional): whether calculating small parts. Defaults to True.
        shls_slice ([type], optional): [description]. Defaults to None.
        non0tab ([type], optional): [description]. Defaults to None.
        out ([type], optional): [description]. Defaults to None.
        verbose ([type], optional): [description]. Defaults to None.

    Returns:
        aoLa, aoLb, aoSa, aoSb (numpy.array): ao basis of the shape [comp,ngrid,n2c]
    """
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    # Calaulating the Large parts.
    # For L part, it can calculate up to
    aoLa, aoLb = mol.eval_gto(feval, coords, comp, shls_slice, non0tab, out=out)
    if with_s:
        # * Note that, the lib has been recompiled
        # * How to recompile can check the github issue #955, asked by hoshishin.
        # * Now, GTOval_ipipsp_spinor nabla nabla sigma dot p |AO> (spinor basis),
        # *     is supported.
        assert(deriv <= 2)  # only GTOval_ipsp_spinor
        ngrid, n2c = aoLa.shape[-2:]
        if out is not None:
            aoSa = numpy.empty((comp,n2c,ngrid), dtype=numpy.complex128)
            aoSb = numpy.empty((comp,n2c,ngrid), dtype=numpy.complex128)
        else:
            out = numpy.ndarray((4,comp,n2c,ngrid), dtype=numpy.complex128, buffer=out)
            aoSa, aoSb = out[2:]
        # calculate ao 0th order value.
        comp = 1
        ao = mol.eval_gto('GTOval_sp_spinor', coords, comp, shls_slice, non0tab)
        aoSa[0] = ao[0].T
        aoSb[0] = ao[1].T
        # list contains all the implemented evaluator for kinectic balanced small component
        # kinectic balanced small component basis.
        fevals = ['GTOval_sp_spinor', 'GTOval_ipsp_spinor', 'GTOval_ipipsp_spinor']
        # ! NOTE GTOval_sp_spinor, GTOval_ipsp_spinor, GTOval_ipipsp_spinor
        # ! have different dimenstion and size, and different from large component.
        # GTOval_sp_spinor is (2, ngrid, nspinor) which 2 is for large_alpha and large_beta
        # GTOval_ipsp_spinor is (2 ,3 , ngrid, nspinor), where 3 is for nabla_x, nabla_y, nabla_z
        # GTOval_ipipsp_spinor is (2 ,9 , ngrid, nspinor),
        # TODO: CONFIRM on github 9 means xx, xy, xz, yx, yy, yz, zx, zy, zz 
        # TODO: and no factor has been producted.
        p1 = 1
        for n in range(1, deriv+1):
            comp = (n+1)*(n+2)//2
            if n==1:
                ao = mol.eval_gto(fevals[n], coords, comp, shls_slice, non0tab)
            elif n==2:
                # Note: this is due to the inconsistent defination of the GTOval_ipipsp_spinor
                ao = mol.eval_gto(fevals[n], coords, comp+3, shls_slice, non0tab) 
            p0, p1 = p1, p1 + comp
            if n == 1:
                aoSa[p0:p1] = ao[0].transpose(0,2,1)
                aoSb[p0:p1] = ao[1].transpose(0,2,1)
            elif n == 2:
                # Small alpha
                aoSa[p0  ] = ao[0,0].T              #xx
                aoSa[p0+1] = ao[0,1].T + ao[0,3].T  #xy
                aoSa[p0+2] = ao[0,2].T + ao[0,6].T  #xz
                aoSa[p0+3] = ao[0,4].T              #yy
                aoSa[p0+4] = ao[0,5].T + ao[0,7].T  #yz
                aoSa[p0+5] = ao[0,8].T              #zz
                # Small beta
                aoSb[p0  ] = ao[1,0].T              #xx
                aoSb[p0+1] = ao[1,1].T + ao[1,3].T  #xy
                aoSb[p0+2] = ao[1,2].T + ao[1,6].T  #xz
                aoSb[p0+3] = ao[1,4].T              #yy
                aoSb[p0+4] = ao[1,5].T + ao[1,7].T  #yz
                aoSb[p0+5] = ao[1,8].T              #zz
        aoSa = aoSa.transpose(0,2,1)
        aoSb = aoSb.transpose(0,2,1)
        if deriv == 0:
            aoSa = aoSa[0]
            aoSb = aoSb[0]
        return aoLa, aoLb, aoSa, aoSb
    else:
        return aoLa, aoLb


def _vxc2x2_to_mat_MD_LDA_opt_MC(mol, ao, weight, vrho, non0tab, shls_slice, ao_loc, ctrl):
    """ vxc2x2 matrix for LL or SS part.
        It should be noted that , \beta \Sigma is the operator, thus ,-1 should be
        producted after this subroutine, if SS part is Used.

    Args:
        mol (gto type): molecule calculated
        ao (tuple of numpy arrays): (aoa, aob); aoa is a numpy array of [4,ngrid,n2c] for alpha component for 
            Large component or Small component, aob is beta component.
        weight (numpy array [ngrid]): weights
        vrho (numpy array [ngrid,2]): [ngrid:,0] is vrho, [ngrid:,1] is vs
        NX (numpy array): Priciple direction [3,ngrid]
        
        non0tab ([type]): [description]
        shls_slice ([type]): [description]
        ao_loc ([type]): [description]
        
        return mat(numpy.array) 2c vxc matrix in ao basis, which is the [n2c,n2c].
    """
    aoa, aob = ao
    vrho, vs = vrho
    # NOTE: vrho[3,ngrid]
    aow = numpy.empty_like(aoa)
    # This can directly product.
    vrhow = vrho*weight
    vsw = vs*weight
    if ctrl == 'LL':
        # x part
        aow = numpy.einsum('pi,p->pi',aoa,vsw[0], out=aow)
        tmp = _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
        mat = tmp + tmp.T.conj()
        # y part
        aow = numpy.einsum('pi,p->pi',aoa,vsw[1], out=aow)
        tmp = _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
        mat+= (tmp - tmp.T.conj())*1.0j
        # z part + rho part 1
        aow = numpy.einsum('pi,p->pi',aoa,vrhow, out=aow)
        aow+= numpy.einsum('pi,p->pi',aoa,vsw[2])
        mat+= _dot_ao_ao(mol, aoa, aow, non0tab, shls_slice, ao_loc)
        # z part + rho part 2
        aow = numpy.einsum('pi,p->pi',aob,vrhow, out=aow)
        aow-= numpy.einsum('pi,p->pi',aob,vsw[2])
        mat+= _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
    elif ctrl == 'SS':
        # x part
        aow = numpy.einsum('pi,p->pi',aoa,vsw[0], out=aow)
        tmp = _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
        mat = -tmp - tmp.T.conj()
        # y part
        aow = numpy.einsum('pi,p->pi',aoa,vsw[1], out=aow)
        tmp = _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
        mat+= - (tmp - tmp.T.conj())*1.0j
        # z part + rho part 1
        aow = numpy.einsum('pi,p->pi',aoa,vrhow, out=aow)
        aow-= numpy.einsum('pi,p->pi',aoa,vsw[2])
        mat+= _dot_ao_ao(mol, aoa, aow, non0tab, shls_slice, ao_loc)
        # z part + rho part 2
        aow = numpy.einsum('pi,p->pi',aob,vrhow, out=aow)
        aow+= numpy.einsum('pi,p->pi',aob,vsw[2])
        mat+= _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
    
    return mat


def _vxc2x2_to_mat_MD_GGA_opt(mol, ao, vrho, non0tab, shls_slice, ao_loc, ctrl):
    """Generate the Vxc matrix for GGA.

    Args:
        mol (gto type): molecule calculated
        ao (tuple of numpy arrays): (aoa, aob); aoa is a numpy array of [4,ngrid,n2c] for alpha component for 
            Large component or Small component, aob is beta component.
        weight (numpy array [ngrid]): weights
        vrho (numpy array [ngrid,2]): [ngrid:,0] is vrho, [ngrid:,1] is vs
        NX (numpy array): Priciple direction [3,ngrid]
        
        non0tab ([type]): [description]
        shls_slice ([type]): [description]
        ao_loc ([type]): [description]
        
        return mat(numpy.array) 2c vxc matrix in ao basis, which is the [n2c,n2c].
    """
    # ! NOTE: double counting should be paid special attention to.
    # ! eg. \frac{\partial \epsilon}{\partial s} N_{I,x} \left( \mu_{b}^{*} \nu_a + \mu_{a}^{*} \nu_b \right)
    # !     will be double count, for to calculating nabla, tmp + tmp.T.conj() will certainly be called.
    # !     Thus, double counting accured, but the 0.5 has been taken in numint_gksmc._uks_gga_wv0
    aoa, aob = ao
    wvrho, wvs = vrho
    aow = numpy.empty_like(aoa[0])
    
    if ctrl == 'LL':
        # x part
        aow = numpy.einsum('npi,np->pi',aoa,wvs[0], out=aow)
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        aow = numpy.einsum('npi,np->pi',aob,wvs[0], out=aow)
        # ! Note, the following line doesn't do the double-counting part.
        tmp+= _dot_ao_ao(mol, aoa[0], aow, non0tab, shls_slice, ao_loc)
        mat = tmp + tmp.T.conj() # ! Note, this line do the double-counting part.
        # y part
        aow = numpy.einsum('npi,np->pi',aoa,wvs[1], out=aow)
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        aow = numpy.einsum('npi,np->pi',aob,wvs[1], out=aow)
        tmp-= _dot_ao_ao(mol, aoa[0], aow, non0tab, shls_slice, ao_loc)
        mat+= (tmp - tmp.T.conj())*1.0j
        # z part + rho part 1
        aow = numpy.einsum('npi,np->pi',aoa,wvrho, out=aow)
        aow+= numpy.einsum('npi,np->pi',aoa,wvs[2])
        tmp = _dot_ao_ao(mol, aoa[0], aow, non0tab, shls_slice, ao_loc)
        mat+= tmp + tmp.T.conj()
        # z part + rho part 2
        aow = numpy.einsum('npi,np->pi',aob,wvrho, out=aow)
        aow-= numpy.einsum('npi,np->pi',aob,wvs[2])
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        mat+= tmp + tmp.T.conj()
    elif ctrl == 'SS':
        # x part
        aow = numpy.einsum('npi,np->pi',aoa,wvs[0], out=aow)
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        aow = numpy.einsum('npi,np->pi',aob,wvs[0], out=aow)
        tmp+= _dot_ao_ao(mol, aoa[0], aow, non0tab, shls_slice, ao_loc)
        mat = -tmp - tmp.T.conj()
        # y part
        aow = numpy.einsum('npi,np->pi',aoa,wvs[1], out=aow)
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        aow = numpy.einsum('npi,np->pi',aob,wvs[1], out=aow)
        tmp-= _dot_ao_ao(mol, aoa[0], aow, non0tab, shls_slice, ao_loc)
        mat+= - (tmp - tmp.T.conj())*1.0j
        # z part + rho part 1
        aow = numpy.einsum('npi,np->pi',aoa,wvrho, out=aow)
        aow-= numpy.einsum('npi,np->pi',aoa,wvs[2])
        tmp = _dot_ao_ao(mol, aoa[0], aow, non0tab, shls_slice, ao_loc)
        mat+= tmp + tmp.T.conj()
        # z part + rho part 2
        aow = numpy.einsum('npi,np->pi',aob,wvrho, out=aow)
        aow+= numpy.einsum('npi,np->pi',aob,wvs[2])
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        mat+= tmp + tmp.T.conj()
    
    return mat


def _vxc2x2_to_mat_MD_GGA_ibp_opt(mol, ao, weight, vrho,non0tab, shls_slice, ao_loc, ctrl):
    """ vxc2x2 matrix for LL or SS part.
        It should be noted that , \beta \Sigma is the operator, thus ,-1 should be
        producted after this subroutine, if SS part is Used.

    Args:
        mol (gto type): molecule calculated
        ao (tuple of numpy arrays): (aoa, aob); aoa is a numpy array of [4,ngrid,n2c] for alpha component for 
            Large component or Small component, aob is beta component.
        weight (numpy array [ngrid]): weights
        vrho (tuple of (vrho, vs)): [ngrid:,0] is vrho, [ngrid:,1] is vs
        NX (numpy array): Priciple direction [3,ngrid]
        
        non0tab ([type]): [description]
        shls_slice ([type]): [description]
        ao_loc ([type]): [description]
        
        return mat(numpy.array) 2c vxc matrix in ao basis, which is the [n2c,n2c].
    """
    # ! NOTE that, there is no double counting problem.
    aoa, aob = ao
    vrho, vs = vrho
    aow = numpy.empty_like(aoa[0])
    
    vrhow = vrho*weight
    vsw = vs*weight
    if ctrl == 'LL':
        # x part
        aow = numpy.einsum('pi,p->pi',aoa[0],vsw[0], out=aow)
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        mat = tmp + tmp.T.conj()
        # y part
        aow = numpy.einsum('pi,p->pi',aoa[0],vsw[1], out=aow)
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        mat+= (tmp - tmp.T.conj())*1.0j
        # z part + rho part 1
        aow = numpy.einsum('pi,p->pi',aoa[0],vrhow, out=aow)
        aow+= numpy.einsum('pi,p->pi',aoa[0],vsw[2])
        mat+= _dot_ao_ao(mol, aoa[0], aow, non0tab, shls_slice, ao_loc)
        # z part + rho part 2
        aow = numpy.einsum('pi,p->pi',aob[0],vrhow, out=aow)
        aow-= numpy.einsum('pi,p->pi',aob[0],vsw[2])
        mat+= _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
    elif ctrl == 'SS':
        # x part
        aow = numpy.einsum('pi,p->pi',aoa[0],vsw[0], out=aow)
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        mat = -tmp - tmp.T.conj()
        # y part
        aow = numpy.einsum('pi,p->pi',aoa[0],vsw[1], out=aow)
        tmp = _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
        mat+= - (tmp - tmp.T.conj())*1.0j
        # z part + rho part 1
        aow = numpy.einsum('pi,p->pi',aoa[0],vrhow, out=aow)
        aow-= numpy.einsum('pi,p->pi',aoa[0],vsw[2])
        mat+= _dot_ao_ao(mol, aoa[0], aow, non0tab, shls_slice, ao_loc)
        # z part + rho part 2
        aow = numpy.einsum('pi,p->pi',aob[0],vrhow, out=aow)
        aow+= numpy.einsum('pi,p->pi',aob[0],vsw[2])
        mat+= _dot_ao_ao(mol, aob[0], aow, non0tab, shls_slice, ao_loc)
    
    return mat
    

def _rho2x2_to_rho_m_allow_GGA(rho2x2):
    """From rho2x2 to calculate rho and M which will be used in GGA calculations.

    Args:
        rho2x2 (tuple): a tuple of (rhoaa, rhoab, rhoba, rhobb)
            rhoaa is numpy.array of [4,ngrid] or [10,ngrid] for IBP calculations

    Returns:
        rho (numpy.array [4,ngrid]): density which contains the density and nabla rho
            or [10,ngrid] for IBP calculations.
        M (numpy.array [3,4,ngrid], or [3,10,ngrid] for ibp calculations.): 
            M[i,j,k]: i--> M_i ; j--> ==0 M; !=0 nabla_j M; k --> ith grid.
    """
    # * NOTE : raa,rbb is real ; rab,rba is complex
    raa, rab, rba, rbb = rho2x2
    ndim = raa.shape[0]
    rho = (raa + rbb).real
    mx = rab.real + rba.real
    my = rba.imag - rab.imag
    mz = raa - rbb
    ngrid = rho.shape[-1]
    m = numpy.zeros((3,ndim,ngrid))
    m[0] = mx
    m[1] = my
    m[2] = mz
    return rho, m

def _dm2c_to_rho2x2_allow_GGA(mol, ao, dm, non0tab, shls_slice, ao_loc, out=None):
    """From density matrix to calculate rho

    Args:
        mol (pyscf.gto class): gto class of the molecule calculated
        ao (tuple of numpy arrays): (aoa, aob); aoa is a numpy array of [4,ngrid,n2c] for alpha component for 
            Large component or Small component, aob is beta component.
        dm (numpy.array): [n2c, n2c], a density matrix for 2c basis.
        non0tab (numpy.array, optional) : Non-zero table
        shls_slice ([type]): [description]
        ao_loc ([type]): [description]
        out ([type], optional): [description]. Defaults to None.

    Returns:
        rhoaa, rhoab, rhoba, rhobb (numpy.array): rhoxx is [4,ngrid] for rho, nabla rho.
            NOTE: rhoba = \sum_{i} \phi_{i}^{b} \phi_{i}^{a,*}
    """
    aoa, aob = ao
    ngrid = aoa.shape[1]
    
    # ! Note : rhoba is \psi^{\alpha,\dagger} \psi^{\beta}, different from notes !
    rhoaa = numpy.zeros((4,ngrid))
    rhoba = numpy.zeros((4,ngrid),dtype = numpy.complex128) 
    rhoab = numpy.zeros((4,ngrid),dtype = numpy.complex128)
    rhobb = numpy.zeros((4,ngrid))
    # $ \sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\alpha} $
    out = _dot_ao_dm(mol, aoa[0], dm, non0tab, shls_slice, ao_loc, out=out)
    for i in range(0,4):
        # $ \sum_{\nu} (\sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\alpha}) \nu^{\alpha,*} $
        rhoaa[i] = numpy.einsum('pi,pi->p', aoa[i].real, out.real)
        rhoaa[i]+= numpy.einsum('pi,pi->p', aoa[i].imag, out.imag)
        # $ \sum_{\nu} (\sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\alpha})^{*} \nu^{\beta} $
        rhoba[i] = numpy.einsum('pi,pi->p', aob[i], out.conj())
    # $ \sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\beta} $
    out = _dot_ao_dm(mol, aob[0], dm, non0tab, shls_slice, ao_loc, out=out)
    for i in range(0,4):
        # $ \sum_{\nu} (\sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\beta})^{*} \nu^{\alpha} $
        rhoab[i] = numpy.einsum('pi,pi->p', aoa[i], out.conj())
        # $ \sum_{\nu} (\sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\beta}) \nu^{\beta,*} $
        # $ \sum_{\nu} (\sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\beta}) \nu^{\beta,*} $
        rhobb[i] = numpy.einsum('pi,pi->p', aob[i].real, out.real)
        rhobb[i]+= numpy.einsum('pi,pi->p', aob[i].imag, out.imag)
    tmp_rho1 = rhoba[1:4].copy()
    tmp_rho2 = rhoab[1:4].copy()
    for i in range(1,4):
        rhoaa[i] = rhoaa[i] * 2.0
        rhoba[i] = tmp_rho1[i-1] + tmp_rho2[i-1].conj()
        rhoab[i] = tmp_rho1[i-1].conj() + tmp_rho2[i-1]
        # rhoba[i] = rhoba[i]*2.0
        # rhoab[i] = rhoab[i]*2.0
        rhobb[i] = rhobb[i] * 2.0
    return rhoaa, rhoab, rhoba, rhobb

def _dm2c_to_rho2x2_allow_GGA_ibp(mol, ao, dm, non0tab, shls_slice, ao_loc, out=None):
    """From density matrix to calculate rho for IBP calculations, for a given
        spinor density matrix and corresponding aos. 
        
        Detailed formular can see 'PySCF四分量计算', written by pzc.

    Args:
        mol (pyscf.gto class): gto class of the molecule calculated
        ao (tuple of numpy arrays): [2,10,ngrid,n2c] for [0] aoa and [1] aob.
            aoa is a numpy array of [10,ngrid,n2c] for alpha component for 
            Large component or Small component, aob is beta component.
        dm (numpy.array): [n2c, n2c], a density matrix for 2c basis.
        non0tab (numpy.array, optional) : Non-zero table
        shls_slice ([type]): [description]
        ao_loc ([type]): [description]
        out ([type], optional): [description]. Defaults to None.

    Returns:
        rhoaa, rhoab, rhoba, rhobb (numpy.array): rhoxx is [10,ngrid] 
            for rho, nabla rho and nabla nabla rho.
            NOTE: rhoba = \sum_{i} \phi_{i}^{b} \phi_{i}^{a,*}
    """
    aoa, aob = ao
    ngrid = aoa.shape[1]
    
    # ! Note : rhoba is \psi^{\alpha,\dagger} \psi^{\beta}, different from notes !
    rhoaa = numpy.zeros((10,ngrid))
    rhoba = numpy.zeros((10,ngrid),dtype = numpy.complex128) 
    rhoab = numpy.zeros((10,ngrid),dtype = numpy.complex128)
    rhobb = numpy.zeros((10,ngrid))
    # Some auxiliary arrays
    offset = numpy.zeros((3,3),dtype = numpy.int8)
    offset[0,0] = 4
    offset[0,1] = 5
    offset[0,2] = 6
    offset[1,1] = 7
    offset[1,2] = 8
    offset[2,2] = 9
    # auxiliary arrays for 2nd rhos.
    rhoba1 = numpy.zeros((3,3,ngrid), dtype = numpy.complex128)
    rhoba2 = numpy.zeros((6,ngrid), dtype = numpy.complex128) # nabla nabla beta
    rhoba3 = numpy.zeros((6,ngrid), dtype = numpy.complex128) # nabla nabla alpha
    
    
    # * The following for rho, nabla rho and nabla nabla rho part.
    # $ \sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\alpha} $
    out = _dot_ao_dm(mol, aoa[0], dm, non0tab, shls_slice, ao_loc, out=out)
    # * rho and nabla rho
    for i in range(0,4):
        # $ \sum_{\nu} (\sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\alpha}) \nu^{\alpha,*} $
        rhoaa[i] = numpy.einsum('pi,pi->p', aoa[i].real, out.real)
        rhoaa[i]+= numpy.einsum('pi,pi->p', aoa[i].imag, out.imag)
        # $ \sum_{\nu} (\sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\alpha})^{*} \nu^{\beta} $
        rhoba[i] = numpy.einsum('pi,pi->p', aob[i], out.conj())
    # \sum_{i\mu\nu} \nu^{\alpha,*}(\vec{r})\nabla_v\nabla_u\mu^{\alpha}(\vec{r}) 
    #   C_{\mu i} C_{\nu i}^{*}    
    # * This is for 2nd part.
    for i in range(4,10): 
        rhoaa[i] = numpy.einsum('pi,pi->p', aoa[i].real, out.real)
        rhoaa[i]+= numpy.einsum('pi,pi->p', aoa[i].imag, out.imag)
        rhoba2[i-4] = numpy.einsum('pi,pi->p', aob[i], out.conj())
        
    # $ \sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \nabla_u \mu^{\alpha} $
    for u in range(3):
        out = _dot_ao_dm(mol, aoa[u+1], dm, non0tab, shls_slice, ao_loc, out=out)
        # rhoaa part.
        for v in range(u,3):
            rhoaa[offset[u,v]]+= numpy.einsum('pi,pi->p', aoa[v+1].real, out.real)
            rhoaa[offset[u,v]]+= numpy.einsum('pi,pi->p', aoa[v+1].imag, out.imag)      
        for v in range(3):
            rhoba1[u,v] = numpy.einsum('pi,pi->p', aob[v+1], out.conj())
        
    
    # $ \sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\beta} $
    out = _dot_ao_dm(mol, aob[0], dm, non0tab, shls_slice, ao_loc, out=out)
    # * rho and nabla rho
    for i in range(0,4):
        # $ \sum_{\nu} (\sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\beta})^{*} \nu^{\alpha} $
        rhoab[i] = numpy.einsum('pi,pi->p', aoa[i], out.conj())
        # $ \sum_{\nu} (\sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \mu^{\beta}) \nu^{\beta,*} $
        rhobb[i] = numpy.einsum('pi,pi->p', aob[i].real, out.real)
        rhobb[i]+= numpy.einsum('pi,pi->p', aob[i].imag, out.imag)
    # \sum_{i\mu\nu} \nu^{\beta,*}(\vec{r})\nabla_v\nabla_u\mu^{\beta}(\vec{r}) 
    #   C_{\mu i} C_{\nu i}^{*}
    # * This is for 2nd part.
    for i in range(4,10):
        rhobb[i] = numpy.einsum('pi,pi->p', aob[i].real, out.real)
        rhobb[i]+= numpy.einsum('pi,pi->p', aob[i].imag, out.imag)
        rhoba3[i-4] = numpy.einsum('pi,pi->p', aoa[i].conj(), out)
    # $ \sum_{\mu i} C_{\mu i} C_{\nu i}^{*} \nabla_u \mu^{\beta} $
    for u in range(3):
        out = _dot_ao_dm(mol, aob[u+1], dm, non0tab, shls_slice, ao_loc, out=out)
        # rhobb part.
        for v in range(u,3):
            rhobb[offset[u,v]]+= numpy.einsum('pi,pi->p', aob[v+1].real, out.real)
            rhobb[offset[u,v]]+= numpy.einsum('pi,pi->p', aob[v+1].imag, out.imag)      
    
    # * get the rho and nabla rho    
    tmp_rho1 = rhoba[1:4].copy()
    tmp_rho2 = rhoab[1:4].copy()
    rhoaa[1:4] = rhoaa[1:4] * 2.0
    rhoba[1:4] = tmp_rho1[0:3] + tmp_rho2[0:3].conj()
    rhoab[1:4] = tmp_rho1[0:3].conj() + tmp_rho2[0:3]
    rhobb[1:4] = rhobb[1:4] * 2.0
    tmp_rho1 = tmp_rho2 = None
    
    # * get the nabla nabla rho
    rhoaa[4:10] = rhoaa[4:10]*2.0
    rhobb[4:10] = rhobb[4:10]*2.0
    rhoba[4] = rhoba3[0] + rhoba1[0,0]*2.0 + rhoba2[0]
    rhoba[5] = rhoba3[1] + rhoba1[0,1] + rhoba1[1,0] + rhoba2[1]
    rhoba[6] = rhoba3[2] + rhoba1[0,2] + rhoba1[2,0] + rhoba2[2]
    rhoba[7] = rhoba3[3] + rhoba1[1,1]*2.0 + rhoba2[3]
    rhoba[8] = rhoba3[4] + rhoba1[1,2] + rhoba1[2,1] + rhoba2[4]
    rhoba[9] = rhoba3[5] + rhoba1[2,2]*2.0 + rhoba2[5]
    rhoab[4:10] = rhoba[4:10].conj()
 
    return rhoaa, rhoab, rhoba, rhobb

def r_vxc_NC(ni, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
          max_memory=2000, verbose=None):
    """calculate Vxc matrix for non-collinear GGA.

    Args:
        ni (pyMC.gksm.numint_gksmc_r class): numint utils.
        mol (pyscf.gto class): gto class of the molecule calculated
        grids (pyscf.grid class): grid informations.
        xc_code (str): xc functional
        dms (tuple): a tuple of density matrix
        spin (int, optional): whether the system is spin polarised. Defaults to 0.
        relativity (int, optional): whether the system is relativistic. Defaults to 0.
        hermi (int, optional): whether the density matrix is hermite of not. Defaults to 1.
        max_memory (int, optional): [description]. Defaults to 2000.
        verbose ([type], optional): [description]. Defaults to None.

    Raises:
        NotImplementedError: GGA functional 
        NotImplementedError: Meta-GGA functional 

    Returns:
        nelec (tuple) (int, int): number of electrons
        excsum (float): functional energy
        vmat (numpy.array): [2*n2c, 2*n2c] for 4-c calculations. Vxc matrix.
    """
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]

    # Because the nao is not specified, thus nao is not defined for n2c or n4c
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    with_s = (nao == n2c*2)  # 4C DM

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    matLL = numpy.zeros((nset,n2c,n2c), dtype=numpy.complex128)
    matSS = numpy.zeros((nset,n2c,n2c), dtype=numpy.complex128)
    if xctype == 'LDA':
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, 0, with_s, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                vrho = vxc[0]
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += (den*exc).sum()

                matLL[idm] += r_numint._vxc2x2_to_mat(mol, ao[:2], weight, rho, vrho,
                                             mask, shls_slice, ao_loc)
                if with_s:
                    matSS[idm] -= r_numint._vxc2x2_to_mat(mol, ao[2:], weight, rho, vrho,
                                                 mask, shls_slice, ao_loc)
                rho = m = exc = vxc = vrho = None
    elif xctype == 'GGA':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    if with_s:
        c1 = .5 / lib.param.LIGHT_SPEED
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        for idm in range(nset):
            vmat[idm,:n2c,:n2c] = matLL[idm]
            vmat[idm,n2c:,n2c:] = matSS[idm] * c1**2
    else:
        vmat = matLL

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
    return nelec, excsum, vmat.reshape(dms.shape)
  
    
def MC_LDA_parallel_kernel(Mx, My, Mz, NX, rho, xc_code, ni, index
                             , relativity, verbose, weight, ngrid, factor):
    init, finish = index
    exctot = 0.0
    vrho_tot = numpy.zeros((ngrid))
    vs_tot = numpy.zeros((3,ngrid))
    Bxc_tot = numpy.zeros((3,ngrid))
    for idrct in range(init, finish):
        s = Mx*NX[idrct,0] \
            + My*NX[idrct,1] \
            + Mz*NX[idrct,2]
            # s[0] = numpy.sqrt(M[0,0]**2 + M[1,0]**2 + M[2,0]**2) 
        exc, exc_cor, vxcn, vxcs = ni.eval_xc_new_ASDP(xc_code, (0.5*(rho+s),0.5*(rho-s)) , spin=1,
                            relativity=relativity, deriv=1,
                            verbose=verbose)
        # averaging factor is produced.
        den = rho * weight
        # import pdb
        # pdb.set_trace()
        exctot += numpy.dot(den, exc)*factor[idrct]
        exctot += numpy.dot(weight, exc_cor)*factor[idrct]

        vrho_tot += vxcn*factor[idrct]
        for i in range(3):
            vs_tot[i] += vxcs*NX[idrct,i]*factor[idrct]

        Bxc_tot[0] += vxcs*NX[idrct,0]*factor[idrct]
        Bxc_tot[1] += vxcs*NX[idrct,1]*factor[idrct]
        Bxc_tot[2] += vxcs*NX[idrct,2]*factor[idrct]
    
    return exctot, vrho_tot, vs_tot, Bxc_tot

def MC_GGA_parallel_kernel(Mx, My, Mz, NX, rho, xc_code, ni, index
                             , relativity, verbose, weight, ngrid, factor):
    init, finish = index
    s = numpy.zeros((4,ngrid))
    exctot = 0.0
    wvrho_tot = numpy.zeros((4,ngrid))
    wvs_tot = numpy.zeros((3,4,ngrid))
    Bxc_tot = numpy.zeros((3,ngrid))
    den = rho[0] * weight
    for idrct in range(init, finish):
        s[:] = Mx*NX[idrct,0] \
             + My*NX[idrct,1] \
             + Mz*NX[idrct,2]
        exc, exccor, vxcn, vxcs = ni.eval_xc_new_ASDP(xc_code, (0.5*(rho+s),0.5*(rho-s)) , spin=1,
                        relativity=relativity, deriv=1,
                        verbose=verbose)
        # averaging factor is produced.
        exc = exc*factor[idrct]
        vxcn = vxcn*factor[idrct]*weight
        vxcs = vxcs*factor[idrct]*weight
        vxcn[0] = vxcn[0]*0.5
        vxcs[0] = vxcs[0]*0.5
        # wva and wvb is [4, ngrid]
        wvrho_tot += vxcn
        for i in range(3):
            for j in range(4):
                wvs_tot[i,j]+= vxcs[j]*NX[idrct,i]
        # averaging factor is produced.
        Bxc_tot[0] += vxcs[0]*NX[idrct,0]
        Bxc_tot[1] += vxcs[0]*NX[idrct,1]
        Bxc_tot[2] += vxcs[0]*NX[idrct,2]
        exctot += (den*exc).sum()
        exctot += (weight*exccor).sum()
    
    return exctot, wvrho_tot, wvs_tot, Bxc_tot

def r_mc_parallel(ni, mol, grids, xc_code, dms, relativity=1, hermi=1,
           max_memory=2000, Ndirect=1454, LIBXCT_factor=1.0E-10, 
           MSL_factor = 0.999, verbose=None, ncpu = None):
    """A Tri-directions subroutine for 4-c calculations.
        LDA and GGA are both implemented.
        
        It should be noted that, in 4c-Dirac calculations. the matrix is in the form:
        ( LL LS )
        ( SL SS )
        
        Note: Multidirections for DKS calculations doesn't support collinear calculations.

    Args:
        ni (pyMC.gksm.numint_gksmc_r class): numint utils.
        
        mol (pyscf.gto class): gto class of the molecule calculated
        
        grids (pyscf.grid class): grid informations.
        
        xc_code (str): xc functional
        
        dms (numpy.array): numpy array for density matrix.
        
        spin (int, optional): whether the system is spin polarised. Defaults to 0.
        
        relativity (int, optional): whether the system is relativistic. Defaults to 0.
        
        hermi (int, optional):  whether the density matrix is hermite of not. Defaults to 1.
        
        max_memory (int, optional):  Defaults to 2000.
        
        NX (numpy.array [ndirect,3]): Projecting directions, get from Thomson Problem.
        
        verbose ([type], optional):  No effect! Defaults to None.

    Raises:
        NotImplementedError: for meta-GGA
        
    Returns:
        nelec (tuple) (int, int): number of electrons
        excsum (float): functional energy
        vmat (numpy.array): [4*nao, 4*nao] for 4-c calculations. Vxc matrix.
    """
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]
    import math
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    nsbatch = math.ceil(Ndirect/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, Ndirect-nsbatch, nsbatch)]
    if NX_list[-1][-1] < Ndirect:
        NX_list.append((NX_list[-1][-1], Ndirect))
    NX,factor = ni.Spoints.make_sph_samples(Ndirect)
    # ~ init some parameters in parallel
    pool = multiprocessing.Pool()
    # ~ parallel run spherical average
    
    if xctype == 'LDA':
        make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    else:
        make_rho, nset, nao = ni._gen_rho_evaluator_GGA(mol, dms, hermi)
    with_s = (nao == n2c*2)  # 4C DM

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    matLL = numpy.zeros((nset,n2c,n2c), dtype=numpy.complex128)
    matSS = numpy.zeros((nset,n2c,n2c), dtype=numpy.complex128)
    numpy.save('coords',grids.coords)
    numpy.save('weights',grids.weights)
    if xctype == 'LDA':
        ipart = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, 0, with_s, max_memory):
            for idm in range(nset):
                rho, M = make_rho(idm, ao, mask, xctype)
                ngrid = rho.shape[-1]
                ipart += 1
                numpy.save('M_relative_part'+str(ipart),M)
                s = numpy.zeros((ngrid))
                vrho_tot = numpy.zeros((ngrid))
                vs_tot = numpy.zeros((3,ngrid))
                Bxc_tot = numpy.zeros((3,ngrid))
                # ~ init some parameters in parallel
                para_results = [] 
                # ~ parallel run spherical average
                # ~ parallel run spherical average
                for para in NX_list:
                    para_results.append(pool.apply_async(MC_LDA_parallel_kernel,
                                                         (M[0], M[1], M[2], NX, rho, xc_code, ni, para
                                                        , relativity, verbose, weight, ngrid, factor)))
                # ~ finisht the parallel part.
                pool.close()
                pool.join()
                # ~ get the final result
                for result_para in para_results:
                    result = result_para.get()
                    excsum[idm] += result[0]
                    Bxc_tot += result[3]
                    vrho_tot+= result[1]
                    vs_tot+= result[2]
                den = rho * weight
                nelec[idm] += den.sum()
                    
                numpy.save('Bxc_tot_part'+str(ipart),Bxc_tot)
                # numpy.save('Bxc_d_binary_part'+str(ipart),Bxc)
                # * Double Counting part.
                matLL[idm] += _vxc2x2_to_mat_MD_LDA_opt_MC(mol, ao[:2], weight, (vrho_tot, vs_tot),
                                            mask, shls_slice, ao_loc, 'LL')
                if with_s:
                    # ! Why this is plus not minus ??
                    matSS[idm] += _vxc2x2_to_mat_MD_LDA_opt_MC(mol, ao[2:], weight, (vrho_tot, vs_tot),
                                                mask, shls_slice, ao_loc, 'SS')
                rho = m = exc = vxc = vrho = None
    elif xctype == 'GGA':
        ipart = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, 1, with_s, max_memory):
            for idm in range(nset):
                rho, M = make_rho(idm, ao, mask, xctype)
                ngrid = rho.shape[1]
                ipart += 1
                numpy.save('M_relative_part'+str(ipart),M)
                numpy.save('NX_relative_part'+str(ipart),NX)
                s = numpy.zeros((4,ngrid))
                wvrho_tot = numpy.zeros((4,ngrid))
                wvs_tot = numpy.zeros((3,4,ngrid))
                Bxc_tot = numpy.zeros((3,ngrid))
                # ~ init some parameters in parallel
                para_results = [] 
                # ~ parallel run spherical average
                for para in NX_list:
                    para_results.append(pool.apply_async(MC_GGA_parallel_kernel,
                                                         (M[0], M[1], M[2], NX, rho, xc_code, ni, para
                                                        , relativity, verbose, weight, ngrid, factor)))
                # ~ finisht the parallel part.
                pool.close()
                pool.join()
                # ~ get the final result
                for result_para in para_results:
                    result = result_para.get()
                    excsum[idm] += result[0]
                    Bxc_tot += result[3]
                    wvrho_tot+= result[1]
                    wvs_tot+= result[2]
                den = rho[0] * weight
                numpy.save('Bxc_tot_part'+str(ipart),Bxc_tot)
                nelec[idm] += den.sum()
                matLL[idm] += _vxc2x2_to_mat_MD_GGA_opt(mol, ao[:2], (wvrho_tot, wvs_tot),
                                            mask, shls_slice, ao_loc, 'LL')
                if with_s:
                    # ! Why this is plus not minus ??
                    matSS[idm] += _vxc2x2_to_mat_MD_GGA_opt(mol, ao[2:], (wvrho_tot, wvs_tot),
                                                mask, shls_slice, ao_loc, 'SS')
                rho = M = exc = vxc = vrho = None
    else:
        raise NotImplementedError

    if with_s:
        c1 = .5 / lib.param.LIGHT_SPEED
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        for idm in range(nset):
            vmat[idm,:n2c,:n2c] = matLL[idm]
            vmat[idm,n2c:,n2c:] = matSS[idm] * c1**2
    else:
        vmat = matLL

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]

    return nelec, excsum, vmat.reshape(dms.shape)

   
def eval_rho_allow_GGA(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    """ 2-C ao --> rho generator
        this subroutine will get the nabla rho and nabla M

    Args:
        mol ([gto class]): 
        ao (numpy array): should contain the derivatives
        dm (numpy array): density matrix for LL or SS parts.
        non0tab (numpy.array, optional): [description]. Defaults to None.
        xctype (str, optional): [description]. Defaults to 'LDA'.
        hermi (int, optional): Whether hermit or not. Defaults to 0.
        verbose ([type], optional): [description]. Defaults to None.

    Raises:
        NotImplementedError: [Lapalacian of the rho and M is not implemented]

    Returns:
        rho[4,ngrid]
        M[3,4,ngrid]
    """
    aoa, aob = ao
    ngrids, n2c = aoa.shape[-2:]
    xctype = xctype.upper()

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()

    if xctype == 'LDA' or xctype == 'GGA':
        tmp = _dm2c_to_rho2x2_allow_GGA(mol, ao, dm, non0tab, shls_slice, ao_loc)
        rho, m = _rho2x2_to_rho_m_allow_GGA(tmp)
    else: # meta-GGA
        raise NotImplementedError
    # numpy.save('rho_new',rho)
    # numpy.save('m_new',m)
    return rho, m    

def eval_rho_allow_GGA_ibp(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    """ 2-C ao --> rho generator
        this subroutine will get the nabla rho and nabla M

    Args:
        mol ([gto class]): 
        ao (numpy array): should contain the derivatives
        dm (numpy array): density matrix for LL or SS parts.
        non0tab (numpy.array, optional): [description]. Defaults to None.
        xctype (str, optional): [description]. Defaults to 'LDA'.
        hermi (int, optional): Whether hermit or not. Defaults to 0.
        verbose ([type], optional): [description]. Defaults to None.

    Raises:
        NotImplementedError: [Lapalacian of the rho and M is not implemented]

    Returns:
        rho[10,ngrid] 10 --> ao,x,y,z,xx,xy,xz,yx,yy,yz,zx,zy,zz 
        M[3,10,ngrid] 3 --> Mx,My,Mz
    """
    aoa, aob = ao
    ngrids, nao2c = aoa.shape[-2:]
    xctype = xctype.upper()
    if xctype == 'LDA':
        raise ValueError("Only GGA can do IBP calculations!")

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()

    if xctype == 'LDA' or xctype == 'GGA':
        tmp = _dm2c_to_rho2x2_allow_GGA_ibp(mol, ao, dm, non0tab, shls_slice, ao_loc)
        rho, m = _rho2x2_to_rho_m_allow_GGA(tmp)
    else: # meta-GGA
        raise NotImplementedError
    # numpy.save('rho_new',rho)
    # numpy.save('m_new',m)
    return rho, m    

class numint_gksmc_r(r_numint.RNumInt):
    
    def __init__(self):
        r_numint.RNumInt.__init__(self)
        self.Spoints = Spoints.Spoints()
    
    r_vxc = r_vxc_NC
    r_mc_parallel = r_mc_parallel
    # eval_rho_allow_GGA = eval_rho_allow_GGA
    def eval_rho_allow_GGA(self, mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
        return eval_rho_allow_GGA(mol, ao, dm, non0tab, xctype, hermi, verbose)
    def eval_rho_allow_GGA_ibp(self, mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
        return eval_rho_allow_GGA_ibp(mol, ao, dm, non0tab, xctype, hermi, verbose)
    
    def _gen_rho_evaluator_GGA(self, mol, dms, hermi=1):
        dms = numpy.asarray(dms)
        # Note: nao is not specified, thus is not labeled by n2c or n4c
        nao = dms.shape[-1]
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            dms = dms.reshape(1,nao,nao)
        ndms = len(dms)
        n2c = mol.nao_2c()
        with_s = (nao == n2c*2)  # 4C DM
        if with_s:
            c1 = .5 / lib.param.LIGHT_SPEED
            dmLL = dms[:,:n2c,:n2c].copy('C')
            dmSS = dms[:,n2c:,n2c:] * c1**2

            def make_rho(idm, ao, non0tab, xctype):
                rho , m  = self.eval_rho_allow_GGA(mol, ao[:2], dmLL[idm], non0tab, xctype)
                rhoS, mS = self.eval_rho_allow_GGA(mol, ao[2:], dmSS[idm], non0tab, xctype)
                rho += rhoS
                # ! M = |\beta\Sigma|
                m[0] -= mS[0]
                m[1] -= mS[1]
                m[2] -= mS[2]
                return rho, m
        else:
            raise NotImplementedError('Only 4c with Small part rho and M is recommended.')
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho_allow_GGA(mol, ao, dms[idm], non0tab, xctype)
        return make_rho, ndms, nao
    
    def _gen_rho_evaluator_GGA_ibp(self, mol, dms, hermi=1):
        dms = numpy.asarray(dms)
        # note that nao is not specified, thus is not labeled by n2c or n4c.
        nao = dms.shape[-1]
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            dms = dms.reshape(1,nao,nao)
        ndms = len(dms)
        n2c = mol.nao_2c()
        with_s = (nao == n2c*2)  # 4C DM
        if with_s:
            c1 = .5 / lib.param.LIGHT_SPEED
            dmLL = dms[:,:n2c,:n2c].copy('C')
            dmSS = dms[:,n2c:,n2c:] * c1**2

            def make_rho(idm, ao, non0tab, xctype):
                rho , m  = self.eval_rho_allow_GGA_ibp(mol, ao[:2], dmLL[idm], non0tab, xctype)
                rhoS, mS = self.eval_rho_allow_GGA_ibp(mol, ao[2:], dmSS[idm], non0tab, xctype)
                rho += rhoS
                # ! M = |\beta\Sigma|
                m[0] -= mS[0]
                m[1] -= mS[1]
                m[2] -= mS[2]
                return rho, m
        else:
            raise NotImplementedError('Only 4c with Small part rho and M is recommended.')
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho_allow_GGA(mol, ao, dms[idm], non0tab, xctype)
        return make_rho, ndms, nao
    
    def eval_xc(self, xc_code, rho, spin=1, relativity=0, deriv=1, omega=None,
                verbose=None, weight=None):
        """This subroutine calculates the xc functional and their derivatives.

        Args:
            xc_code (str): XC functional
            rho (tuple): (rho,s), where rho and s is [4,ngrid] or [10,ngrid]
            spin (int, optional): spin polarized if spin > 0 Defaults to 1.
            relativity (int, optional): whether do relativity. Defaults to 0.
                Note, this is no effect!
            deriv (int, optional): derivative order. Defaults to 1.
            omega ([type], optional): . Defaults to None.
            verbose (int, optional): verbose level. Defaults to None.
            weight (numpy.array, optional): weight matrix. Defaults to None.

        Returns:
            (exc, vxc, fxc): for LDA. fxc is none, and vxc[0][0] = vrho
                and vxc[0][1] = vs
            (exc, wva, wvb): for GGA, wva, wvb are derivative of rhoa and rhob
            (exc, wva, wvb, wvrho_nrho, wvnrho_nrho): for GGA ibp calculations.
                wva, wvb are derivative of rhoa and rhob
        """
        if omega is None: omega = self.omega
        r, s = rho[:2]
        rhou = (r + s) * .5
        rhod = (r - s) * .5
        rho = (rhou, rhod)
        xc = self.libxc.eval_xc(xc_code, rho, 1, relativity, deriv,
                                omega, verbose)
        exc, vxc = xc[:2]
        xctype = self._xc_type(xc_code)
        # * update vxc[0] inplace
        if xctype == 'LDA':
            vrho = vxc[0]
            vr, vm = (vrho[:,0]+vrho[:,1])*.5, (vrho[:,0]-vrho[:,1])*.5
            vrho[:,0] = vr
            vrho[:,1] = vm
            
            return xc
        
        if xctype == 'GGA':
            if deriv == 1:
                # ! v+v.T should be applied in the caller
                # * wva[4,ngrid] \frac{\partial epsilon}{partial rhou}* 0.5
                # *              \frac{\partial epsilon}{partial \nabla rhou}
                wva, wvb = numint_gksmc._uks_gga_wv0((rhou,rhod), vxc, weight)
                
                return exc, wva, wvb
            elif deriv == 2:
                # ! v+v.T should be applied in the caller
                fxc = xc[2]
                wva, wvb, wvrho_nrho, wvnrho_nrho \
                        = numint_gksmc.uks_gga_wv0_intbypart_noweight((rhou,rhod), vxc, fxc)
                return exc, wva, wvb, wvrho_nrho, wvnrho_nrho
                
        
    def block_loop(self, mol, grids, nao, deriv=0, with_s=False, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
        '''
        Define this macro to loop over grids by blocks.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ngrids = grids.weights.size
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
        # NOTE to index ni.non0tab, the blksize needs to be the integer multiplier of BLKSIZE   
        if blksize is None:
            blksize = min(int(max_memory*1e6/((comp*4+4)*nao*16*BLKSIZE))*BLKSIZE, ngrids)
            blksize = max(blksize, BLKSIZE)
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                 dtype=numpy.uint8)

        if buf is None:
            buf = numpy.empty((4,comp,blksize,nao), dtype=numpy.complex128)
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao = self.eval_ao(mol, coords, deriv=deriv, with_s=with_s,
                              non0tab=non0, out=buf)
            yield ao, non0, weight, coords
    def eval_ao(self, mol, coords, deriv=0, with_s=True, shls_slice=None,
                non0tab=None, out=None, verbose=None):
        return eval_ao(mol, coords, deriv, with_s, shls_slice, non0tab, out, verbose)
    
    def eval_xc_new_ASDP(self, xc_code, rho, spin=1, relativity=0, deriv=1, omega=None,
                verbose=None, ibp = False):
        """calculate the xc functional used in new ASDP formalism.

        Args:
            xc_code (str): xc functional name
            rho (tuple): Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            spin (int, optional): wheter spin to be polarised. Defaults to 0.
            relativity (int, optional): whether use relativity, no effect. Defaults to 0.
            deriv (int, optional): derivatives of the density matrix. Defaults to 1.
            omega ([type], optional): [description]. Defaults to None.
            verbose ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        xctype = self._xc_type(xc_code)
        deriv_new = deriv + 1
        if omega is None: omega = self.omega
        # calculate and  split the functional
        exc, vxc, fxc, kxc= self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv_new,
                                  omega, verbose)[:4]
        # vrho, vsigma, vlapl, vtau = vxc
        # v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2
        #    , v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau = fxc
        
        # transformations of old variables
        # 1st order
        vrho = vxc[0]
        v2rho2 = fxc[0]
        u, d = vrho.T
        # 2nd order
        u_u, u_d, d_d = v2rho2.T
        # import pdb
        # pdb.set_trace()
        if ibp:
            # * GGA potetianl same part!
            vsigma = vxc[1]
            v2rhosigma = fxc[1]
            v2sigma2 = fxc[2]
            s = rho[0][0] - rho[1][0]
            ngrid = rho[0][0].shape[-1]
            nablas = rho[0][1:4] - rho[1][1:4]
            nablan = rho[0][1:4] + rho[1][1:4]
            # xx,xy,xz,yy,yz,zz
            nabla2s = rho[0][4:10] - rho[1][4:10]
            nabla2n = rho[0][4:10] + rho[1][4:10]
            # initiate some output arrays
            out_n = numpy.zeros((ngrid)) 
            out_s = numpy.zeros((ngrid))  
            
            offset2 = numint_gksmc.get_2d_offset()
            offset3 = numint_gksmc.get_3d_offset()
            # ~ Third order part BEGIND
            # One of the most troblesome codes are done in get_kxc_in_s_n
            # n_s_Ns : (3, ngrid) x y z
            # s_s_Ns : (3, ngrid) x y z
            # n_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
            # s_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
            # s_Nn_Ns : (3, 3, ngrid) (x y z) times (x y z)
            # Nn_Ns_Ns : (3,6,ngrid) (x y z) times (xx xy xz yy yz zz)
            # Ns_Ns_Ns : (10, ngrid) xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
            n_s_Ns, s_s_Ns, n_s_Nn, s_s_Nn, n_Ns_Ns, s_Ns_Ns, s_Nn_Ns, \
                n_Nn_Ns, s_Nn_Nn, Nn_Ns_Ns, Ns_Ns_Ns, Nn_Nn_Ns = numint_gksmc.get_kxc_in_s_n(rho, v2rhosigma, v2sigma2, kxc) 

            # init some temeperate paremeters!  
            # This part is also troublesome!        
            # ~ Third order part Finish
            
            # ~ Second order part BEGIND
            # calculate part of the derivatives
            # wva, wvb (2D numpy array): 
            # wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
            # wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
            #                               0      1    2      3     4     5
            # ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
            #   6     7     8      9    10   11      12    13    14   
            # bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
            #  15    16    17    18    19    20   
            wva, wvb, wvrho_nrho, wvnrho_nrho =\
                numint_gksmc.uks_gga_wv0_intbypart_noweight(rho, (vrho, vsigma), (v2rho2, v2rhosigma, v2sigma2))
            # initiate some temperate variables.
            pn_Ns = numpy.zeros((3,ngrid))
            pn_Nn = numpy.zeros((3,ngrid))
            ps_Ns = numpy.zeros((3,ngrid))
            ps_Nn = numpy.zeros((3,ngrid))
            pNn_Ns = numpy.zeros((3,3,ngrid))
            pNn_Nn = numpy.zeros((6,ngrid))
            pNs_Ns = numpy.zeros((6,ngrid))
            
            # construct temporary variables
            pn = 0.5*(wva + wvb) # * include the nabla part.
            ps = 0.5*(wva - wvb) # * include the nabla part.
            pn_s = 0.25*(u_u - d_d)
            ps_s = 0.25*(u_u - 2*u_d + d_d)
            
            pn_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] + wvrho_nrho[6] + wvrho_nrho[9] )
            pn_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] + wvrho_nrho[7] + wvrho_nrho[10])
            pn_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] + wvrho_nrho[8] + wvrho_nrho[11])
            
            pn_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] + wvrho_nrho[6] - wvrho_nrho[9] )
            pn_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] + wvrho_nrho[7] - wvrho_nrho[10])
            pn_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] + wvrho_nrho[8] - wvrho_nrho[11])
            
            ps_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] - wvrho_nrho[6] + wvrho_nrho[9] )
            ps_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] - wvrho_nrho[7] + wvrho_nrho[10])
            ps_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] - wvrho_nrho[8] + wvrho_nrho[11])
            
            ps_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] - wvrho_nrho[6] - wvrho_nrho[9] )
            ps_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] - wvrho_nrho[7] - wvrho_nrho[10])
            ps_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] - wvrho_nrho[8] - wvrho_nrho[11])
            
            pNn_Ns[0,0] = (wvnrho_nrho[0] - wvnrho_nrho[15])*0.25
            pNn_Ns[0,1] = (wvnrho_nrho[1] - wvnrho_nrho[7] + wvnrho_nrho[9]  - wvnrho_nrho[16])*0.25
            pNn_Ns[0,2] = (wvnrho_nrho[2] - wvnrho_nrho[8] + wvnrho_nrho[12] - wvnrho_nrho[17])*0.25
            pNn_Ns[1,0] = (wvnrho_nrho[1] - wvnrho_nrho[9] + wvnrho_nrho[7]  - wvnrho_nrho[16])*0.25
            pNn_Ns[1,1] = (wvnrho_nrho[3] - wvnrho_nrho[18])*0.25
            pNn_Ns[1,2] = (wvnrho_nrho[4] - wvnrho_nrho[11] + wvnrho_nrho[13] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,0] = (wvnrho_nrho[2] - wvnrho_nrho[12] + wvnrho_nrho[8]  - wvnrho_nrho[17])*0.25
            pNn_Ns[2,1] = (wvnrho_nrho[4] - wvnrho_nrho[13] + wvnrho_nrho[11] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,2] = (wvnrho_nrho[5] - wvnrho_nrho[20])*0.25
            
            pNs_Ns[0] = (wvnrho_nrho[0] - wvnrho_nrho[6 ] - wvnrho_nrho[6 ] + wvnrho_nrho[15])*0.25 # xx
            pNs_Ns[1] = (wvnrho_nrho[1] - wvnrho_nrho[7 ] - wvnrho_nrho[9 ] + wvnrho_nrho[16])*0.25 # xy
            pNs_Ns[2] = (wvnrho_nrho[2] - wvnrho_nrho[8 ] - wvnrho_nrho[12] + wvnrho_nrho[17])*0.25 # xz
            pNs_Ns[3] = (wvnrho_nrho[3] - wvnrho_nrho[10] - wvnrho_nrho[10] + wvnrho_nrho[18])*0.25 # yy
            pNs_Ns[4] = (wvnrho_nrho[4] - wvnrho_nrho[11] - wvnrho_nrho[13] + wvnrho_nrho[19])*0.25 # yz
            pNs_Ns[5] = (wvnrho_nrho[5] - wvnrho_nrho[14] - wvnrho_nrho[14] + wvnrho_nrho[20])*0.25 # zz
            
            pNn_Nn[0] = (wvnrho_nrho[0] + wvnrho_nrho[6 ] + wvnrho_nrho[6 ] + wvnrho_nrho[15])*0.25 # xx
            pNn_Nn[1] = (wvnrho_nrho[1] + wvnrho_nrho[7 ] + wvnrho_nrho[9 ] + wvnrho_nrho[16])*0.25 # xy
            pNn_Nn[2] = (wvnrho_nrho[2] + wvnrho_nrho[8 ] + wvnrho_nrho[12] + wvnrho_nrho[17])*0.25 # xz
            pNn_Nn[3] = (wvnrho_nrho[3] + wvnrho_nrho[10] + wvnrho_nrho[10] + wvnrho_nrho[18])*0.25 # yy
            pNn_Nn[4] = (wvnrho_nrho[4] + wvnrho_nrho[11] + wvnrho_nrho[13] + wvnrho_nrho[19])*0.25 # yz
            pNn_Nn[5] = (wvnrho_nrho[5] + wvnrho_nrho[14] + wvnrho_nrho[14] + wvnrho_nrho[20])*0.25 # zz
            wva = wvb = wvrho_nrho = wvnrho_nrho =None
            # ~ Second order part FINISH
            
            # ~ ~ ~
            # ~ Combine the second and third part to generate final potential derivatives
            # ~ ~ ~
            # The following part will use some temerate arrays
            # This part is the potential independent of gradient
            out_n = pn[0] + s*pn_s + nablas[0]*pn_Ns[0]+ nablas[1]*pn_Ns[1]+ nablas[2]*pn_Ns[2]
            out_s = 2*ps[0] + s*ps_s + nablas[0]*ps_Ns[0]+ nablas[1]*ps_Ns[1]+ nablas[2]*ps_Ns[2]
            # ~ Frist rho dependent part.
            # ~ temp1N is a temperate array to save paremeters.
            temp1N = numpy.zeros((ngrid))
            for u in range(3):
                # ~ 1st part
                # temp1N = gt_n_Nn
                temp1N = pn_Nn[u] + s*n_s_Nn[u] + nablas[0]*n_Nn_Ns[u,0] \
                    + nablas[1]*n_Nn_Ns[u,1] + nablas[2]*n_Nn_Ns[u,2]
                # ~ n_Nn part
                out_n -= temp1N*nablan[u]
                
                # ~ 2nd part
                # temp1N = gt_s_Nn
                temp1N = 2*ps_Nn[u] + s*s_s_Nn[u] + nablas[0]*s_Nn_Ns[u,0] \
                    + nablas[1]*s_Nn_Ns[u,1] + nablas[2]*s_Nn_Ns[u,2]
                # ~ s_Nn part
                out_n -= temp1N*nablas[u]
                
                # ~ 3rd part
                for v in range(3):
                    # temp1N = gt_Nn_Nn [u,v]
                    temp1N = pNn_Nn[offset2[u,v]] + s*s_Nn_Nn[offset2[u,v]] + nablas[0]*Nn_Nn_Ns[offset2[u,v],0] \
                        + nablas[1]*Nn_Nn_Ns[offset2[u,v],1] + nablas[2]*Nn_Nn_Ns[offset2[u,v],2]

                    out_n -= temp1N*nabla2n[offset2[u,v]]
                    
                    # ~ 4th part This part should be correct.
                    # temp1N = gt_Nn_Ns [u,v]
                    temp1N = 2*pNn_Ns[u,v] + s*s_Nn_Ns[u,v] + nablas[0]*Nn_Ns_Ns[u,offset2[v,0]] \
                        + nablas[1]*Nn_Ns_Ns[u,offset2[v,1]] + nablas[2]*Nn_Ns_Ns[u,offset2[v,2]]
                        
                    out_n -= temp1N*nabla2s[offset2[u,v]]
            # ~ spin dependent part
            for u in range(3):
                # ~ 1st part 
                # temp1N = n_Ns
                temp1N = 2*pn_Ns[u] + s*n_s_Ns[u] + nablas[0]*n_Ns_Ns[offset2[u,0]] \
                    + nablas[1]*n_Ns_Ns[offset2[u,1]] + nablas[2]*n_Ns_Ns[offset2[u,2]]
                out_s -= temp1N*nablan[u]
                
                # ~ 2nd part
                # temp1N = s_Ns
                temp1N = 3*ps_Ns[u] + s*s_s_Ns[u] + nablas[0]*s_Ns_Ns[offset2[u,0]] \
                    + nablas[1]*s_Ns_Ns[offset2[u,1]] + nablas[2]*s_Ns_Ns[offset2[u,2]]
                out_s -= temp1N*nablas[u]
                
                # ~ 3rd part
                # temp1N = Nn_Ns
                for v in range(3):
                    temp1N = 2*pNn_Ns[u,v] + s*s_Nn_Ns[u,v] + nablas[0]*Nn_Ns_Ns[u,offset2[v,0]] \
                        + nablas[1]*Nn_Ns_Ns[u,offset2[v,1]] + nablas[2]*Nn_Ns_Ns[u,offset2[v,2]]
                    out_s -= temp1N*nabla2n[offset2[u,v]]
                    
                    # ~ 4th part
                    # temp1N = Ns_Ns
                    temp1N = 3*pNs_Ns[offset2[u,v]] + s*s_Ns_Ns[offset2[u,v]] + nablas[0]*Ns_Ns_Ns[offset3[u,v,0]] \
                        + nablas[1]*Ns_Ns_Ns[offset3[u,v,1]] + nablas[2]*Ns_Ns_Ns[offset3[u,v,2]]
                    out_s -= temp1N*nabla2s[offset2[u,v]]
            
            return exc, s*ps[0] + nablas[0]*ps[1] + nablas[1]*ps[2] + nablas[2]*ps[3],\
                    out_n, out_s
                        
        else:
            if xctype == 'LDA':
                # transform to s    
                s = rho[0] - rho[1]
                # construct temporary variables
                pn = 0.5*(u + d)
                ps = 0.5*(u - d)
                pn_s = 0.25*(u_u - d_d)
                ps_s = 0.25*(u_u - 2*u_d + d_d)
                # construct new variables
                n_new = pn + s*pn_s
                s_new = 2*ps + s*ps_s
                # import pdb
                # pdb.set_trace()
                return exc, s*ps, n_new, s_new
                # return exc, pn, ps
            elif xctype == 'GGA':
                # transform to s 
                vsigma = vxc[1]
                v2rhosigma = fxc[1]
                v2sigma2 = fxc[2]
                s = rho[0][0] - rho[1][0]
                ngrid = rho[0][0].shape[-1]
                nablas = rho[0][1:4] - rho[1][1:4]
                # initiate some output arrays
                out_n = numpy.zeros((4,ngrid)) 
                out_s = numpy.zeros((4,ngrid))         
                # calculate part of the derivatives
                # wva, wvb (2D numpy array): 
                # wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
                # wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
                #                               0      1    2      3     4     5
                # ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
                #   6     7     8      9    10   11      12    13    14   
                # bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
                #  15    16    17    18    19    20   
                wva, wvb, wvrho_nrho, wvnrho_nrho =\
                    numint_gksmc.uks_gga_wv0_intbypart_noweight(rho, (vrho, vsigma), (v2rho2, v2rhosigma, v2sigma2))
                # initiate some temperate variables.
                pn_Ns = numpy.zeros((3,ngrid))
                ps_Ns = numpy.zeros((3,ngrid))
                ps_Nn = numpy.zeros((3,ngrid))
                pNn_Ns = numpy.zeros((3,3,ngrid))
                pNs_Ns = numpy.zeros((6,ngrid))
                # construct temporary variables
                pn = 0.5*(wva + wvb) # * include the nabla part.
                ps = 0.5*(wva - wvb) # * include the nabla part.
                pn_s = 0.25*(u_u - d_d)
                ps_s = 0.25*(u_u - 2*u_d + d_d)
                pn_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] + wvrho_nrho[6] - wvrho_nrho[9] )
                pn_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] + wvrho_nrho[7] - wvrho_nrho[10])
                pn_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] + wvrho_nrho[8] - wvrho_nrho[11])
                
                ps_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] - wvrho_nrho[6] + wvrho_nrho[9] )
                ps_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] - wvrho_nrho[7] + wvrho_nrho[10])
                ps_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] - wvrho_nrho[8] + wvrho_nrho[11])
                
                ps_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] - wvrho_nrho[6] - wvrho_nrho[9] )
                ps_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] - wvrho_nrho[7] - wvrho_nrho[10])
                ps_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] - wvrho_nrho[8] - wvrho_nrho[11])
                
                pNn_Ns[0,0] = (wvnrho_nrho[0] - wvnrho_nrho[15])*0.25
                pNn_Ns[0,1] = (wvnrho_nrho[1] - wvnrho_nrho[7] + wvnrho_nrho[9]  - wvnrho_nrho[16])*0.25
                pNn_Ns[0,2] = (wvnrho_nrho[2] - wvnrho_nrho[8] + wvnrho_nrho[12] - wvnrho_nrho[17])*0.25
                pNn_Ns[1,0] = (wvnrho_nrho[1] - wvnrho_nrho[9] + wvnrho_nrho[7]  - wvnrho_nrho[16])*0.25
                pNn_Ns[1,1] = (wvnrho_nrho[3] - wvnrho_nrho[18])*0.25
                pNn_Ns[1,2] = (wvnrho_nrho[4] - wvnrho_nrho[11] + wvnrho_nrho[13] - wvnrho_nrho[19])*0.25
                pNn_Ns[2,0] = (wvnrho_nrho[2] - wvnrho_nrho[12] + wvnrho_nrho[8]  - wvnrho_nrho[17])*0.25
                pNn_Ns[2,1] = (wvnrho_nrho[4] - wvnrho_nrho[13] + wvnrho_nrho[11] - wvnrho_nrho[19])*0.25
                pNn_Ns[2,2] = (wvnrho_nrho[5] - wvnrho_nrho[20])*0.25
                
                pNs_Ns[0] = (wvnrho_nrho[0] - wvnrho_nrho[6 ] - wvnrho_nrho[6 ] + wvnrho_nrho[15])*0.25 # xx
                pNs_Ns[1] = (wvnrho_nrho[1] - wvnrho_nrho[7 ] - wvnrho_nrho[9 ] + wvnrho_nrho[16])*0.25 # xy
                pNs_Ns[2] = (wvnrho_nrho[2] - wvnrho_nrho[8 ] - wvnrho_nrho[12] + wvnrho_nrho[17])*0.25 # xz
                pNs_Ns[3] = (wvnrho_nrho[3] - wvnrho_nrho[10] - wvnrho_nrho[10] + wvnrho_nrho[18])*0.25 # yy
                pNs_Ns[4] = (wvnrho_nrho[4] - wvnrho_nrho[11] - wvnrho_nrho[13] + wvnrho_nrho[19])*0.25 # yz
                pNs_Ns[5] = (wvnrho_nrho[5] - wvnrho_nrho[14] - wvnrho_nrho[14] + wvnrho_nrho[20])*0.25 # zz
                
                # * construct the final functional derivatives!
                
                out_n[0] = pn[0] + s*pn_s + nablas[0]*pn_Ns[0]+ nablas[1]*pn_Ns[1]+ nablas[2]*pn_Ns[2]
                out_s[0] = 2*ps[0] + s*ps_s + nablas[0]*ps_Ns[0]+ nablas[1]*ps_Ns[1]+ nablas[2]*ps_Ns[2]
                
                out_n[1] = pn[1] + s*ps_Nn[0] + nablas[0]*pNn_Ns[0,0]+ nablas[1]*pNn_Ns[0,1]+ nablas[2]*pNn_Ns[0,2]
                out_n[2] = pn[2] + s*ps_Nn[1] + nablas[0]*pNn_Ns[1,0]+ nablas[1]*pNn_Ns[1,1]+ nablas[2]*pNn_Ns[1,2]
                out_n[3] = pn[3] + s*ps_Nn[2] + nablas[0]*pNn_Ns[2,0]+ nablas[1]*pNn_Ns[2,1]+ nablas[2]*pNn_Ns[2,2]
                out_s[1] = 2*ps[1] + s*ps_Ns[0] + nablas[0]*pNs_Ns[0] + nablas[1]*pNs_Ns[1] + nablas[2]*pNs_Ns[2]
                out_s[2] = 2*ps[2] + s*ps_Ns[1] + nablas[0]*pNs_Ns[1] + nablas[1]*pNs_Ns[3] + nablas[2]*pNs_Ns[4]
                out_s[3] = 2*ps[3] + s*ps_Ns[2] + nablas[0]*pNs_Ns[2] + nablas[1]*pNs_Ns[4] + nablas[2]*pNs_Ns[5]
                
                
                # return exc, numpy.zeros((ngrid)), pn, ps
                
                return exc, s*ps[0] + nablas[0]*ps[1] + nablas[1]*ps[2] + nablas[2]*ps[3],\
                    out_n, out_s
            
            elif xctype == 'MGGA':
                raise NotImplementedError("Meta-GGA is not implemented")
        
           
        
