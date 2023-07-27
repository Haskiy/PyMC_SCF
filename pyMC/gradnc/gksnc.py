#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-11-12 01:18:16
LastEditTime: 2022-11-12 01:21:49
LastEditors: Li Hao
Description: Non-relativistic GKS analytical nuclear gradients.

FilePath: /pyMC/gradnc/__init__.py
Motto: A + B = C!
'''

import numpy
import scipy
from pyscf import __config__, lib
from pyscf.dft import gen_grid, numint
from pyscf.grad import ghf_lh as ghf_grad
from pyscf.grad import rks as rks_grad
from pyscf.grad import uhf as uhf_grad
from pyscf.lib import logger


def get_veff(ks_grad, mol=None, dm=None):
    # import pdb
    # pdb.set_trace()
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    if mf.nlc != '':
        raise NotImplementedError
    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    nso = dm.shape[-1]
    nao = nso//2

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        raise NotImplementedError
        # exc, vxc = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
        #                                  max_memory=max_memory,
        #                                  verbose=ks_grad.verbose)
        # logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        if ks_grad.base.collinear == 'col':
            exc, vxc_tmp = get_vxc(ni, mol, grids, mf.xc, dm,
                            max_memory=max_memory, verbose=ks_grad.verbose)
            t0 = logger.timer(ks_grad, 'vxc', *t0)
            
            vxc = numpy.zeros((3,nso,nso))
            vxc[0] += numpy.asarray(scipy.linalg.block_diag(vxc_tmp[0,0],vxc_tmp[1,0]))
            vxc[1] += numpy.asarray(scipy.linalg.block_diag(vxc_tmp[0,1],vxc_tmp[1,1]))
            vxc[2] += numpy.asarray(scipy.linalg.block_diag(vxc_tmp[0,2],vxc_tmp[1,2]))
            
        elif ks_grad.base.collinear == 'mcol':
            exc, vxc_tmp = get_vxc_mc(ni,mol, grids, mf.xc, dm, 
                            max_memory=max_memory,verbose=ks_grad.verbose)
            t0 = logger.timer(ks_grad, 'vxc', *t0)
            
            vxc = numpy.zeros((3,nso,nso),dtype=numpy.complex128)
            vxc[0] += numpy.asarray(scipy.linalg.block_diag(vxc_tmp[0,0],vxc_tmp[3,0]))
            vxc[1] += numpy.asarray(scipy.linalg.block_diag(vxc_tmp[0,1],vxc_tmp[3,1]))
            vxc[2] += numpy.asarray(scipy.linalg.block_diag(vxc_tmp[0,2],vxc_tmp[3,2]))
            
            vxc[:,:nao,nao:] += vxc_tmp[1]
            vxc[:,nao:,:nao] += vxc_tmp[2]
    
    
    if abs(hyb) < 1e-10:
        vj = numpy.zeros((3,nso,nso),dtype=numpy.complex128)
        vj1,vj2= ks_grad.get_j(mol, (dm[:nao,:nao].real,dm[nao:,nao:].real))
        vj[:,:nao,:nao] = vj1+vj2
        vj[:,nao:,nao:] = vj1+vj2
        vxc += vj
    else:
        vj = numpy.zeros((3,nso,nso),dtype=numpy.complex128)
        vj1,vj2= ks_grad.get_j(mol, (dm[:nao,:nao].real,dm[nao:,nao:].real))
        vj[:,:nao,:nao] = vj1+vj2
        vj[:,nao:,nao:] = vj1+vj2
        vxc += vj
        
        vk = numpy.zeros((3,nso,nso),dtype=numpy.complex128)
        vk1,vk2 = ks_grad.get_k(mol, (dm[:nao,:nao].real,dm[nao:,nao:].real))
        vk1 *= hyb
        vk2 *= hyb
        
        vk3 = ks_grad.get_k(mol, (dm[:nao,nao:]+dm[nao:,:nao]).real)
        vk4 = ks_grad.get_k(mol, (dm[:nao,nao:]*1.0j-dm[nao:,:nao]*1.0j).real)
        vk3 *= hyb
        vk4 *= hyb
        
        vk[:,:nao,:nao] += vk1
        vk[:,nao:,nao:] += vk2
        vk[:,:nao,nao:] += vk3 - vk4*1.0j
        vk[:,nao:,:nao] += vk4 + vk4*1.0j
        
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            with mol.with_range_coulomb(omega):
                vk1,vk2 = ks_grad.get_k(mol, (dm[:nao,:nao].real,dm[nao:,nao:].real))*(alpha - hyb)
                vk3 = ks_grad.get_k(mol, (dm[:nao,nao:]+dm[nao:,:nao]).real)*(alpha - hyb)
                vk4 = ks_grad.get_k(mol, (dm[:nao,nao:]*1.0j-dm[nao:,:nao]*1.0j).real)*(alpha - hyb)
                
                vk[:,:nao,:nao] += vk1
                vk[:,nao:,nao:] += vk2
                vk[:,:nao,nao:] += vk3 - vk4*1.0j
                vk[:,nao:,:nao] += vk4 + vk4*1.0j
                
        vxc -= vk
    return lib.tag_array(vxc, exc1_grid=exc)


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    # dm_aa,dm_bb = dms
    make_rho,nset,nso = ni._gen_rho_evaluator(mol, dms, hermi)[:3]
    nao = nso//2
    # make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0], mask, xctype)
                rho_a = 0.5*(rho[0]+rho[3])
                rho_b = 0.5*(rho[0]-rho[3])
                vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1,
                                verbose=verbose)[1][0]
                vxcn = 0.5*(vxc[:,0]+vxc[:,1])
                vxcs = 0.5*(vxc[:,0]-vxc[:,1])
                
                v_aa = vxcn + vxcs
                v_bb = vxcn - vxcs
              
                aow = numint._scale_ao(ao[0], weight*v_aa[:])
                rks_grad._d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
                aow = numint._scale_ao(ao[0], weight*v_bb[:])
                rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:4], mask, xctype) 
                rho_a = 0.5*(rho[0]+rho[3])
                rho_b = 0.5*(rho[0]-rho[3])
                
                vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1,
                                verbose=verbose)[1]
                wva, wvb = numint._uks_gga_wv0((rho_a,rho_b), vxc, weight)
                rks_grad._gga_grad_sum_(vmat[0], mol, ao, wva, mask, ao_loc)
                rks_grad._gga_grad_sum_(vmat[1], mol, ao, wvb, mask, ao_loc)

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
            
                rho = make_rho(idm, ao[:10], mask, xctype)
                rho_new = numpy.zeros((4,6,rho.shape[-1]))
                rho_new[:,0:4] = rho[:,0:4]
                rho_new[:,5]   = rho[:,4]
                rho_a = 0.5*(rho_new[0]+rho_new[3])
                rho_b = 0.5*(rho_new[0]-rho_new[3])
                
                vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1,
                                verbose=verbose)[1]
                
                wva, wvb = numint._uks_mgga_wv0((rho_a,rho_b), vxc, weight)
                rks_grad._gga_grad_sum_(vmat[0], mol, ao, wva, mask, ao_loc)
                rks_grad._gga_grad_sum_(vmat[1], mol, ao, wvb, mask, ao_loc)

                # *2 because wv[5] is scaled by 0.5 in _uks_mgga_wv0
                rks_grad._tau_grad_dot_(vmat[0], mol, ao, wva[5]*2, mask, ao_loc, True)
                rks_grad._tau_grad_dot_(vmat[1], mol, ao, wvb[5]*2, mask, ao_loc, True)

    exc = numpy.zeros((mol.natm,3))
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def get_vxc_mc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    # dm_aa,dm_bb = dms
    make_rho,nset,nso = ni._gen_rho_evaluator(mol, dms, hermi)[:3]
    nao = nso//2
    # make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((4,3,nao,nao),dtype=numpy.complex128)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0], mask, xctype)
                eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
                
                vxc = eval_xc(xc_code, rho, deriv=1, xctype=xctype)[1]
                vxcn,vxc_mx,vxc_my,vxc_mz = vxc
                ngrid = vxcn.shape[-1]
                
                v_aa = (vxcn + vxc_mz).reshape(ngrid)
                v_ab_r = vxc_mx.reshape(ngrid)
                v_ab_i = vxc_my.reshape(ngrid)*1.0j
                v_bb = (vxcn - vxc_mz).reshape(ngrid)
              
                aow = numint._scale_ao(ao[0], weight*v_aa[:])
                rks_grad._d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
                
                aow = numint._scale_ao(ao[0], weight*v_ab_r[:])
                rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)
                aow = numint._scale_ao(ao[0], weight*v_ab_r[:])
                rks_grad._d1_dot_(vmat[2], mol, ao[1:4], aow, mask, ao_loc, True)
                
                aow = numint._scale_ao(ao[0], -1.0*weight*v_ab_i[:])
                rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)
                aow = numint._scale_ao(ao[0], weight*v_ab_i[:])
                rks_grad._d1_dot_(vmat[2], mol, ao[1:4], aow, mask, ao_loc, True)
                
                aow = numint._scale_ao(ao[0], weight*v_bb[:])
                rks_grad._d1_dot_(vmat[3], mol, ao[1:4], aow, mask, ao_loc, True)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0:4], mask, xctype)
                eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
                
                vxc = eval_xc(xc_code, rho, deriv=1, xctype=xctype)[1]
                vxcn,vxc_mx,vxc_my,vxc_mz = vxc
                ngrid = vxcn.shape[-1]
                
                wv_aa = vxcn[0] + vxc_mz[0]
                wv_ab_r = vxc_mx[0]
                wv_ab_i = vxc_my[0]*1.0j
                wv_bb = vxcn[0] - vxc_mz[0]
                
                wv_naa = vxcn[1:4] + vxc_mz[1:4]
                wv_nab_r = vxc_mx[1:4]
                wv_nab_i = vxc_my[1:4]*1.0j
                wv_nbb = vxcn[1:4] - vxc_mz[1:4]
                
                aow = numint._scale_ao(ao[0], weight*wv_aa[:])
                rks_grad._d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
                
                aow = numint._scale_ao(ao[0], weight*wv_ab_r[:])
                rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)
                aow = numint._scale_ao(ao[0], weight*wv_ab_r[:])
                rks_grad._d1_dot_(vmat[2], mol, ao[1:4], aow, mask, ao_loc, True)
                
                aow = numint._scale_ao(ao[0], -1.0*weight*wv_ab_i[:])
                rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)
                aow = numint._scale_ao(ao[0], weight*wv_ab_i[:])
                rks_grad._d1_dot_(vmat[2], mol, ao[1:4], aow, mask, ao_loc, True)
                
                aow = numint._scale_ao(ao[0], weight*wv_bb[:])
                rks_grad._d1_dot_(vmat[3], mol, ao[1:4], aow, mask, ao_loc, True)
                
                # XX, XY, XZ = 4, 5, 6
                # YX, YY, YZ = 5, 7, 8
                # ZX, ZY, ZZ = 6, 8, 9
                vmat[0] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_naa[:],ao[1:4],optimize = True)
                vmat[1] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_nab_r[:],ao[1:4],optimize = True)
                vmat[1] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],-1.0*weight*wv_nab_i[:],ao[1:4],optimize = True)
                vmat[2] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_nab_r[:],ao[1:4],optimize = True)
                vmat[2] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_nab_i[:],ao[1:4],optimize = True)
                vmat[3] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_nbb[:],ao[1:4],optimize = True)
                
                ao_new = numpy.zeros((3,3,ngrid,nao))
                ao_new[:3,0] = ao[4:7]
                ao_new[0,:3] = ao[4:7]
                ao_new[1,1] = ao[7]
                ao_new[2,2] = ao[9]
                ao_new[1,2] = ao[8]
                ao_new[2,1] = ao[8]
                
                vmat[0] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_naa[:],ao[0],optimize = True)
                vmat[1] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_nab_r[:],ao[0],optimize = True)
                vmat[1] += numpy.einsum('xynu,yn,nv->xuv',ao_new,-1.0*weight*wv_nab_i[:],ao[0],optimize = True)
                vmat[2] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_nab_r[:],ao[0],optimize = True)
                vmat[2] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_nab_i[:],ao[0],optimize = True)
                vmat[3] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_nbb[:],ao[0],optimize = True)

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:10], mask, xctype)
                eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
                
                vxc = eval_xc(xc_code, rho, deriv=1, xctype=xctype)[1]
                # vxc_mx.shape=(4,5,ngrid), where 5 means 1,x,y,z,tau
                vxcn,vxc_mx,vxc_my,vxc_mz = vxc
                
                ngrid = vxcn.shape[-1]
                
                wv_aa = vxcn[0] + vxc_mz[0] 
                wv_ab_r = vxc_mx[0]
                wv_ab_i = vxc_my[0]*1.0j
                wv_bb = vxcn[0] - vxc_mz[0]
                
                tauu_aa = vxcn[4] + vxc_mz[4] 
                tauu_ab_r = vxc_mx[4]
                tauu_ab_i = vxc_my[4]*1.0j
                tauu_bb = vxcn[4] - vxc_mz[4]
                
                wv_naa = vxcn[1:4] + vxc_mz[1:4]
                wv_nab_r = vxc_mx[1:4]
                wv_nab_i = vxc_my[1:4]*1.0j
                wv_nbb = vxcn[1:4] - vxc_mz[1:4]
                
                aow = numint._scale_ao(ao[0], weight*wv_aa[:])
                rks_grad._d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
                
                aow = numint._scale_ao(ao[0], weight*wv_ab_r[:])
                rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)
                aow = numint._scale_ao(ao[0], weight*wv_ab_r[:])
                rks_grad._d1_dot_(vmat[2], mol, ao[1:4], aow, mask, ao_loc, True)
                
                aow = numint._scale_ao(ao[0], -1.0*weight*wv_ab_i[:])
                rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)
                aow = numint._scale_ao(ao[0], weight*wv_ab_i[:])
                rks_grad._d1_dot_(vmat[2], mol, ao[1:4], aow, mask, ao_loc, True)
                
                aow = numint._scale_ao(ao[0], weight*wv_bb[:])
                rks_grad._d1_dot_(vmat[3], mol, ao[1:4], aow, mask, ao_loc, True)
                
                # XX, XY, XZ = 4, 5, 6
                # YX, YY, YZ = 5, 7, 8
                # ZX, ZY, ZZ = 6, 8, 9
                vmat[0] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_naa[:],ao[1:4],optimize = True)
                vmat[1] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_nab_r[:],ao[1:4],optimize = True)
                vmat[1] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],-1.0*weight*wv_nab_i[:],ao[1:4],optimize = True)
                vmat[2] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_nab_r[:],ao[1:4],optimize = True)
                vmat[2] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_nab_i[:],ao[1:4],optimize = True)
                vmat[3] += numpy.einsum('xnu,yn,ynv->xuv',ao[1:4],weight*wv_nbb[:],ao[1:4],optimize = True)
                
                ao_new = numpy.zeros((3,3,ngrid,nao))
                ao_new[:3,0] = ao[4:7]
                ao_new[0,:3] = ao[4:7]
                ao_new[1,1] = ao[7]
                ao_new[2,2] = ao[9]
                ao_new[1,2] = ao[8]
                ao_new[2,1] = ao[8]
                
                vmat[0] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_naa[:],ao[0],optimize = True)
                vmat[1] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_nab_r[:],ao[0],optimize = True)
                vmat[1] += numpy.einsum('xynu,yn,nv->xuv',ao_new,-1.0*weight*wv_nab_i[:],ao[0],optimize = True)
                vmat[2] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_nab_r[:],ao[0],optimize = True)
                vmat[2] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_nab_i[:],ao[0],optimize = True)
                vmat[3] += numpy.einsum('xynu,yn,nv->xuv',ao_new,weight*wv_nbb[:],ao[0],optimize = True)
                
                # # #
                vmat[0] += numpy.einsum('xynu,n,ynv->xuv',ao_new,weight*tauu_aa[:],ao[1:4],optimize = True)*.5
                vmat[1] += numpy.einsum('xynu,n,ynv->xuv',ao_new,weight*tauu_ab_r[:],ao[1:4],optimize = True)*.5
                vmat[1] += numpy.einsum('xynu,n,ynv->xuv',ao_new,-1.0*weight*tauu_ab_i[:],ao[1:4],optimize = True)*.5
                vmat[2] += numpy.einsum('xynu,n,ynv->xuv',ao_new,weight*tauu_ab_r[:],ao[1:4],optimize = True)*.5
                vmat[2] += numpy.einsum('xynu,n,ynv->xuv',ao_new,weight*tauu_ab_i[:],ao[1:4],optimize = True)*.5
                vmat[3] += numpy.einsum('xynu,n,ynv->xuv',ao_new,weight*tauu_bb[:],ao[1:4],optimize = True)*.5

    exc = numpy.zeros((mol.natm,3))
    return exc, -vmat


class Gradients(ghf_grad.Gradients):

    grid_response = getattr(__config__, 'grad_uks_Gradients_grid_response', False)

    def __init__(self, mf):
        ghf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.grid_response = False
        self._keys = self._keys.union(['grid_response', 'grids'])

    def dump_flags(self, verbose=None):
        uhf_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        return self

    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        '''Hook for extra contributions in analytical gradients.

        Contributions like the response of auxiliary basis in density fitting
        method, the grid response in DFT numerical integration can be put in
        this function.
        '''
        if self.grid_response:
            vhf = envs['vhf']
            log = envs['log']
            log.debug('grids response for atom %d %s',
                      atom_id, vhf.exc1_grid[atom_id])
            return vhf.exc1_grid[atom_id]
        else:
            return 0

Grad = Gradients

from pyscf import dft

dft.gks.GKS.Gradients = dft.gks_symm.GKS.Gradients = lib.class_as_method(Gradients)
