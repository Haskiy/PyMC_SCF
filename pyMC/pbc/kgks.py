#!/usr/bin/env python
'''
Author: Li Hao
Date: 2021-09-17 14:40:00
LastEditTime: 2021-12-08 14:31:14
LastEditors: Li Hao
Description: 
    Generalized Kohn-Sham for Solid Calculations.
FilePath: \pyMC\pbc\kgks.py

    A + B = C!
'''

'''
Non-relativistic Generalized Kohn-Sham for periodic systems with k-point sampling

See Also:
    pyscf.pbc.dft.gks.py :  Non-relativistic Generalized Kohn-Sham for periodic
                            systems with k-point sampling
'''

import time
import numpy 
from pyscf import lib
import scipy.linalg
from pyscf.lib import logger
from pyscf.pbc.scf import kghf 
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import multigrid
from pyscf import __config__

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional for UKS.  See pyscf/pbc/dft/gks.py
    :func:`get_veff` fore more details.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (time.perf_counter(), time.time())

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_uks(ks.with_df, ks.xc, dm, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc

    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 3 and
                    kpts_band is None)

    dm = numpy.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2

    dm_a = dm[...,:nao,:nao]
    dm_b = dm[...,nao:,nao:]
    dms = numpy.asarray([dm_a, dm_b])

    # import pdb
    # pdb.set_trace()
    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = rks.prune_small_rho_grids_(ks, cell, dm_a + dm_b, ks.grids, kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)
 
    n, exc, vxctmp = ks._numint.nr_uks(cell, ks.grids, ks.xc, dms, 0,
                                    kpts, kpts_band)
    
    # The if ... else ... is compatible for kbands calculation.
    if kpts_band is None: 
        vxc = numpy.empty((len(kpts),nso,nso),dtype=numpy.complex128)
        for i in range(0,len(kpts)):
            vxc[i] = numpy.asarray(scipy.linalg.block_diag(vxctmp[0,i],vxctmp[1,i]), dtype=numpy.complex128)
    else:
        vxc = numpy.empty((len(kpts_band),nso,nso),dtype=numpy.complex128)
        for i in range(0,len(kpts_band)):
            vxc[i] = numpy.asarray(scipy.linalg.block_diag(vxctmp[0,i],vxctmp[1,i]), dtype=numpy.complex128)

    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)
    weight = 1./len(kpts)

    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        vxc += vj
        
    else:
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vk *= hyb
        
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
            
        if ground_state:
            exc -=(numpy.einsum('Kij,Kji', dm, vk)).real * .5 * weight

    if ground_state:
        ecoul = numpy.einsum('Kij,Kji', dm, vj).real * .5 * weight
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None): 
    # import pdb
    # pdb.set_trace()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    weight = 1./len(h1e_kpts)
    e1 = weight *(numpy.einsum('kij,kji', h1e_kpts, dm_kpts))
    tot_e = e1 + vhf.ecoul + vhf.exc
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = vhf.ecoul.real
    mf.scf_summary['exc'] = vhf.exc.real
    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s', e1, vhf.ecoul, vhf.exc)
    return tot_e.real, vhf.ecoul + vhf.exc


@lib.with_doc(kghf.get_rho.__doc__)
def get_rho(mf, dm=None, grids=None, kpts=None):
    from pyscf.pbc.dft import krks
    if dm is None:
        dm = mf.make_rdm1()
    return krks.get_rho(mf, dm, grids, kpts)

class KGKS(rks.KohnShamDFT, kghf.KGHF):
    '''GKS class adapted for PBCs with k-point sampling.
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        kghf.KGHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        kghf.KGHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = energy_elec
    get_rho = get_rho

    density_fit = rks._patch_df_beckegrids(kghf.KGHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(kghf.KGHF.mix_density_fit)
    
    def nuc_grad_method(self):
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    mf = KGKS(cell, cell.make_kpts([2,1,1]))
    print(mf.kernel())
