#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-04-29 14:49:29
LastEditTime: 2022-04-12 09:52:20
LastEditors: Li Hao
Description: 
    Generalized Kohn-Sham with Tri-directions with symmetry
FilePath: \pyMC\gksmc\gksmc_symm.py

 May the force be with you!
'''


import time
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ghf
from pyscf.dft import rks
from pyMC.gksmc import numint_gksmc
from pyMC.gksmc import rks_gksmc, gks_sym_general
from pyMC.tools import group_proj

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for GKS with Tri-directions.
    '''
    # import pdb
    # pdb.set_trace()

    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    t0 = (time.process_time(), time.time())
    t0 = numint_gksmc.timer_no_clock(ks, 'No meaning print test', *t0)

    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 2)

    assert(hermi == 1)
    dm = numpy.asarray(dm)
    numpy.save('dm_backup', dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dm_aa = dm[...,:nao,:nao]   #.real
    dm_ab = dm[...,:nao,nao:]   #.real
    dm_ba = dm[...,nao:,:nao]   #.real
    dm_bb = dm[...,nao:,nao:]   #.real

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            ks.grids = rks.prune_small_rho_grids_(ks, mol, dm_aa+dm_bb, ks.grids)
        t0 = logger.timer(ks, 'setting up grids', *t0)
    if ks.nlc != '':
        raise NotImplementedError
    max_memory = ks.max_memory - lib.current_memory()[0]
    ni = ks._numint
    if ks.ibp:
        n, exc, vxctmp = ni.nr_mc_parallel_ibp(mol, ks.grids, ks.xc, (dm_aa, dm_ab, dm_ba, dm_bb),
            max_memory=max_memory, Ndirect=ks.Ndirect, LIBXCT_factor=ks.LIBXCT_factor, 
            MSL_factor=ks.MSL_factor, ncpu = ks.ncpu)
    else:
        n, exc, vxctmp = ni.nr_mc_parallel(mol, ks.grids, ks.xc, (dm_aa, dm_ab, dm_ba, dm_bb),
            max_memory=max_memory, Ndirect=ks.Ndirect, LIBXCT_factor=ks.LIBXCT_factor, 
            MSL_factor=ks.MSL_factor, ncpu = ks.ncpu)

    
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)
    vxc = numpy.asarray(scipy.linalg.block_diag(vxctmp[0],vxctmp[3]), dtype=numpy.complex128)
    vxc[:nao,nao:] = vxctmp[1].copy()
    vxc[nao:,:nao] = vxctmp[2].copy()
    
    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj = ks.get_j(mol, ddm, hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm, hermi)
        vxc += vj
    else:
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, ddm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj += vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, dm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -= numpy.einsum('ij,ji', dm, vk).real * .5
    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    
    return vxc

class GKSMC_symm(gks_sym_general.GKS_symm):
    '''Generalized Kohn-Sham with spherical average'''
    def __init__(self, mol, xc='LDA,VWN'):
        gks_sym_general.GKS_symm.__init__(self, mol)
        # Number of the directions. Default to be 1, collinear calculation.
        self.Ndirect = 1454
        # Principle directions. Default to be 100
        self.LIBXCT_factor = None
        self.MSL_factor = None
        self.ncpu = None
        self.ibp = False
        # Add the new keys
        self._keys = self._keys.union(['Ndirect', 'LIBXCT_factor', 'MSL_factor', 
                                       'ncpu', 'ibp'])
    def dump_flags(self, verbose=None):
        ghf.GHF.dump_flags(self, verbose)
        rks_gksmc.KohnShamDFT_MD.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = rks.energy_elec

    def nuc_grad_method(self):
        raise NotImplementedError
    

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 3
    mol.atom = 'H 0 0 0; H 0 0 1; O .5 .6 .2'
    mol.basis = 'ccpvdz'
    mol.build()

    mf = GKSMC(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    dm = mf.init_guess_by_1e(mol)
    dm = dm + 0j
    nao = mol.nao_nr()
    numpy.random.seed(12)
    dm[:nao,nao:] = numpy.random.random((nao,nao)) * .1j
    dm[nao:,:nao] = dm[:nao,nao:].T.conj()
    mf.kernel(dm)
    mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    mf.analyze()
    print(mf.spin_square())
    print(mf.e_tot - -76.2760115704274)