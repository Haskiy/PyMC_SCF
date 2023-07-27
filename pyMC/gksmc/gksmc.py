#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-01-18 09:04:27
LastEditTime: 2023-02-25 08:51:37
LastEditors: Li Hao
Description: 
    Generalized Kohn-Sham with Tri-directions.
FilePath: /pyMC/gksmc/gksmc.py

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
from pyMC.gksmc import rks_gksmc

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
    dm_aa = dm[...,:nao,:nao]
    dm_ab = dm[...,:nao,nao:]
    dm_ba = dm[...,nao:,:nao]
    dm_bb = dm[...,nao:,nao:]

    # import pdb
    # pdb.set_trace()
    
    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            ks.grids = rks.prune_small_rho_grids_(ks, mol, dm_aa+dm_bb, ks.grids)
        t0 = logger.timer(ks, 'setting up grids', *t0)
    if ks.nlc != '':
        raise NotImplementedError
    max_memory = ks.max_memory - lib.current_memory()[0]
    
    ni = ks._numint
    ni.Spoints.Tdistrion = ks.Sp_dismethod
    # To Do: Gauss_legendre's Npoints ---> assert())
    
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
    # import pdb
    # pdb.set_trace()

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

class GKSMC(rks_gksmc.KohnShamDFT_MD, ghf.GHF):
    '''Generalized Kohn-Sham with spherical average'''
    def __init__(self, mol, xc='LDA,VWN'):
        ghf.GHF.__init__(self, mol)
        rks_gksmc.KohnShamDFT_MD.__init__(self, xc)
        # Number of the directions. Default to be 1, collinear calculation.
        self.Ndirect = 1454
        self.LIBXCT_factor = None
        self.MSL_factor = None
        self.ncpu = None
        self.ibp = False
        self.Sp_dismethod = 'LebedevGrid'
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
    molcoords = """
    O                  0.00000000    0.00000000   -0.10983178 ;
    H                  0.00000000   -0.75754080    0.47724786 ;
    H                 -0.00000000    0.75754080    0.47724786 ;
    """

    THRESHOLD_list = [(-1.0, 10.0), (1.0E-10, 0.999)]
    Ndirect = 1454
    functional_list = ['SVWN','PBE','PBE']
    IBP_list = [False, False, True]
    output_dict = {}

    mol = gto.Mole()
    mol.atom = molcoords
    mol.spin=1 # ^ Spin
    mol.charge = 1
    mol.basis = "cc-pvtz" # ^ Basis
    mol.max_memory = 50000
    mol.build()

    BENCHMARK = [
        # No threshold
        # Threshold
    ]

    for THRESHOLD in THRESHOLD_list:
        for idx, xc in enumerate(functional_list):
            mftot = GKSMC(mol)
            mftot.xc = xc
            mftot.ibp = IBP_list[idx]
            mftot.LIBXCT_factor = THRESHOLD[0]
            mftot.MSL_factor = THRESHOLD[1]
            mftot.Ndirect = Ndirect
            mftot.max_cycle = 50
            mftot.kernel()