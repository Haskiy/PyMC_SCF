#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-03-12 10:56:30
LastEditTime: 2022-04-12 09:52:09
LastEditors: Li Hao
Description: 
    4-component Dirac-Kohn-Sham with Tri-directions.
FilePath: \pyMC\gksmc\gksmc_r.py

 May the force be with you!
'''


import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import dhf
from pyscf.dft import rks
from pyMC.gksmc import numint_gksmc_r
from pyMC.gksmc import rks_gksmc


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional

    .. note::
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference Vxc potential matrix.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    t0 = (time.process_time(), time.time())
    t0 = logger.timer(ks, 'No meaning print test', *t0)
    numpy.save('dm_backup', dm)

    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 2)

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            ks.grids = rks.prune_small_rho_grids_(ks, mol, dm, ks.grids)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        if ks.ibp:
            raise NotImplementedError("Multi-collinear 4c Bxc is not implemented!")
        else:
            n, exc, vxc = ks._numint.r_mc_parallel(mol, ks.grids, ks.xc, dm, hermi=hermi,
                                       max_memory=max_memory, Ndirect=ks.Ndirect, 
                                       LIBXCT_factor=ks.LIBXCT_factor, 
                                        MSL_factor=ks.MSL_factor, ncpu = ks.ncpu)

        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
    if abs(hyb) < 1e-10:
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
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
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
            exc -= numpy.einsum('ij,ji', dm, vk).real * hyb * .5

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


energy_elec = rks.energy_elec

class GKSMC_r(rks_gksmc.KohnShamDFT_MD, dhf.UHF):
    '''Generalized Kohn-Sham with Tri-directions'''
    def __init__(self, mol, xc='LDA,VWN', toque_bxc = True):
        dhf.UHF.__init__(self, mol)
        rks_gksmc.KohnShamDFT_MD.__init__(self, xc)
        # gl._init()
        self._numint = numint_gksmc_r.numint_gksmc_r()
        self.Ndirect = 1454
        # Principle directions. Default to be 100
        self.LIBXCT_factor = 1.0E-10
        self.MSL_factor = 0.999
        self.ncpu = None
        self.ibp = False
        # Add the new keys
        self._keys = self._keys.union(['Ndirect', 'LIBXCT_factor', 'MSL_factor', 
                                       'ncpu', 'ibp'])
        
    def dump_flags(self, verbose=None):
        dhf.UHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = energy_elec
    
    def x2c1e(self):
        from pyscf.x2c import x2c
        x2chf = x2c.UKS(self.mol)
        x2c_keys = x2chf._keys
        x2chf.__dict__.update(self.__dict__)
        x2chf._keys = self._keys.union(x2c_keys)
        return x2chf
    x2c = x2c1e


    def nuc_grad_method(self):
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 3
    mol.atom = 'H 0 0 0; H 0 0 1; O .5 .6 .2'
    mol.basis = 'ccpvdz'
    mol.build()

    mf = GKSMC_r(mol)
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