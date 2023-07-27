'''
Author: Li Hao
Date: 2021-11-10 16:44:07
LastEditTime: 2021-12-08 18:26:57
LastEditors: Li Hao
Description: 

FilePath: \pyMC\pbc\gksm_new.py
Motto: A + B = C!
'''

import time
import numpy
import pyscf.dft
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import ghf as pbcghf
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import multigrid
from pyMC.pbc import numint_gksm
from pyscf import __config__
from pyMC.lib import priciple_direction

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    '''Coulomb + XC functional for GKS of PBC.
    '''

    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpt is None: kpt = ks.kpt
    t0 = (time.process_time(), time.time())

    ks.read_priciple_direction()
    if ks.Ndirect == 1:
        ctrlNX = 'collinear'
    else:
        ctrlNX = 'Multidirections'
        
    dm = numpy.asarray(dm)
    numpy.save('dm_backup', dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dm_a = dm[...,:nao,:nao]
    dm_b = dm[...,nao:,nao:]
    
    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10 or abs(alpha) > 1e-10

    # import pdb
    # pdb.set_trace()
    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_uks(ks.with_df, ks.xc, dm, hermi,
                                       kpt.reshape(1,3), kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc

    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 2 and kpts_band is None)

    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = rks.prune_small_rho_grids_(ks, cell, dm_a + dm_b, ks.grids, kpt)
        t0 = logger.timer(ks, 'setting up grids', *t0)
    max_memory = ks.max_memory - lib.current_memory()[0]
    
    # import pdb
    # pdb.set_trace()
    n, exc, vxctmp = ks._numint.nr_new_ASDP_parallel(cell, ks.grids, ks.xc, dm,
                    kpts=kpt, kpts_band=kpts_band,max_memory=max_memory, NX=ks.NX)
    # import pdb
    # pdb.set_trace()
    vxc = numpy.asarray(scipy.linalg.block_diag(vxctmp[0],vxctmp[3]), dtype=numpy.complex128)
    vxc[:nao,nao:] = vxctmp[1].copy()
    vxc[nao:,:nao] = vxctmp[2].copy()
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)
    if vxctmp.ndim == 4:
        raise NotImplementedError

    #enabling range-separated hybrids
    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
        vxc += vj
    else:
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpt, kpts_band)
        vk *= hyb
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -=(numpy.einsum('ij,ji', dm, vk)).real * .5

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

@lib.with_doc(pbcghf.get_rho.__doc__)
def get_rho(mf, dm=None, grids=None, kpt=None):
    if dm is None:
        dm = mf.make_rdm1()
    return rks.get_rho(mf, dm, grids, kpt)

class GKSM_new(rks.KohnShamDFT, pbcghf.GHF):
    '''Generalized Kohn-Sham with spherical average for PBCs'''
    def __init__(self, cell, kpt=numpy.zeros(3), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbcghf.GHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)
        # Number of the directions. Default to be 1, collinear calculation.
        self.Ndirect = 1
        # Principle directions. Default to be 100
        self.NX = None
        # Add the new keys
        self._keys = self._keys.union(['Ndirect', 'NX'])
        self._numint = numint_gksm.NumInt()
        
    def dump_flags(self, verbose=None):
        pbcghf.GHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = pyscf.dft.rks.energy_elec
    get_rho = get_rho

    density_fit = rks._patch_df_beckegrids(pbcghf.GHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(pbcghf.GHF.mix_density_fit)

    def nuc_grad_method(self):
        raise NotImplementedError
    
    def read_priciple_direction(self):
        """
        Read priciple directions
        """
        self.NX = priciple_direction.NX[self.Ndirect]
        


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
    mf = GKSM_new(cell)
    print(mf.kernel())