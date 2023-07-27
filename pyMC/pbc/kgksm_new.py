'''
Author: Li Hao
Date: 2021-11-12 14:08:57
LastEditTime: 2022-03-25 10:06:27
LastEditors: Pu Zhichen
Description: 

FilePath: \pyMC\pbc\kgksm_new.py
Motto: A + B = C!
'''
import time
import numpy 
from pyscf import lib
import scipy.linalg
from pyscf.lib import logger
from pyscf.pbc.scf import kghf 
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyMC.gksm_util import rks_gksm
from pyscf.pbc.dft import multigrid
from pyscf import __config__
from pyMC.pbc import numint_gksm
from pyMC.lib import priciple_direction

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional for UKS.  See pyscf/pbc/dft/gks.py
    :func:`get_veff` fore more details.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (time.perf_counter(), time.time())

    ks.read_priciple_direction()
    if ks.Ndirect == 1:
        ctrlNX = 'collinear'
    else:
        ctrlNX = 'Multidirections'

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
    numpy.save('dm_backup', dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dm_a = dm[...,:nao,:nao]
    dm_b = dm[...,nao:,nao:]
    
    # if ks.grids.non0tab is None:
    #     ks.grids.build(with_non0tab=True)
    #     if (isinstance(ks.grids, gen_grid.BeckeGrids) and
    #         ks.small_rho_cutoff > 1e-20 and ground_state):
    #         ks.grids = rks.prune_small_rho_grids_(ks, cell, dm_a + dm_b, ks.grids, kpts) # need
    #     t0 = logger.timer(ks, 'setting up grids', *t0)
 
    max_memory = ks.max_memory - lib.current_memory()[0]

    # import pdb
    # pdb.set_trace()
    # The if ... else ... is compatible for kbands calculation.
    if kpts_band is None:
        # import pdb
        # pdb.set_trace()
        n, exc, vxctmp = ks._numint.nrk_new_ASDP_parallel(cell, ks.grids, ks.xc, dm,
                        kpts=kpts, kpts_band=kpts_band,max_memory=max_memory, NX=ks.NX, 
                        THRESHOLD=ks.THRESHOLD, THRESHOLD_lc=ks.THRESHOLD_lc)
        vxc = numpy.empty((len(kpts),nso,nso),dtype=numpy.complex128)
        for i in range(0,len(kpts)):
            vxc[i] = numpy.asarray(scipy.linalg.block_diag(vxctmp[0,i],vxctmp[3,i]), dtype=numpy.complex128)
            vxc[i][:nao,nao:] = vxctmp[1,i].copy()
            vxc[i][nao:,:nao] = vxctmp[2,i].copy()
    else:
        n, exc, vxctmp = ks._numint.nrk_new_ASDP_parallel(cell, ks.grids, ks.xc, dm,
                        kpts=kpts, kpts_band=kpts_band,max_memory=max_memory, NX=ks.NX)
        # n, exc, vxctmp = ks._numint.nrk_new_ASDP_parallel(cell, ks.grids, ks.xc, dm,
        #                 kpts=kpts, kpts_band=None,max_memory=max_memory, NX=ks.NX)
        # ks._numint.nrk_new_ASDP_parallel(cell, ks.grids, ks.xc, dm,
        #                 kpts=kpts, kpts_band=None,max_memory=max_memory, NX=ks.NX)
        vxc = numpy.empty((len(kpts_band),nso,nso),dtype=numpy.complex128)
        for i in range(0,len(kpts_band)):
            vxc[i] = numpy.asarray(scipy.linalg.block_diag(vxctmp[0,i],vxctmp[3,i]), dtype=numpy.complex128)
            vxc[i][:nao,nao:] = vxctmp[1,i].copy()
            vxc[i][nao:,:nao] = vxctmp[2,i].copy()

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

class KGKSM_new(rks.KohnShamDFT, kghf.KGHF):
    '''GKS class adapted for PBCs with k-point sampling.
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        kghf.KGHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)
        # self.__dict__.update(self)  TO DO 
        #kgksm.__dict__.update(scf.load('cr3_kgksm_lda_N800.chk'))
        self._numint = numint_gksm.KNumInt()
         # Number of the directions. Default to be 1, collinear calculation.
        self.Ndirect = 1
        # Principle directions. Default to be 100
        self.NX = None
        self.THRESHOLD = 1.0E-10
        self.THRESHOLD_lc = 0.99
        self.N_instable = 0
        # Add the new keys
        self._keys = self._keys.union(['Ndirect', 'NX'])

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
    mf = KGKSM_new(cell, cell.make_kpts([2,1,1]))
    print(mf.kernel())
