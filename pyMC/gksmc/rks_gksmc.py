#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-01-18 15:00:35
LastEditTime: 2022-04-12 09:53:04
LastEditors: Li Hao
Description: 
    A functional file to inherite by other subroutines.
FilePath: \pyMC\gksmc\rks_gksmc.py

 May the force be with you!
'''


from pyscf.dft import rks
from pyMC.gksmc import numint_gksmc
from pyscf.dft import gen_grid
from pyscf import __config__


def _dft_gksm_common_init_(mf, xc='LDA,VWN'):
    mf.xc = xc
    mf.nlc = ''
    mf.grids = gen_grid.Grids(mf.mol)
    mf.grids.level = getattr(__config__, 'dft_rks_RKS_grids_level',
                             mf.grids.level)
    mf.nlcgrids = gen_grid.Grids(mf.mol)
    mf.nlcgrids.level = getattr(__config__, 'dft_rks_RKS_nlcgrids_level',
                                mf.nlcgrids.level)
    # Use rho to filter grids
    mf.small_rho_cutoff = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)
##################################################
# don't modify the following attributes, they are not input options
    mf._numint = numint_gksmc.numint_gksmc()
    mf._keys = mf._keys.union(['xc', 'nlc', 'omega', 'grids', 'nlcgrids',
                               'small_rho_cutoff'])

class KohnShamDFT_MD(rks.KohnShamDFT):
    '''
    Attributes for Kohn-Sham DFT:
        xc : str
            'X_name,C_name' for the XC functional.  Default is 'lda,vwn'
        nlc : str
            'NLC_name' for the NLC functional.  Default is '' (i.e., None)
        omega : float
            Omega of the range-separated Coulomb operator e^{-omega r_{12}^2} / r_{12}
        grids : Grids object
            grids.level (0 - 9)  big number for large mesh grids. Default is 3

            radii_adjust
                | radi.treutler_atomic_radii_adjust (default)
                | radi.becke_atomic_radii_adjust
                | None : to switch off atomic radii adjustment

            grids.atomic_radii
                | radi.BRAGG_RADII  (default)
                | radi.COVALENT_RADII
                | None : to switch off atomic radii adjustment

            grids.radi_method  scheme for radial grids
                | radi.treutler  (default)
                | radi.delley
                | radi.mura_knowles
                | radi.gauss_chebyshev

            grids.becke_scheme  weight partition function
                | gen_grid.original_becke  (default)
                | gen_grid.stratmann

            grids.prune  scheme to reduce number of grids
                | gen_grid.nwchem_prune  (default)
                | gen_grid.sg1_prune
                | gen_grid.treutler_prune
                | None : to switch off grids pruning

            grids.symmetry  True/False  to symmetrize mesh grids (TODO)

            grids.atom_grid  Set (radial, angular) grids for particular atoms.
            Eg, grids.atom_grid = {'H': (20,110)} will generate 20 radial
            grids and 110 angular grids for H atom.

        small_rho_cutoff : float
            Drop grids if their contribution to total electrons smaller than
            this cutoff value.  Default is 1e-7.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', verbose=0)
    >>> mf = dft.RKS(mol)
    >>> mf.xc = 'b3lyp'
    >>> mf.kernel()
    -76.415443079840458
    '''
    __init__ = _dft_gksm_common_init_