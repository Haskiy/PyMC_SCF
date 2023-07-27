#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-02-27 10:11:25
LastEditTime: 2023-04-25 05:07:26
LastEditors: Li Hao
Description: 
    partition the cluster to seperate atoms
FilePath: /pyMC/tools/partition2atom.py

 May the force be with you!
'''


import time
from pyscf import scf,gto,dft
from pyscf.dft import numint
import numpy
from pyscf.dft.gen_grid import make_mask, BLKSIZE

def partition2atom(ks):
    """Partition the grids into atoms, accordingt to the distance from the grid to atom.

    Args:
        ks (GKS or KS class): Saves the atomic coordinates and numerical grids.

    Returns:
        grids2atom [numpy array of ints]: saves the atom label which the specific grid belongs to.
    """
    atom_coords = ks.mol.atom_coords()
    grids_coords = ks.grids.coords
    ngrid = grids_coords.shape[0]
    grids2atom = - numpy.ones((ngrid),dtype = numpy.int)
    for igrid in range(ngrid):
        d = numpy.sqrt((grids_coords[igrid,0]-atom_coords[:,0])**2
                       +(grids_coords[igrid,1]-atom_coords[:,1])**2
                       +(grids_coords[igrid,2]-atom_coords[:,2])**2)
        grids2atom[igrid] = numpy.argmin(d)
    return grids2atom

def partition2atom_rt(atom_coords, grids_coords):
    """Partition the grids into atoms, accordingt to the distance from the grid to atom.

    Args:
        ks (GKS or KS class): Saves the atomic coordinates and numerical grids.

    Returns:
        grids2atom [numpy array of ints]: saves the atom label which the specific grid belongs to.
    """
    ngrid = grids_coords.shape[0]
    grids2atom = - numpy.ones((ngrid),dtype = numpy.int)
    for igrid in range(ngrid):
        d = numpy.sqrt((grids_coords[igrid,0]-atom_coords[:,0])**2
                       +(grids_coords[igrid,1]-atom_coords[:,1])**2
                       +(grids_coords[igrid,2]-atom_coords[:,2])**2)
        grids2atom[igrid] = numpy.argmin(d)
    return grids2atom

def get_M(ks,grids2atom,M):
    """get the M in each atom

    Args:
        ks (GKS or KS class): Saves the atomic coordinates and numerical grids.
        grids2atom (numpy array of ints): saves the atom label which the specific grid belongs to.
        M (1D tuple of Mx, My, Mz in which Mx is 1D [ngrid] numpy array): saves the M vector

    Returns:
        Matom [2D numpy array of [natom,3]]: saves the M belongs to each atom.
    """
    Mx, My, Mz = M
    atom_coords = ks.mol.atom_coords()
    natom = atom_coords.shape[0]
    weights = ks.grids.weights
    grids_coords = ks.grids.coords
    ngrid = weights.shape[-1]
    Matom = numpy.zeros((natom,3))
    for igrid in range(ngrid):
        Matom[grids2atom[igrid], 0]+= weights[igrid]*Mx[igrid]
        Matom[grids2atom[igrid], 1]+= weights[igrid]*My[igrid]
        Matom[grids2atom[igrid], 2]+= weights[igrid]*Mz[igrid]
        
    return Matom    


def mpuion(dmi, Si, natm = 3):
    # import pdb
    # pdb.set_trace()
    nao = dmi.shape[-1]//2
    dm_aa = dmi[:nao,:nao]
    dm_ab = dmi[:nao,nao:]
    dm_ba = dmi[nao:,:nao]
    dm_bb = dmi[nao:,nao:]
    dmx = dm_ab + dm_ba
    # dmy = (dm_ab - dm_ba)*1.0j
    dmy = ((dm_ab - dm_ba)*1.0j).real
    dmz = dm_aa - dm_bb
    dmt = dm_aa + dm_bb
    S = Si[:nao,:nao]
    nao_natm = nao//natm
    m = numpy.zeros((natm,3), dtype = numpy.complex128)
    rho = numpy.zeros((natm), dtype = numpy.complex128)
    # import pdb
    # pdb.set_trace()
    for iatm in range(natm):   
        naos = iatm*nao_natm            # number of nao starts
        naoe = (iatm+1)*nao_natm        # number of nao ends
        m[iatm][0] = numpy.einsum('uv,uv',dmx[:,naos:naoe],S[:,naos:naoe])
        m[iatm][1] = numpy.einsum('uv,uv',dmy[:,naos:naoe],S[:,naos:naoe])
        m[iatm][2] = numpy.einsum('uv,uv',dmz[:,naos:naoe],S[:,naos:naoe])
        rho[iatm]  = numpy.einsum('uv,uv',dmt[:,naos:naoe],S[:,naos:naoe])
    return m, rho

def hirshfield_partition(molu, coords, atoms):
    mf = scf.UHF(molu)
    mf = scf.addons.frac_occ(mf)
    mf.kernel()
    dmu = mf.make_rdm1()
    
    natm = atoms.shape[0]
    ngrid = coords.shape[0]
    rho_atom = numpy.zeros((natm,ngrid))
    weight_atom = numpy.zeros((natm,ngrid))
    
    ni = dft.numint.NumInt()
    for iatm in range(natm):
        mol_iatm = gto.Mole()
        mol_iatm.verbose = 6
        iatm_coords = ' ' + str(atoms[iatm][0]) + ' ' \
            + str(atoms[iatm][1]) + ' ' + str(atoms[iatm][2])
        mol_iatm.atom = molu.atom.split()[0] + iatm_coords
        mol_iatm.spin = molu.spin 
        mol_iatm.basis = molu.basis
        mol_iatm.symmetry=False 
        mol_iatm.build()
        ao = ni.eval_ao(mol_iatm, coords)
        
        rho_atom[iatm] = ni.eval_rho(mol_iatm, ao, dmu[0]+dmu[1])
    rho_pro = rho_atom.sum(axis=0)
    idx_zero = rho_pro < 1e-30
    for iatm in range(natm):
        with numpy.errstate(divide='ignore',invalid='ignore'):
            weight_atom[iatm] = rho_atom[iatm]/rho_pro
        weight_atom[iatm,idx_zero] = 0.0
        
    return weight_atom


