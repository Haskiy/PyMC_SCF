#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-05-18 10:15:56
LastEditTime: 2022-04-12 09:52:02
LastEditors: Li Hao
Description: 
    A general symmetry-adapted gks class,
    which other classes using symmetry should inherite this.
FilePath: \pyMC\gksmc\gksmc_r_symm.py

 May the force be with you!
'''


from functools import reduce

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import dhf
from pyscf.scf import chkfile
from pyscf.dft import rks
from pyMC.gksmc import numint_gksmc
from pyMC.gksmc import numint_gksmc_r
from pyMC.gksmc import gks_sym_general
from pyscf import __config__

# TODO : This part is haunted by the problem of code reusing, should be corrected.

def _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, orbsym, title='',
                    verbose=logger.DEBUG):
    """Print mo energy informations.
       mo energy should contains following information:
        1. mo energy is grouped by irreps.
        2. each irreps should contain both the PES and NES
    

    Args:
        mol (mole_symm type defined by pzc): inheriting mol class
        mo_energy (numpy array): MO energy
        mo_occ (numpy array): MO occupation
        ehomo (float): HOMO energy
        elumo (float): LUMO energy
        orbsym (list): orbital symmetry id.
        title (str, optional): title string. Defaults to ''.
        verbose ([type], optional): [description]. Defaults to logger.DEBUG.
    """
    log = logger.new_logger(mol, verbose)
    if round(ehomo,10) == round(elumo,10):
        log.warn('%s system HOMO %.15g == system LUMO %.15g',
                 title, ehomo, elumo)
    for i, ir in enumerate(mol.irrep_id):
        irname = mol.irrep_name[i]
        ir_idx = (orbsym == ir)
        nocc = numpy.count_nonzero(mo_occ[ir_idx])
        n2c = mo_occ[ir_idx].shape[0]//2
        e_ir = mo_energy[ir_idx]
        if nocc == 0:
            log.debug('%s%s nocc = 0', title, irname)
        else:
            log.debug('%s%s nocc = %d  HOMO = %.15g  LUMO = %.15g',
                      title, irname, nocc, e_ir[n2c+nocc-1], e_ir[n2c+nocc])
            if round(e_ir[n2c+nocc-1],10) > round(elumo,10):
                log.warn('%s%s HOMO %.15g > system LUMO %.15g',
                         title, irname, e_ir[n2c+nocc-1], elumo)
            if round(e_ir[n2c+nocc-1],10) == round(elumo,10):
                log.warn('%s%s HOMO %.15g == system LUMO %.15g',
                         title, irname, e_ir[n2c+nocc-1], elumo)
            if round(e_ir[n2c+nocc],10) < round(ehomo,10):
                log.warn('%s%s LUMO %.15g < system HOMO %.15g',
                         title, irname, e_ir[n2c+nocc], ehomo)
            if round(e_ir[n2c+nocc],10) == round(ehomo,10):
                log.warn('%s%s LUMO %.15g == system HOMO %.15g',
                         title, irname, e_ir[n2c+nocc], ehomo)
        log.debug('   mo_energy NES = %s', e_ir[:n2c])
        log.debug('   mo_energy PES = %s', e_ir[n2c:])


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
    t0 = numint_gksmc.timer_no_clock(ks, 'No meaning print test', *t0)
    
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

def get_grad(mo_coeff, mo_occ, fock_ao):
    '''
        DKS using symmetry-adapted SCF Gradients
    '''
    g = dhf.get_grad(mo_coeff, mo_occ, fock_ao)
    
    # * Folloing part do the symmetry correction.
    occidx = mo_occ > 0
    viridx = ~occidx
    orbsym = getattr(mo_coeff, 'orbsym')

    sym_forbid = orbsym[viridx].reshape(-1,1) != orbsym[occidx]
    g[sym_forbid.ravel()] = 0
    return g

class GKSMC_r_symm(rks.KohnShamDFT, dhf.UHF):
    '''
    Generalized Kohn-Sham with Tri-directions
    '''
    """
        It should be noted that, many parts should be modified for symmetry adapted Dirac4c
        1. build :Dirac4c--> self.opt
                 Symmetry--> initial fix electrons.
        2. eig :nearly exactly similar to symmetry-adapted GKS or GHF
        3. get_occ : this part is a combination of Dirac4c and Symmetry. For all the mo should be seperated,
            and grouped into different irreps. And NES should be paid special attention to.
        4. _dump_mo_energy : this part is a combination of Dirac4c and Symmetry.
        5. get_grad : Using the DKS get_gradient and 
        5._finalize 
        
    """
    def __init__(self, mol, xc='LDA,VWN'):
        gks_sym_general.GKS_symm.__init__(self, mol)
        dhf.UHF.__init__(self, mol)
        self.Dsymmetry = True
        self._numint = numint_gksmc_r.numint_gksmc_r()
        self.Ndirect = 1454
        # Principle directions. Default to be 100
        self.LIBXCT_factor = 1.0E-10
        self.MSL_factor = 0.999
        self.ncpu = None
        self.ibp = False
        # Add the new keys
        self._keys = self._keys.union(['Ndirect', 'LIBXCT_factor', 'MSL_factor', 
                                       'ncpu', 'ibp', 'Dsymmetry'])
        
    def dump_flags(self, verbose=None):
        dhf.UHF.dump_flags(self, verbose)
        gks_sym_general.GKS_symm.dump_flags(self, verbose)
        return self

    # ^ Following 2 lines for DHF calculations
    
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
    
    def build(self, mol=None):
        """Build a Symmetry adapted 4c calculation.

        Args:
            mol (mole_symm type defined by pzc): inheriting mol class
        
        """
        self.opt = None
        gks_sym_general.GKS_symm.build(self, mol)  
    
    def eig(self, h, s):
        """Overwitte the eig in original class in dhf.UHF with the Relativistic calculation.
           This subroutine is aiming at solve the generalised eigenvalue problem HC=SCE with specific symmetry.
           The fock matrix is diagonalised in each irrep. It should be NOTED that C and E is grouped by irrep.
           What's more, basis in one 2-dimensional irrep or higher-dimensional irrep is averaged.
           For example,
           C_{E_1} = C_{E_2} = (C_{E_1} + C_{E_2})/2

        Args:
            h (numpy array): fock matrix.
            s (numpy array): overlap matrix.

        Raises:
            ValueError: Dsymmetry should be used and symmetry should not be used.

        Returns:
            e (numpy array): eigenvaluse (orbital energies) are grouped in irreps.
            c (numpy array): mo coeff are grouped in irreps.
        """
        mol = self.mol
        if not mol.Dsymmetry or mol.symmetry:
            raise ValueError('It should be noted that only using double group should reach here.\
                Check the input!')
            
        s_ori = s.copy()

        nirrep = len(mol.symm_orb)
        symm_orb = mol.symm_orb
        s = [reduce(numpy.dot, (c.T.conj(),s,c)) for c in symm_orb]
        h = [reduce(numpy.dot, (c.T.conj(),h,c)) for c in symm_orb]
        for iequal in mol.equal_basis:
            h, s = gks_sym_general._equal_basis_irrep(h, s, mol.equal_basis[iequal])
        cs = []
        es = []
        orbsym = []
        # diagonalise in each irrep.
        for ir in range(nirrep):
            shift = numpy.eye(s[ir].shape[-1])*1.0E-15
            e, c = self._eigh(h[ir], s[ir] + shift)
            # * Grouped by symmetry!
            cs.append(c)
            es.append(e)
            orbsym.append([mol.irrep_id[ir]] * e.size)
        # average each basis in one irrep.
        e = numpy.hstack(es)
        c = gks_sym_general._so2ao_mo_coeff(symm_orb, cs)
        # * Note that the mo_coeff is tagged by symmetry id can using getattr(c, 'orbsym') get the irrep_id
        c = lib.tag_array(c, orbsym=numpy.hstack(orbsym))
        return e, c
    
    def get_occ(self, mo_energy=None, mo_coeff=None):
        """This is a very important subroutine to perform:
           1. get the occupation pattern.
           2. check whether to fix some orbitals.
           
           Because get_occ is working for symmetry adapted DKS calculations, thus,
           1. orbitals should be sorted and grouped by symmetry.
           2. NES should be sorted out and not be occupied.

        Args:
            mo_energy (numpy.array, optional): mo energy. Defaults to None.
            mo_coeff (numpy.array, optional): mo coeff. Defaults to None.

        Raises:
            ValueError: not using Dsymmetry or using the original keyword symmetry

        Returns:
            mo_occ (numpy.array): mo_occ pattern.
        """
        # Info prepare.
        # ! mo_energy is grouped as symmetry.
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        c = lib.param.LIGHT_SPEED
        n4c = len(mo_energy)
        n2c = n4c // 2
        if not mol.Dsymmetry or mol.symmetry:
            raise ValueError('It should be noted that Dsymmetry not symmetry should be used to\
                            implement double group symmetry!')

        orbsym = getattr(mo_coeff, 'orbsym')
        mo_occ = numpy.zeros_like(mo_energy)
        rest_idx = numpy.ones(mo_occ.size, dtype=bool)
        nelec_fix = 0

        # * This part sort out the nooc of different irreps for pre-determinant parts.
        for i, ir in enumerate(mol.irrep_id):
            irname = mol.irrep_name[i]
            ir_idx = numpy.where(orbsym == ir)[0]
            # * This part fix the number of electrons in one irrep.
            if irname in self.irrep_nelec:
                n = self.irrep_nelec[irname]
                occ_sort_all = numpy.argsort(mo_energy[ir_idx].round(9), kind='mergesort')
                occ_sort = occ_sort_all[mo_energy[ir_idx].shape[-1]//2:]
                occ_idx  = ir_idx[occ_sort[:n]]
                mo_occ[occ_idx] = 1
                nelec_fix += n
                rest_idx[ir_idx] = False
        nelec_float = mol.nelectron - nelec_fix
        assert(nelec_float >= 0)
        
        # * get the occ. pattern if no fix electrons!
        occ_sort_all = numpy.argsort(mo_energy[rest_idx].round(9), kind='mergesort')
        if nelec_float > 0:
            rest_idx = numpy.where(rest_idx)[0]
            occ_sort_all = numpy.argsort(mo_energy[rest_idx].round(9), kind='mergesort')
            occ_sort = occ_sort_all[n2c:] 
            occ_idx  = rest_idx[occ_sort[:nelec_float]]
            mo_occ[occ_idx] = 1

        vir_idx = (mo_occ==0)
        # vir_idx_NES is the ids of unoccupied 
        occ_sort_all_orbiatl = numpy.argsort(mo_energy.round(9), kind='mergesort')
        vir_idx_NES = numpy.array([i for i in numpy.where(vir_idx)[0].tolist() if i not in occ_sort_all_orbiatl[n2c:].tolist()])
        vir_idx_PES = numpy.array([i for i in numpy.where(vir_idx)[0].tolist() if i not in vir_idx_NES])
        if self.verbose >= logger.INFO and numpy.count_nonzero(vir_idx) > 0:
            ehomo = max(mo_energy[~vir_idx])
            elumo = min(mo_energy[vir_idx_PES]) # ! LUMO should be the PES
            noccs = []
            for i, ir in enumerate(mol.irrep_id):
                irname = mol.irrep_name[i]
                ir_idx = (orbsym == ir)

                noccs.append(int(mo_occ[ir_idx].sum()))
                if ehomo in mo_energy[ir_idx]:
                    irhomo = irname
                if elumo in mo_energy[ir_idx]:
                    irlumo = irname
            logger.info(self, 'HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                        irhomo, ehomo, irlumo, elumo)

            logger.debug(self, 'irrep_nelec = %s', noccs)
            _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, orbsym,
                                    verbose=self.verbose)

        return mo_occ
    
    def get_grad(self, mo_coeff, mo_occ, fock=None):
        """Get the gradient of symmetry-adapted Dirac SCF

        Args:
            mo_coeff (numpy.array with tag): mo coeff
            mo_occ (numpy.array): mo occ
            fock (numpy.array, optional): Fock matrix. Defaults to None.

        Returns:
            get_grad (subroutine): calculate the gradients
        """
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)
    
    def _finalize(self):
        """Finish the SCF calculation

        Returns:
            return the class
        """
        dhf.UHF._finalize(self)

        n2c = self.mo_energy.shape[-1]//2
        # Using mergesort because it is stable. We don't want to change the
        # ordering of the symmetry labels when two orbitals are degenerated.
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ> 0].round(9), kind='mergesort')
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0].round(9), kind='mergesort')
        # * get the offset used in Dirac4c calculations.
        v_NES = v_sort[:n2c]
        v_PES = v_sort[n2c:]
        orbsym = getattr(self.mo_coeff,'orbsym')
        self.mo_energy = numpy.hstack((self.mo_energy[self.mo_occ==0][v_NES],
                                       self.mo_energy[self.mo_occ> 0][o_sort],
                                       self.mo_energy[self.mo_occ==0][v_PES]))
        self.mo_coeff = numpy.hstack((self.mo_coeff[:,self.mo_occ==0].take(v_NES, axis=1),
                                      self.mo_coeff[:,self.mo_occ> 0].take(o_sort, axis=1),
                                      self.mo_coeff[:,self.mo_occ==0].take(v_PES, axis=1)))
        orbsym = numpy.hstack((orbsym[self.mo_occ==0][v_NES],
                               orbsym[self.mo_occ> 0][o_sort],
                               orbsym[self.mo_occ==0][v_PES]))
        self.mo_coeff = lib.tag_array(self.mo_coeff, orbsym=orbsym)
        nocc = len(o_sort)
        self.mo_occ[:n2c] = 0
        self.mo_occ[n2c:n2c+nocc] = 1
        self.mo_occ[n2c+nocc:] = 0
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                             self.mo_coeff, self.mo_occ, overwrite_mol=False)
        return self
        

    def nuc_grad_method(self):
        raise NotImplementedError
    