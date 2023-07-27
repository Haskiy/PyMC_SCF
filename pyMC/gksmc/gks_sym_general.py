#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-04-29 15:51:40
LastEditTime: 2022-04-12 09:51:44
LastEditors: Li Hao
Description: 
    A general symmetry-adapted gks class, 
    which other classes using symmetry should inherite this.
FilePath: \pyMC\gksmc\gks_sym_general.py

 May the force be with you!
'''

from functools import reduce

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ghf, ghf_symm
from pyscf.scf import hf_symm
from pyscf.scf import chkfile
from pyscf.dft import rks, gks
from pyMC.gksmc import numint_gksmc
from pyMC.gksmc import rks_gksmc, gksmc
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
MO_BASE = getattr(__config__, 'MO_BASE', 1)

def spin_square(mo, s=1):
    r'''Spin of the GHF wavefunction

    .. math::

        S^2 = \frac{1}{2}(S_+ S_-  +  S_- S_+) + S_z^2

    where :math:`S_+ = \sum_i S_{i+}` is effective for all beta occupied
    orbitals; :math:`S_- = \sum_i S_{i-}` is effective for all alpha occupied
    orbitals.

    1. There are two possibilities for :math:`S_+ S_-`
        1) same electron :math:`S_+ S_- = \sum_i s_{i+} s_{i-}`,

        .. math::

            \sum_i \langle UHF|s_{i+} s_{i-}|UHF\rangle
             = \sum_{pq}\langle p|s_+s_-|q\rangle \gamma_{qp} = n_\alpha

        2) different electrons :math:`S_+ S_- = \sum s_{i+} s_{j-},  (i\neq j)`.
        There are in total :math:`n(n-1)` terms.  As a two-particle operator,

        .. math::

            \langle S_+ S_- \rangle
            =\sum_{ij}(\langle i^\alpha|i^\beta\rangle \langle j^\beta|j^\alpha\rangle
            - \langle i^\alpha|j^\beta\rangle \langle j^\beta|i^\alpha\rangle)

    2. Similarly, for :math:`S_- S_+`
        1) same electron

        .. math::

           \sum_i \langle s_{i-} s_{i+}\rangle = n_\beta

        2) different electrons

        .. math::

            \langle S_- S_+ \rangle
            =\sum_{ij}(\langle i^\beta|i^\alpha\rangle \langle j^\alpha|j^\beta\rangle
            - \langle i^\beta|j^\alpha\rangle \langle j^\alpha|i^\beta\rangle)

    3. For :math:`S_z^2`
        1) same electron

        .. math::

            \langle s_z^2\rangle = \frac{1}{4}(n_\alpha + n_\beta)

        2) different electrons

        .. math::

            &\sum_{ij}(\langle ij|s_{z1}s_{z2}|ij\rangle
                      -\langle ij|s_{z1}s_{z2}|ji\rangle) \\
            &=\frac{1}{4}\sum_{ij}(\langle i^\alpha|i^\alpha\rangle \langle j^\alpha|j^\alpha\rangle
             - \langle i^\alpha|i^\alpha\rangle \langle j^\beta|j^\beta\rangle
             - \langle i^\beta|i^\beta\rangle \langle j^\alpha|j^\alpha\rangle
             + \langle i^\beta|i^\beta\rangle \langle j^\beta|j^\beta\rangle) \\
            &-\frac{1}{4}\sum_{ij}(\langle i^\alpha|j^\alpha\rangle \langle j^\alpha|i^\alpha\rangle
             - \langle i^\alpha|j^\alpha\rangle \langle j^\beta|i^\beta\rangle
             - \langle i^\beta|j^\beta\rangle \langle j^\alpha|i^\alpha\rangle
             + \langle i^\beta|j^\beta\rangle\langle j^\beta|i^\beta\rangle) \\
            &=\frac{1}{4}\sum_{ij}|\langle i^\alpha|i^\alpha\rangle - \langle i^\beta|i^\beta\rangle|^2
             -\frac{1}{4}\sum_{ij}|\langle i^\alpha|j^\alpha\rangle - \langle i^\beta|j^\beta\rangle|^2 \\
            &=\frac{1}{4}(n_\alpha - n_\beta)^2
             -\frac{1}{4}\sum_{ij}|\langle i^\alpha|j^\alpha\rangle - \langle i^\beta|j^\beta\rangle|^2

    Args:
        mo : a list of 2 ndarrays
            Occupied alpha and occupied beta orbitals

    Kwargs:
        s : ndarray
            AO overlap

    Returns:
        A list of two floats.  The first is the expectation value of S^2.
        The second is the corresponding 2S+1

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', charge=1, spin=1, verbose=0)
    >>> mf = scf.UHF(mol)
    >>> mf.kernel()
    -75.623975516256706
    >>> mo = (mf.mo_coeff[0][:,mf.mo_occ[0]>0], mf.mo_coeff[1][:,mf.mo_occ[1]>0])
    >>> print('S^2 = %.7f, 2S+1 = %.7f' % spin_square(mo, mol.intor('int1e_ovlp_sph')))
    S^2 = 0.7570150, 2S+1 = 2.0070027
    '''
    nao = mo.shape[0] // 2
    if isinstance(s, numpy.ndarray):
        assert(s.size == nao**2 or numpy.allclose(s[:nao,:nao], s[nao:,nao:]))
        s = s[:nao,:nao]
    mo_a = mo[:nao]
    mo_b = mo[nao:]
    saa = reduce(numpy.dot, (mo_a.conj().T, s, mo_a))
    sbb = reduce(numpy.dot, (mo_b.conj().T, s, mo_b))
    sab = reduce(numpy.dot, (mo_a.conj().T, s, mo_b))
    sba = sab.conj().T
    nocc_a = saa.trace()
    nocc_b = sbb.trace()
    ssxy = (nocc_a+nocc_b) * .5
    ssxy+= sba.trace() * sab.trace() - numpy.einsum('ij,ji->', sba, sab)
    ssz  = (nocc_a+nocc_b) * .25
    ssz += (nocc_a-nocc_b)**2 * .25
    tmp  = saa - sbb
    ssz -= numpy.einsum('ij,ji', tmp, tmp) * .25
    print('0.5*(S_+ S_-  +  S_- S_+) = %.8f  S_z^2 = %.8f' %(ssxy.real, ssz.real))
    ss = (ssxy + ssz).real
    s = numpy.sqrt(ss+.25) - .5
    return ss, s*2+1


def _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, orbsym, title='',
                    verbose=logger.DEBUG):
    """Print mo energy informations, which should be more attention to.

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
        nso = numpy.count_nonzero(ir_idx)
        nocc = numpy.count_nonzero(mo_occ[ir_idx])
        e_ir = mo_energy[ir_idx]
        if nocc == 0:
            log.debug('%s%s nocc = 0', title, irname)
        elif nocc == nso:
            log.debug('%s%s nocc = %d  HOMO = %.15g',
                      title, irname, nocc, e_ir[nocc-1])
        else:
            log.debug('%s%s nocc = %d  HOMO = %.15g  LUMO = %.15g',
                      title, irname, nocc, e_ir[nocc-1], e_ir[nocc])
            if round(e_ir[nocc-1],10) > round(elumo,10):
                log.warn('%s%s HOMO %.15g > system LUMO %.15g',
                         title, irname, e_ir[nocc-1], elumo)
            if round(e_ir[nocc-1],10) == round(elumo,10):
                log.warn('%s%s HOMO %.15g == system LUMO %.15g',
                         title, irname, e_ir[nocc-1], elumo)
            if round(e_ir[nocc],10) < round(ehomo,10):
                log.warn('%s%s LUMO %.15g < system HOMO %.15g',
                         title, irname, e_ir[nocc], ehomo)
            if round(e_ir[nocc],10) == round(ehomo,10):
                log.warn('%s%s LUMO %.15g == system HOMO %.15g',
                         title, irname, e_ir[nocc], ehomo)
        log.debug('   mo_energy = %s', e_ir)


def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    mol = mf.mol
    if not mol.Dsymmetry:
        raise ValueError("Dsymmetry should be used!")

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    ovlp_ao = mf.get_ovlp()
    log = logger.new_logger(mf, verbose)
    if log.verbose >= logger.NOTE:
        nirrep = len(mol.irrep_id)
        orbsym = mf.get_orbsym(mo_coeff, ovlp_ao)
        wfnsym = 0
        noccs = [sum(orbsym[mo_occ>0]==ir) for ir in mol.irrep_id]
        log.note('total symmetry = %s', mol.id_2_irrep(mol.groupname, wfnsym))
        log.note('occupancy for each irrep:  ' + (' %4s'*nirrep), *mol.irrep_name)
        log.note('double occ                 ' + (' %4d'*nirrep), *noccs)
        log.note('**** MO energy ****')
        irname_full = {}
        for k,ir in enumerate(mol.irrep_id):
            irname_full[ir] = mol.irrep_name[k]
        irorbcnt = {}
        for k, j in enumerate(orbsym):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            log.note('MO #%d (%s #%d), energy= %.15g occ= %g',
                     k+MO_BASE, irname_full[j], irorbcnt[j], mo_energy[k],
                     mo_occ[k])

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    dip = mf.dip_moment(mol, dm, verbose=log)
    if with_meta_lowdin:
        pop_and_chg = mf.mulliken_meta(mol, dm, s=ovlp_ao, verbose=log)
    else:
        pop_and_chg = mf.mulliken_pop(mol, dm, s=ovlp_ao, verbose=log)
    return pop_and_chg, dip

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix in occupied, virtual
    subspaces separatedly (without change occupancy).
    '''
    mol = mf.mol
    if not mol.Dsymmetry:
        raise ValueError("Dsymmetry should be used!")

    if getattr(mo_coeff, 'orbsym', None) is not None:
        return hf_symm.canonicalize(mf, mo_coeff, mo_occ, fock)
    else:
        raise NotImplementedError

# ! It should be noted that, at present, ONLY diagonalizing FC = CE is using symmetry.
# TODO : More functions like pyscf.scf.ghf_symm.py

def _so2ao_mo_coeff(symm_orb,C):
    """This subroutine gets the full mo_coeff from ao basis to mo basis.
        Which involves the D_{ao so}@C_{so mo}.
        This subroutine is aimming at D_{ao so}@C_{so mo}.

    Args:
        symm_orb (list): Saving the D_{ao so} in each irrep.    
        C (list): Saving the C_{ao so} in each irrep.

    Returns:
        A numpy.array : Full ao2mo coeff.
    """
    return numpy.hstack([symm_orb[ir]@C[ir] for ir in range(symm_orb.__len__())])
    # return numpy.hstack([numpy.dot(symm_orb[ir],C[ir]) \
    #                             for ir in range(symm_orb.__len__())])

def _equal_basis_irrep(f, s, eq_table):
    """This subroutine is aimming at average the different basis in one multi-dimensional irreps to
       get a more symmetry-averaged solution for diagonalising the Fock matrix.

    Args:
        f (list): A list of numpy arrays, the length of f is equal to the number of basis,
            saving the Fock matrix in each basis.
        s (list): A list of numpy arrays, saving the overlap matrix of each basis.
        eq_table (tuple): saving the irrep_id of basis belonging to the same multi-dimensional irrep.

    Returns:
        f (list): after averaged
        s (list): after averaged
    """
    neq = eq_table.__len__()
    f_ave = reduce(lambda x,y : x+y, [f[i] for i in eq_table])/neq
    s_ave = reduce(lambda x,y : x+y, [s[i] for i in eq_table])/neq
    # import pdb
    # pdb.set_trace()
    for ibasis in eq_table:
        f[ibasis] = f_ave
        s[ibasis] = s_ave
        
    return f, s


class GKS_symm(rks_gksmc.KohnShamDFT_MD, ghf_symm.GHF):
    '''Generalized Kohn-Sham with Tri-directions'''
    
    def __init__(self, mol, xc='LDA,VWN'):
        ghf.GHF.__init__(self, mol)
        rks_gksmc.KohnShamDFT_MD.__init__(self, xc)
        self.irrep_nelec = {}
        self._keys = self._keys.union(['irrep_nelec'])
        
    def dump_flags(self, verbose=None):
        ghf.GHF.dump_flags(self, verbose)
        if self.irrep_nelec:
            logger.info(self, 'irrep_nelec %s', self.irrep_nelec)
        return self
        
    def build(self, mol=None):
        
        if mol is None: mol = self.mol
        if mol.symmetry:
            raise ValueError("")
        if mol.Dsymmetry:
            for irname in self.irrep_nelec:
                if irname not in mol.irrep_name:
                    logger.warn(self, 'Molecule does not have irrep %s', irname)

            nelec_fix = self.irrep_nelec.values()
            if any(isinstance(x, (tuple, list)) for x in nelec_fix):
                msg =('Number of alpha/beta electrons cannot be assigned '
                      'separately in GHF.  irrep_nelec = %s' % self.irrep_nelec)
                raise ValueError(msg)
            nelec_fix = sum(nelec_fix)
            float_irname = set(mol.irrep_name) - set(self.irrep_nelec)
            if nelec_fix > mol.nelectron:
                msg =('More electrons defined by irrep_nelec than total num electrons. '
                      'mol.nelectron = %d  irrep_nelec = %s' %
                      (mol.nelectron, self.irrep_nelec))
                raise ValueError(msg)
            else:
                logger.info(mol, 'Freeze %d electrons in irreps %s',
                            nelec_fix, self.irrep_nelec.keys())

            if len(float_irname) == 0 and nelec_fix != mol.nelectron:
                msg =('Num electrons defined by irrep_nelec != total num electrons. '
                      'mol.nelectron = %d  irrep_nelec = %s' %
                      (mol.nelectron, self.irrep_nelec))
                raise ValueError(msg)
            else:
                logger.info(mol, '    %d free electrons in irreps %s',
                            mol.nelectron-nelec_fix, ' '.join(float_irname))
        return ghf.GHF.build(self, mol)
    
    def eig(self, h, s):
        """Overwitte the eig in original class.
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

        nirrep = len(mol.symm_orb)
        symm_orb = mol.symm_orb
        s = [reduce(numpy.dot, (c.T.conj(),s,c)) for c in symm_orb]
        h = [reduce(numpy.dot, (c.T.conj(),h,c)) for c in symm_orb]
        for iequal in mol.equal_basis:
            h, s = _equal_basis_irrep(h, s, mol.equal_basis[iequal])
        cs = []
        es = []
        orbsym = []
        # diagonalise in each irrep.
        for ir in range(nirrep):
            e, c = self._eigh(h[ir], s[ir])
            # * Grouped by symmetry!
            cs.append(c)
            es.append(e)
            orbsym.append([mol.irrep_id[ir]] * e.size)
        # average each basis in one irrep.
        e = numpy.hstack(es)
        c = _so2ao_mo_coeff(symm_orb, cs)
        # * Note that the mo_coeff is tagged by symmetry id can using getattr(c, 'orbsym') get the irrep_id
        c = lib.tag_array(c, orbsym=numpy.hstack(orbsym))
        return e, c

    def get_occ(self, mo_energy=None, mo_coeff=None):
        """This is a very important subroutine to perform:
           1. get the occupation pattern.
           2. check whether to fix some orbitals

        Args:
            mo_energy (numpy.array, optional): mo energy. Defaults to None.
            mo_coeff (numpy.array, optional): mo coeff. Defaults to None.

        Raises:
            ValueError: not using Dsymmetry or using the original keyword symmetry

        Returns:
            mo_occ (numpy.array): mo_occ pattern.
        """
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        if not mol.Dsymmetry or mol.symmetry:
            raise ValueError('It should be noted that Dsymmetry not symmetry should be used to\
                            implement double group symmetry!')

        orbsym = getattr(mo_coeff, 'orbsym')
        mo_occ = numpy.zeros_like(mo_energy)
        rest_idx = numpy.ones(mo_occ.size, dtype=bool)
        nelec_fix = 0

        for i, ir in enumerate(mol.irrep_id):
            irname = mol.irrep_name[i]
            ir_idx = numpy.where(orbsym == ir)[0]
            # * This part fix the number of electrons in one irrep.
            if irname in self.irrep_nelec:
                n = self.irrep_nelec[irname]
                occ_sort = numpy.argsort(mo_energy[ir_idx].round(9), kind='mergesort')
                occ_idx  = ir_idx[occ_sort[:n]]
                mo_occ[occ_idx] = 1
                nelec_fix += n
                rest_idx[ir_idx] = False
        nelec_float = mol.nelectron - nelec_fix
        assert(nelec_float >= 0)
        # * get the occ. pattern !
        if nelec_float > 0:
            rest_idx = numpy.where(rest_idx)[0]
            occ_sort = numpy.argsort(mo_energy[rest_idx].round(9), kind='mergesort')
            occ_idx  = rest_idx[occ_sort[:nelec_float]]
            mo_occ[occ_idx] = 1

        vir_idx = (mo_occ==0)
        if self.verbose >= logger.INFO and numpy.count_nonzero(vir_idx) > 0:
            ehomo = max(mo_energy[~vir_idx])
            elumo = min(mo_energy[ vir_idx])
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

            if mo_coeff is not None and self.verbose >= logger.DEBUG:
                ss, s = spin_square(mo_coeff[:,mo_occ>0], self.get_ovlp())
                logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ
    
    def get_grad(self, mo_coeff, mo_occ, fock=None):
        g = ghf.GHF.get_grad(self, mo_coeff, mo_occ, fock)
        if self.mol.Dsymmetry:
            occidx = mo_occ > 0
            viridx = ~occidx
            orbsym = getattr(mo_coeff, 'orbsym')
            sym_forbid = orbsym[viridx].reshape(-1,1) != orbsym[occidx]
            g[sym_forbid.ravel()] = 0
        return g
    
    def _finalize(self):
        ghf.GHF._finalize(self)

        # Using mergesort because it is stable. We don't want to change the
        # ordering of the symmetry labels when two orbitals are degenerated.
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ> 0].round(9), kind='mergesort')
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0].round(9), kind='mergesort')
        orbsym = self.get_orbsym(self.mo_coeff, self.get_ovlp())
        self.mo_energy = numpy.hstack((self.mo_energy[self.mo_occ> 0][o_sort],
                                       self.mo_energy[self.mo_occ==0][v_sort]))
        self.mo_coeff = numpy.hstack((self.mo_coeff[:,self.mo_occ> 0].take(o_sort, axis=1),
                                      self.mo_coeff[:,self.mo_occ==0].take(v_sort, axis=1)))
        orbsym = numpy.hstack((orbsym[self.mo_occ> 0][o_sort],
                               orbsym[self.mo_occ==0][v_sort]))
        self.mo_coeff = lib.tag_array(self.mo_coeff, orbsym=orbsym)
        nocc = len(o_sort)
        self.mo_occ[:nocc] = 1
        self.mo_occ[nocc:] = 0
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                             self.mo_coeff, self.mo_occ, overwrite_mol=False)
        return self
    
    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None, s=None):
        if mol is None: mol = self.mol
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if s is None: s = self.get_ovlp()
        return hf_symm.get_irrep_nelec(mol, mo_coeff, mo_occ, s)
    

class GKSM(GKS_symm):
    '''Generalized Kohn-Sham with Tri-directions'''
    def __init__(self, mol, xc='LDA,VWN', toque_bxc = True):
        GKS_symm.__init__(self, mol)
        # gl._init()
        self._toque_bxc = toque_bxc
        self._molu = None
        self._group = 'D5'
        self._average = 'vxc'
        self._pcycle = 0
        
    def dump_flags(self, verbose=None):
        GKS_symm.dump_flags(self, verbose)
        rks_gksmc.KohnShamDFT_MD.dump_flags(self, verbose)
        return self
    
    def calculate_Bxc(self, mol, dm):
        if dm == None:
            dmtot = self.make_rdm1()
            

    get_veff = gks.get_veff
    energy_elec = rks.energy_elec

    def nuc_grad_method(self):
        raise NotImplementedError
        
    @property
    def molu(self):
        return self._molu
    @molu.setter
    def molu(self, molu_inp):
        self._molu = molu_inp
        
    @property
    def group(self):
        return self._group
    @group.setter
    def group(self, group_inp):
        self._group = group_inp
        
    @property
    def average(self):
        return self._average
    @average.setter
    def average(self, average_inp):
        self._average = average_inp
        
    @property
    def pcycle(self):
        return self._pcycle
    @pcycle.setter
    def pcycle(self, pcycle_inp):
        self._pcycle = pcycle_inp

    @property
    def toque_bxc(self):
        return self._toque_bxc
    @toque_bxc.setter
    def toque_bxc(self,toque_bxc_input):
        self._toque_bxc = toque_bxc_input

