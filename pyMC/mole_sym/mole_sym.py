#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-04-27 15:00:39
LastEditTime: 2023-05-24 02:06:40
LastEditors: Li Hao
Description: 
    A new class to do double group symmetry-adapted calculations.
FilePath: /pyMC/mole_sym/mole_sym.py

 May the force be with you!
'''


import os, sys
import types
import re
import platform
import gc
import time
import json
import ctypes
import numpy
import h5py
import scipy.special
import scipy.linalg
from pyscf import lib
from pyscf.lib import param
from pyscf.data import elements
from pyscf.lib import logger
from pyscf.gto import cmd_args
from pyscf.gto import basis
from pyscf.gto import moleintor
from pyscf.gto.eval_gto import eval_gto
from pyscf.gto.ecp import core_configuration
from pyscf import __config__
from functools import reduce
from pyscf import gto
import warnings
from pyscf.scf import hf
from pyscf.scf import uhf
from pyscf.scf import chkfile
from pyMC.lib import group_info
from pyMC.mole_sym import orbital_symm_utils


def dumps(mol):
    '''Serialize Mole object to a JSON formatted str.
    '''
    exclude_keys = set(('output', 'stdout', '_keys', 'ovlp', 
                        # Constructing in function loads
                        'symm_orb', 'irrep_id', 'irrep_name'))
    nparray_keys = set(('_atm', '_bas', '_env', '_ecpbas',
                        '_symm_orig', '_symm_axes'))

    moldic = dict(mol.__dict__)
    for k in exclude_keys:
        del(moldic[k])
    for k in nparray_keys:
        if isinstance(moldic[k], numpy.ndarray):
            moldic[k] = moldic[k].tolist()
    moldic['atom'] = repr(mol.atom)
    moldic['basis']= repr(mol.basis)
    moldic['ecp' ] = repr(mol.ecp)

    try:
        return json.dumps(moldic)
    except TypeError:
        def skip_value(dic):
            dic1 = {}
            for k,v in dic.items():
                if (v is None or
                    isinstance(v, (str, gto.mole.unicode, bool, int, float))):
                    dic1[k] = v
                elif isinstance(v, (list, tuple)):
                    dic1[k] = v   # Should I recursively skip_vaule?
                elif isinstance(v, set):
                    dic1[k] = list(v)
                elif isinstance(v, dict):
                    dic1[k] = skip_value(v)
                else:
                    msg =('Function mol.dumps drops attribute %s because '
                          'it is not JSON-serializable' % k)
                    warnings.warn(msg)
            return dic1
        return json.dumps(skip_value(moldic), skipkeys=True)

class Mole_sym(gto.mole.Mole):
    """A Mole class used in double group symmetry calculations.

    Args:
        gto ([type]): [description]
    """
    def __init__(self):
        
        gto.mole.Mole.__init__(self)
        # self.Dsymmetry controls whether to do double group symmetry calculations.
        # ! Dsymmetry must be TRUE, for if you want do no-symmetry calculations. Why don't you use pyscf.gto.Mole ?
        self.Dsymmetry = False
        # symm_threshold for projecting. For most cases, this should not be changed !
        self.symm_threshold = 1.0E-8
        # ovlp for overlap matrix. Used in checking the linear-dependence of the projectd basis.
        self.ovlp = None
        # equal_basis: a table of equivelent basis belonging to the same multi-dimensional irreps.
        self.equal_basis = None
        # id_2_irrep: irrep id to irrep name
        self.id_2_irrep = None
        # dirac4c: whether do 4c calculations of nr calculations, WHEN USING DOUBLE GROUP SYMMETRY.
        self.dirac4c = False
        # vortex: Dirac4c calculations can perform the vortex like M pattern, this pattern has its own group.
        #         This attribute determines whether to perform vortex like pattern.
        self.vortex = False
        # * Add new attributes Dsymmetry
        keys_new = set(('Dsymmetry', 'verbose', 'symm_threshold', 'ovlp', 'id_2_irrep', 'dirac4c', 'vortex'))
        self._keys = set(self.__dict__.keys()).union(keys_new)
    
    dumps = dumps
        
    def build(self, singleatom = None, dump_input=True, parse_arg=True,
              verbose=None, output=None, max_memory=None,
              atom=None, basis=None, unit=None, nucmod=None, ecp=None,
              charge=None, spin=0, symmetry=None, symmetry_subgroup=None,
              cart=None):
        '''Setup moleclue and initialize some control parameters.  Whenever you
        change the value of the attributes of :class:`Mole`, you need call
        this function to refresh the internal data of Mole.

        Kwargs:
            singleatom : Mole class 
                which is used in projecting the orbital.
            dump_input : bool
                whether to dump the contents of input file in the output file
            parse_arg : bool
                whether to read the sys.argv and overwrite the relevant parameters
            verbose : int
                Print level.  If given, overwrite :attr:`Mole.verbose`
            output : str or None
                Output file.  If given, overwrite :attr:`Mole.output`
            max_memory : int, float
                Allowd memory in MB.  If given, overwrite :attr:`Mole.max_memory`
            atom : list or str
                To define molecluar structure.
            basis : dict or str
                To define basis set.
            nucmod : dict or str
                Nuclear model.  If given, overwrite :attr:`Mole.nucmod`
            charge : int
                Charge of molecule. It affects the electron numbers
                If given, overwrite :attr:`Mole.charge`
            spin : int
                2S, num. alpha electrons - num. beta electrons to control
                multiplicity. If setting spin = None , multiplicity will be
                guessed based on the neutral molecule.
                If given, overwrite :attr:`Mole.spin`
            symmetry : bool or str
                Whether to use symmetry.  If given a string of point group
                name, the given point group symmetry will be used.

        '''
        gc.collect()  # To release circular referred objects
        # import pdb
        # pdb.set_trace()
        if isinstance(dump_input, (str, gto.mole.unicode)):
            sys.stderr.write('Assigning the first argument %s to mol.atom\n' %
                             dump_input)
            dump_input, atom = True, dump_input
        
        if singleatom == None:
            raise ValueError('No gto.mole object of single atom')

        if verbose is not None: self.verbose = verbose
        if output is not None: self.output = output
        if max_memory is not None: self.max_memory = max_memory
        if atom is not None: self.atom = atom
        if basis is not None: self.basis = basis
        if unit is not None: self.unit = unit
        if nucmod is not None: self.nucmod = nucmod
        if ecp is not None: self.ecp = ecp
        if charge is not None: self.charge = charge
        if spin != 0: self.spin = spin
        if symmetry is not None: self.symmetry = symmetry
        if symmetry_subgroup is not None: self.symmetry_subgroup = symmetry_subgroup
        if cart is not None: self.cart = cart
        # TODO : add a new subroutine to check whether the input is correct for symmetry-adapted calculations.

        if parse_arg:
            gto.mole._update_from_cmdargs_(self)

        # avoid opening output file twice
        if (self.output is not None
            # StringIO() does not have attribute 'name'
            and getattr(self.stdout, 'name', None) != self.output):

            if self.verbose > logger.QUIET:
                if os.path.isfile(self.output):
                    print('overwrite output file: %s' % self.output)
                else:
                    print('output file: %s' % self.output)

            if self.output == '/dev/null':
                self.stdout = open(os.devnull, 'w')
            else:
                self.stdout = open(self.output, 'w')

        if self.verbose >= logger.WARN:
            self.check_sanity()

        self._atom = self.format_atom(self.atom, unit=self.unit)
        uniq_atoms = set([a[0] for a in self._atom])

        if isinstance(self.basis, (str, gto.mole.unicode, tuple, list)):
            # specify global basis for whole molecule
            _basis = dict(((a, self.basis) for a in uniq_atoms))
        elif 'default' in self.basis:
            default_basis = self.basis['default']
            _basis = dict(((a, default_basis) for a in uniq_atoms))
            _basis.update(self.basis)
            del(_basis['default'])
        else:
            _basis = self.basis
        self._basis = self.format_basis(_basis)

# TODO: Consider ECP info in point group symmetry initialization
        if self.ecp:
            # Unless explicitly input, ECP should not be assigned to ghost atoms
            if isinstance(self.ecp, (str, gto.mole.unicode)):
                _ecp = dict([(a, str(self.ecp))
                             for a in uniq_atoms if not gto.mole.is_ghost_atom(a)])
            elif 'default' in self.ecp:
                default_ecp = self.ecp['default']
                _ecp = dict(((a, default_ecp)
                             for a in uniq_atoms if not gto.mole.is_ghost_atom(a)))
                _ecp.update(self.ecp)
                del(_ecp['default'])
            else:
                _ecp = self.ecp
            self._ecp = self.format_ecp(_ecp)

        env = self._env[:gto.mole.PTR_ENV_START]
        self._atm, self._bas, self._env = \
                self.make_env(self._atom, self._basis, env, self.nucmod,
                              self.nucprop)
        self._atm, self._ecpbas, self._env = \
                self.make_ecp_env(self._atm, self._ecp, self._env)

        if self.spin is None:
            self.spin = self.nelectron % 2
        else:
            # Access self.nelec in which the code checks whether the spin and
            # number of electrons are consistent.
            self.nelec

        if self.symmetry:
            raise ValueError('It should be noted that, symmetry should not be used!')
        if self.Dsymmetry:    
            from pyscf import symm
            self.topgroup, orig, axes = symm.detect_symm(self._atom, self._basis)

            if isinstance(self.symmetry, (str, gto.mole.unicode)):
                self.symmetry = str(symm.std_symb(self.symmetry))
                axes = symm.subgroup(self.symmetry, axes)[1]
                self.groupname = group_info.SGROUP2D[self.topgroup]
            else:
                axes = symm.as_subgroup(self.topgroup, axes,
                                                        self.symmetry_subgroup)[1]
                self.groupname = group_info.SGROUP2D[self.topgroup]
            if self.vortex:
                # if not self.dirac4c:
                #     raise ValueError("It should be noted that to get the vortex-like pattern, \
                #         it's better to use Dirac 4c calculations, not GKS")
                self.groupname = group_info.GROUP_2_VORTEX_GROUP[self.groupname]
            self._symm_orig = orig
            self._symm_axes = axes

            if self.cart and self.groupname in ('Dooh', 'Coov'):
                raise NotImplementedError('Dooh or Coov double group is not implemented.')
                if self.groupname == 'Dooh':
                    self.groupname, lgroup = 'D2h', 'Dooh'
                else:
                    self.groupname, lgroup = 'C2v', 'Coov'
                logger.warn(self, 'This version does not support linear molecule '
                            'symmetry %s for cartesian GTO basis.  Its subgroup '
                            '%s is used', lgroup, self.groupname)
                
        

        if dump_input and not self._built and self.verbose > logger.NOTE:
            self.dump_input()

        logger.debug3(self, 'arg.atm = %s', str(self._atm))
        logger.debug3(self, 'arg.bas = %s', str(self._bas))
        logger.debug3(self, 'arg.env = %s', str(self._env))
        logger.debug3(self, 'ecpbas  = %s', str(self._ecpbas))

        # ! self._built = True is moved here !
        self._built = True
        
        # * Get the symmetry-adapted basis
        # TODO : The folloing codes are terrible! SHOULD BE RECODED!
        if not self.dirac4c:
            # * Non-relativistic calculation.
            self.symm_orb, self.irrep_id, self.id_2_irrep = \
                        orbital_symm_utils.symm_adapted_basis_double_group(self, singleatom, self.groupname)
            self.irrep_name = [self.id_2_irrep[ir] for ir in self.irrep_id]
            i = -1
            self.equal_basis = {}

            for ipair in range(group_info.EQUAL_BASIS[self.groupname].__len__()):
                i+=1
                if group_info.EQUAL_BASIS[self.groupname][ipair][0] in self.irrep_id:
                    tmp = group_info.EQUAL_BASIS[self.groupname][ipair]
                    self.equal_basis[i] = [int(numpy.where(numpy.array(self.irrep_id)==irrepid)[0]) for irrepid in tmp]
            # * Using the irrep matrix representation to generate another basis in multi-dimensional representations
            # * from one known representation.
            
            self.symm_orb = orbital_symm_utils.generate_equal_basis(self, singleatom,\
                self.groupname, self.symm_orb, self.equal_basis)
        else:
            # * Dirac 4c calculation
            self.symm_orb, self.irrep_id, self.id_2_irrep = \
                        orbital_symm_utils.symm_adapted_basis_double_group_dirac_spinor(self, singleatom, self.groupname, self.ovlp)
            self.irrep_name = [self.id_2_irrep[ir] for ir in self.irrep_id]
            i = -1
            self.equal_basis = {}

            for ipair in range(group_info.EQUAL_BASIS[self.groupname].__len__()):
                i+=1
                if group_info.EQUAL_BASIS[self.groupname][ipair][0] in self.irrep_id:
                    tmp = group_info.EQUAL_BASIS[self.groupname][ipair]
                    self.equal_basis[i] = [int(numpy.where(numpy.array(self.irrep_id)==irrepid)[0]) for irrepid in tmp]
            # * Using the irrep matrix representation to generate another basis in multi-dimensional representations
            # * from one known representation.
            
            self.symm_orb = orbital_symm_utils.generate_equal_basis(self, singleatom,\
                self.groupname, self.symm_orb, self.equal_basis)
        
        return self