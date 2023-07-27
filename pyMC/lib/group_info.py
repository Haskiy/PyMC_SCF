#!/usr/bin/env python
r'''
Author: Zhichen Pu
Date: 2021-03-26 18:32:16
LastEditTime: 2021-06-08 19:27:41
LastEditors: Pu Zhichen
Description: 
    group informations.
FilePath: \pyMC\lib\group_info.py
'''

import time
import numpy
import pyscf
import scipy
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf.dft import numint
from pyMC.tools import Dmatrix, rotate_dm
import scipy.linalg
from pyscf import __config__
from scipy.spatial.transform import Rotation as R
from pyMC.lib import group_info

epsilon = numpy.exp(2*numpy.pi*1.0j/3)
TwoC25 = numpy.cos(2.0/5.0*numpy.pi)*2.0
TwoC45 = numpy.cos(4.0/5.0*numpy.pi)*2.0
epsilon5 = numpy.exp(4*numpy.pi*1.0j/5)
delta = numpy.exp(2*numpy.pi*1.0j/5)

# ! Cr5 is defined as follow
#       y
#       2
#   3
#           1   x
#   4
#       5
# * D5_g is defined in Point-Group Theory Tables 
# * Simon L. Altmann and Peter Herzig
# Todo This is cumbersome. It should be changed:
# Todo      1. User define interface.
# Todo      2. Automately decide combining with pyscf to generate at least all the rotation angles both
# Todo         and Euler angles.

D5_theta = numpy.array([[0.0, 0.0, 0.0], #E
                        [1.2566370614359172, 0.0, 0.0], #C51
                        [5.026548245743669, 0.0, 0.0],  #C54
                        [2.5132741228718345, 0.0, 0.0], #C52
                        [3.7699111843077517, 0.0, 0.0], #C53
                        [3.14159265359, 3.14159265359, 0.           ],  #1
                        [-0.628318530718,  3.141592638689,  0.            ], #2
                        [1.884955592154, 3.14159265359 , 0.            ], #3
                        [-1.884955592154,  3.14159265359 ,  0.            ], #4
                        [0.628318530718, 3.141592638689, 0.            ]])  #5
D5_chi = {
    'A1': numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    'A2': numpy.array([1.0, 1.0, 1.0, 1.0, 1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0,-1.0,-1.0,-1.0,-1.0,-1.0]),
    'E1': numpy.array([2.0, TwoC25, TwoC25, TwoC45, TwoC45, 0.0, 0.0, 0.0, 0.0, 0.0,
                       2.0, TwoC25, TwoC25, TwoC45, TwoC45, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E2': numpy.array([2.0, TwoC45, TwoC45, TwoC25, TwoC25, 0.0, 0.0, 0.0, 0.0, 0.0,
                       2.0, TwoC45, TwoC45, TwoC25, TwoC25, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E12': numpy.array([2.0,-TwoC45,-TwoC45, TwoC25, TwoC25, 0.0, 0.0, 0.0, 0.0, 0.0,
                        -2.0, TwoC45, TwoC45,-TwoC25,-TwoC25, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E32': numpy.array([2.0,-TwoC25,-TwoC25, TwoC45, TwoC45, 0.0, 0.0, 0.0, 0.0, 0.0,
                       -2.0, TwoC25, TwoC25,-TwoC45,-TwoC45, 0.0, 0.0, 0.0, 0.0, 0.0]),
    '1E52': numpy.array([1.0,-1.0,-1.0, 1.0, 1.0, 1.0j, 1.0j, 1.0j, 1.0j, 1.0j,
                        -1.0, 1.0, 1.0,-1.0,-1.0,-1.0j,-1.0j,-1.0j,-1.0j,-1.0j]),
    '2E52': numpy.array([1.0,-1.0,-1.0, 1.0, 1.0,-1.0j,-1.0j,-1.0j,-1.0j,-1.0j,
                        -1.0, 1.0, 1.0,-1.0,-1.0, 1.0j, 1.0j, 1.0j, 1.0j, 1.0j]),
}

D5_salpha = {
    'A1': 1,
    'A2': 1,
    'E1': 2,
    'E2': 2,
    'E12': 2,
    'E32': 2,
    '1E52': 1,
    '2E52': 1,
}

D5_atom_change = numpy.array([[1, 2, 3, 4, 5], #E
                              [2, 3, 4, 5, 1], #C51
                              [5, 1, 2, 3, 4], #C54
                              [3, 4, 5, 1, 2], #C52
                              [4, 5, 1, 2, 3], #C53
                              [1, 5, 4, 3, 2], #1
                              [3, 2, 1, 5, 4], #2
                              [5, 4, 3, 2, 1], #3
                              [2, 1, 5, 4, 3], #4
                              [4, 3, 2, 1, 5] #5
                              ])
D5_theta_rotvec = {
    'nx': numpy.array([[0.0, 0.0, 1.0],     # E
                       [0.0, 0.0, 1.0],     #C51
                       [0.0, 0.0, 1.0],     #C54
                       [0.0, 0.0, 1.0],     #C52
                       [0.0, 0.0, 1.0],     #C53
                       [ 1.66554536, -0.00000000, -0.00000000],    # 1
                       [ 0.51468182,  1.58402777, -0.00000000],    # 2
                       [-1.34745450,  0.97898300, -0.00000000],    # 3
                       [-1.34745450, -0.97898300, -0.00000000],    # 4
                       [ 0.51468182, -1.58402777, -0.00000000],    # 5
                       [0.0, 0.0, 1.0],     # E
                       [0.0, 0.0, 1.0],     #C51
                       [0.0, 0.0, 1.0],     #C54
                       [0.0, 0.0, 1.0],     #C52
                       [0.0, 0.0, 1.0],     #C53
                       [ 1.66554536, -0.00000000, -0.00000000],    # 1
                       [ 0.51468182,  1.58402777, -0.00000000],    # 2
                       [-1.34745450,  0.97898300, -0.00000000],    # 3
                       [-1.34745450, -0.97898300, -0.00000000],    # 4
                       [ 0.51468182, -1.58402777, -0.00000000]]),  # 5
    'theta': numpy.array([0.0,                  # E
                          numpy.pi*2.0/5.0,     # C51
                         -numpy.pi*2.0/5.0,     #-C51
                          numpy.pi*4.0/5.0,     # C52
                         -numpy.pi*4.0/5.0,     #-C52
                          numpy.pi,             # 1
                          numpy.pi,             # 2
                          numpy.pi,             # 3
                          numpy.pi,             # 4
                          numpy.pi,             # 5
                          0.0 + numpy.pi*2,                  # Etilde
                          numpy.pi*2.0/5.0 + numpy.pi*2,     # C51tilde
                         -numpy.pi*2.0/5.0 + numpy.pi*2,     #-C51tilde
                          numpy.pi*4.0/5.0 + numpy.pi*2,     # C52tilde
                         -numpy.pi*4.0/5.0 + numpy.pi*2,     #-C52tilde
                          numpy.pi + numpy.pi*2,             # 1tilde
                          numpy.pi + numpy.pi*2,             # 2tilde
                          numpy.pi + numpy.pi*2,             # 3tilde
                          numpy.pi + numpy.pi*2,             # 4tilde
                          numpy.pi + numpy.pi*2              # 5tilde
                          ])
}

D5_matrix_rep ={
    'A1': numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    'A2': numpy.array([1.0, 1.0, 1.0, 1.0, 1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0,-1.0,-1.0,-1.0,-1.0,-1.0]),
    'E1_1': numpy.array([1.0, delta.conj(), delta, epsilon5.conj(), epsilon5, 0.0, 0.0, 0.0, 0.0, 0.0,
                         1.0, delta.conj(), delta, epsilon5.conj(), epsilon5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E1_2': numpy.array([1.0, delta, delta.conj(), epsilon5, epsilon5.conj(), 0.0, 0.0, 0.0, 0.0, 0.0,
                         1.0, delta, delta.conj(), epsilon5, epsilon5.conj(), 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E2_1': numpy.array([1.0, epsilon5.conj(), epsilon5, delta, delta.conj(), 0.0, 0.0, 0.0, 0.0, 0.0,
                         1.0, epsilon5.conj(), epsilon5, delta, delta.conj(), 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E2_2': numpy.array([1.0, epsilon5, epsilon5.conj(), delta.conj(), delta, 0.0, 0.0, 0.0, 0.0, 0.0,
                         1.0, epsilon5, epsilon5.conj(), delta.conj(), delta, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E12_1': numpy.array([1.0,-epsilon5,-epsilon5.conj(), delta.conj(), delta, 0.0, 0.0, 0.0, 0.0, 0.0,
                         -1.0, epsilon5, epsilon5.conj(),-delta.conj(),-delta, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E12_2': numpy.array([1.0,-epsilon5.conj(),-epsilon5, delta, delta.conj(), 0.0, 0.0, 0.0, 0.0, 0.0,
                         -1.0, epsilon5.conj(), epsilon5,-delta,-delta.conj(), 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E32_1': numpy.array([1.0,-delta,-delta.conj(), epsilon5, epsilon5.conj(), 0.0, 0.0, 0.0, 0.0, 0.0,
                         -1.0, delta, delta.conj(),-epsilon5,-epsilon5.conj(), 0.0, 0.0, 0.0, 0.0, 0.0]),
    'E32_2': numpy.array([1.0,-delta.conj(),-delta, epsilon5.conj(), epsilon5, 0.0, 0.0, 0.0, 0.0, 0.0,
                         -1.0, delta.conj(), delta,-epsilon5.conj(),-epsilon5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    '1E52': numpy.array([1.0,-1.0,-1.0, 1.0, 1.0, 1.0j, 1.0j, 1.0j, 1.0j, 1.0j,
                        -1.0, 1.0, 1.0,-1.0,-1.0,-1.0j,-1.0j,-1.0j,-1.0j,-1.0j]),
    '2E52': numpy.array([1.0,-1.0,-1.0, 1.0, 1.0,-1.0j,-1.0j,-1.0j,-1.0j,-1.0j,
                        -1.0, 1.0, 1.0,-1.0,-1.0, 1.0j, 1.0j, 1.0j, 1.0j, 1.0j])
}

D5_matrix_2_chi = {
    'A1':    'A1',
    'A2':    'A2', 
    'E1_1':  'E1',
    'E1_2':  'E1',
    'E2_1':  'E2',
    'E2_2':  'E2',
    'E12_1': 'E12',
    'E12_2': 'E12',
    'E32_1': 'E32',
    'E32_2': 'E32',
    '1E52':  '1E52',
    '2E52':  '2E52'
}

D5_irrep_id = {
    'A1':    0,
    'A2':    1, 
    'E1_1':  2,
    'E1_2':  3,
    'E2_1':  4,
    'E2_2':  5,
    'E12_1': 6,
    'E12_2': 7,
    'E32_1': 8,
    'E32_2': 9,
    '1E52':  10,
    '2E52':  11
    }

D5_equal_basis = {
    0 : (2,3),
    1 : (4,5),
    2 : (6,7),
    3 : (8,9)
}

D5_equal_basis_shift = {
    0 : (numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0,-1.0,-epsilon5,-delta.conj(),-delta,-epsilon5.conj(),
                       0.0, 0.0, 0.0, 0.0, 0.0,-1.0,-epsilon5,-delta.conj(),-delta,-epsilon5.conj()]), ) ,
    1 : (numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, delta.conj(), epsilon5.conj(), epsilon5, delta,
                       0.0, 0.0, 0.0, 0.0, 0.0, 1.0, delta.conj(), epsilon5.conj(), epsilon5, delta]), ) ,
    2 : (numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0,-1.0j,-delta*1.0j,-epsilon5*1.0j,-epsilon5.conj()*1.0j,-delta.conj()*1.0j,
                       0.0, 0.0, 0.0, 0.0, 0.0, 1.0j, delta*1.0j, epsilon5*1.0j, epsilon5.conj()*1.0j, delta.conj()*1.0j]), ), 
    3 : (numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0j, epsilon5.conj()*1.0j, delta*1.0j, delta.conj()*1.0j, epsilon5*1.0j,
                       0.0, 0.0, 0.0, 0.0, 0.0,-1.0j,-epsilon5.conj()*1.0j,-delta*1.0j,-delta.conj()*1.0j,-epsilon5*1.0j]), )
}

# ! Cr3 is defined as follow
#   Saved in (111) surface.
#   1-->(0 1 1)
#   2-->(1 1 0)
#   3-->(1 0 1)
# * D3_g is defined in Point-Group Theory Tables 
# * Simon L. Altmann and Peter Herzig


D3_theta = numpy.array([[0.0, 0.0, 0.            ],
                        [2.094395102393,  0.0, 0.0],
                        [-2.094395102393,  0.0,  0.0],
                        [3.141592653589793, 3.141592653589793, 0.               ],
                        [1.047197551196598, 3.141592653589793, 0.               ],
                        [-1.047197551196598,  3.141592653589793,  0.               ]])  #3Etild
D3_chi = {
    'A1': numpy.array( [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    'A2': numpy.array( [1.0, 1.0, 1.0,-1.0,-1.0,-1.0,
                        1.0, 1.0, 1.0,-1.0,-1.0,-1.0]),
    'E': numpy.array(  [2.0, -1.0, -1.0, 0.0, 0.0, 0.0,
                        2.0, -1.0, -1.0, 0.0, 0.0, 0.0]),
    'E12': numpy.array([2.0,  1.0,  1.0, 0.0, 0.0, 0.0,
                       -2.0, -1.0, -1.0, 0.0, 0.0, 0.0]),
    '1E32': numpy.array([1.0, -1.0, -1.0, 1.0j, 1.0j, 1.0j,
                        -1.0,  1.0,  1.0,-1.0j,-1.0j,-1.0j]),
    '2E32': numpy.array([1.0, -1.0, -1.0,-1.0j,-1.0j,-1.0j,
                        -1.0,  1.0,  1.0, 1.0j, 1.0j, 1.0j])
}

D3_matrix_rep = {
    'A1': numpy.array( [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    'A2': numpy.array( [1.0, 1.0, 1.0,-1.0,-1.0,-1.0,
                        1.0, 1.0, 1.0,-1.0,-1.0,-1.0]),
    'E_1':numpy.array( [1.0, epsilon.conj(), epsilon, 0.0, 0.0, 0.0,
                        1.0, epsilon.conj(), epsilon, 0.0, 0.0, 0.0]),
    'E_2':numpy.array( [1.0, epsilon, epsilon.conj(), 0.0, 0.0, 0.0,
                        1.0, epsilon, epsilon.conj(), 0.0, 0.0, 0.0]),
    'E12_1': numpy.array([ 1.0,-epsilon,-epsilon.conj(), 0.0, 0.0, 0.0,
                          -1.0, epsilon, epsilon.conj(), 0.0, 0.0, 0.0]),
    'E12_2': numpy.array([ 1.0,-epsilon.conj(),-epsilon, 0.0, 0.0, 0.0,
                          -1.0, epsilon.conj(), epsilon, 0.0, 0.0, 0.0]),
    '1E32': numpy.array([1.0, -1.0, -1.0, 1.0j, 1.0j, 1.0j,
                        -1.0,  1.0,  1.0,-1.0j,-1.0j,-1.0j]),
    '2E32': numpy.array([1.0, -1.0, -1.0,-1.0j,-1.0j,-1.0j,
                        -1.0,  1.0,  1.0, 1.0j, 1.0j, 1.0j])
}

# TODO : the keys 0,1 are meaning less, should be changed to irrep name.
D3_equal_basis = {
    0 : (2,3),
    1 : (4,5)
}
D3_equal_basis_shift = {
    0 : (numpy.array( [0.0, 0.0, 0.0,-1.0,-epsilon,-epsilon.conj(),
                      0.0, 0.0, 0.0,-1.0,-epsilon,-epsilon.conj()]),),
    1 : (numpy.array( [0.0, 0.0, 0.0,-1.0j,-1.0j*epsilon,-1.0j*epsilon.conj(),
                       0.0, 0.0, 0.0, 1.0j, 1.0j*epsilon, 1.0j*epsilon.conj()]),)
}

D3_irrep_id = {
    'A1':   0,
    'A2':   1,
    'E_1':  2,
    'E_2':  3,
    'E12_1':4,
    'E12_2':5,
    '1E32': 6,
    '2E32': 7
}

D3_matrix_2_chi = {
    'A1':   'A1',
    'A2':   'A2',
    'E_1':  'E',
    'E_2':  'E',
    'E12_1':'E12',
    'E12_2':'E12',
    '1E32': '1E32',
    '2E32': '2E32'
}

D3_salpha = {
    'A1': 1,
    'A2': 1,
    'E': 2,
    'E12': 2,
    '1E32': 1,
    '2E32': 1
}

# D3_atom_change = numpy.array([[1, 2, 3], # E      111 plane
#                               [3, 1, 2], # C1
#                               [2, 3, 1], # C2
#                               [1, 3, 2], # 1 
#                               [3, 2, 1], # 2 
#                               [2, 1, 3]  # 3 
#                               ])
D3_atom_change = numpy.array([[1, 2, 3], # E 
                              [2, 3, 1], # C1
                              [3, 1, 2], # C2
                              [1, 3, 2], # 1 
                              [3, 2, 1], # 2 
                              [2, 1, 3]  # 3 
                              ])

# D3_theta_rotvec = {
#     'nx': numpy.array([[1.0, 1.0, 1.0],     # E
#                        [1.0, 1.0, 1.0],     # C1
#                        [1.0, 1.0, 1.0],     # C2
#                        [-2.0/3.0, 1.0/3.0, 1.0/3.0],    # 1
#                        [ 1.0/3.0, 1.0/3.0,-2.0/3.0],    # 2
#                        [ 1.0/3.0,-2.0/3.0, 1.0/3.0],
#                        [1.0, 1.0, 1.0],     # Etilde
#                        [1.0, 1.0, 1.0],     # C1Etilde
#                        [1.0, 1.0, 1.0],     # C2Etilde
#                        [-2.0/3.0, 1.0/3.0, 1.0/3.0],    # 1Etilde
#                        [ 1.0/3.0, 1.0/3.0,-2.0/3.0],    # 2Etilde
#                        [ 1.0/3.0,-2.0/3.0, 1.0/3.0]]),  # 3Etilde
#     'theta': numpy.array([0.0,
#                           numpy.pi*2.0/3.0,
#                           -numpy.pi*2.0/3.0,
#                           numpy.pi,
#                           numpy.pi,
#                           numpy.pi,
#                           numpy.pi*2, # Etilde
#                           numpy.pi*2.0/3.0+numpy.pi*2,
#                           -numpy.pi*2.0/3.0+numpy.pi*2,
#                           numpy.pi+numpy.pi*2,
#                           numpy.pi+numpy.pi*2,
#                           numpy.pi+numpy.pi*2])
# }

D3_theta_rotvec = {
    'nx': numpy.array([[0.0, 0.0, 1.0],     # E
                       [0.0, 0.0, 1.0],     # C1
                       [0.0, 0.0, 1.0],     # C2
                       [1.0, 0.0, 0.0],    # 1
                       [-0.5, 0.5*numpy.sqrt(3), 0.0],    # 2
                       [-0.5,-0.5*numpy.sqrt(3), 0.0],
                       [0.0, 0.0, 1.0],     # E
                       [0.0, 0.0, 1.0],     # C1
                       [0.0, 0.0, 1.0],     # C2
                       [1.0, 0.0, 0.0],    # 1
                       [-0.5, 0.5*numpy.sqrt(3), 0.0],    # 2
                       [-0.5,-0.5*numpy.sqrt(3), 0.0]]),  # 3Etilde
    'theta': numpy.array([0.0,
                          numpy.pi*2.0/3.0,
                          -numpy.pi*2.0/3.0,
                          numpy.pi,
                          numpy.pi,
                          numpy.pi,
                          numpy.pi*2, # Etilde
                          numpy.pi*2.0/3.0+numpy.pi*2,
                          -numpy.pi*2.0/3.0+numpy.pi*2,
                          numpy.pi+numpy.pi*2,
                          numpy.pi+numpy.pi*2,
                          numpy.pi+numpy.pi*2])
}

# * https://zh.webqc.org/symmetrypointgroup-d2.html
# ! rotate in z y x order
D2_theta = numpy.array([[0.             ,  0.            ,  0.],
                        [0.             , 1.570796326795 ,  1.570796326795],
                        [ 1.570796326795, 1.570796326795 ,  -3.14159265359 ],
                        [2.677945044589 , 2.300523983022 ,  0.463647609001],
                        [-2.356194490192, 1.230959417341 ,  -0.785398163397],
                        [-1.107148717794, 2.300523983022 ,  -2.034443935796]])  #3Etild
D2_chi = {
    'A': numpy.array( [1.0, 1.0, 1.0, 1.0]),
    'B1': numpy.array( [1.0, 1.0,-1.0,-1.0]),
    'B2': numpy.array( [1.0, -1.0, 1.0, -1.0]),
    'B3': numpy.array([1.0, -1.0, -1.0, 1.0])
}

D2_salpha = {
    'A': 1,
    'B1': 1,
    'B2': 1,
    'B3': 1
}

D2_atom_change = numpy.array([[1, 2], # E 
                              [1, 2], # C1
                              [2, 1], # C2
                              [2, 1]
                              ])

D2_theta_rotvec = {
    'nx': numpy.array([[0.0, 0.0, 1.0],     # E
                       [0.0, 0.0, 1.0],     # x
                       [0.0, 1.0, 0.0],     # y
                       [1.0, 0.0, 0.0]      # z
                       ]),  # 3
    'theta': numpy.array([0.0,
                          numpy.pi,
                          numpy.pi,
                          numpy.pi])
}

# * C3 double group
C3_chi = {
    'A':     numpy.array( [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    '1E':    numpy.array( [1.0, epsilon.conj(), epsilon, 1.0, epsilon.conj(), epsilon]),
    '2E':    numpy.array( [1.0, epsilon, epsilon.conj(), 1.0, epsilon, epsilon.conj()]),
    '1E12':  numpy.array( [1.0,-epsilon.conj(),-epsilon,-1.0, epsilon.conj(), epsilon]),
    '2E12':  numpy.array( [1.0,-epsilon,-epsilon.conj(),-1.0, epsilon, epsilon.conj()]),
    'A32':   numpy.array( [1.0,-1.0,-1.0,-1.0, 1.0, 1.0])
}

C3_matrix_rep = C3_chi # One dimension

# TODO : the keys 0,1 are meaning less, should be changed to irrep name.
C3_equal_basis = {
}
C3_equal_basis_shift = {
}

C3_irrep_id = {
    'A':     0,
    '1E':    1,
    '2E':    2,
    '1E12':  3,
    '2E12':  4,
    'A32':   5
}

C3_matrix_2_chi = {
    'A':     'A'   ,
    '1E':    '1E'  ,
    '2E':    '2E'  ,
    '1E12':  '1E12',
    '2E12':  '2E12',
    'A32':   'A32'
}

C3_salpha = {
    'A':     1,
    '1E':    1,
    '2E':    1,
    '1E12':  1,
    '2E12':  1,
    'A32':   1
}

C3_theta_rotvec = {
    'nx': None,
    'theta': numpy.array([0.0,
                          numpy.pi*2.0/3.0,
                          -numpy.pi*2.0/3.0,
                          numpy.pi*2, # Etilde
                          numpy.pi*2.0/3.0+numpy.pi*2,
                          -numpy.pi*2.0/3.0+numpy.pi*2])
}

# ! Following should be updated, when adding new group.
# ! Following should be updated, when adding new group.
GROUP = ['D5', 'D3', 'C3']

NG = {'D5' : 20,
      'D3' : 12,
      'C3' : 6}

THETA = {'D5' : D5_theta,
         'D3' : D3_theta,
         'C3' : None}

CHI = {'D5' : D5_chi,
       'D3' : D3_chi,
       'C3' : C3_chi}

MATRIX_REP = {'D5' : D5_matrix_rep,
              'D3' : D3_matrix_rep,
              'C3' : C3_matrix_rep}

SALPHA = {'D5' : D5_salpha,
          'D3' : D3_salpha,
          'C3' : C3_salpha}

ATOM_CHANGE = {'D5' : D5_atom_change,
               'D3' : D3_atom_change,
               'C3' : None}

U_ROTATE = {'D5' : D5_theta_rotvec,
            'D3' : D3_theta_rotvec,
            'C3' : C3_theta_rotvec}

SGROUP2D = {'D3h' : 'D3',
            'D5h' : 'D5'}

IRREP_ID_TABLE_DOUBLE = {'D3' : D3_irrep_id,
                         'D5' : D5_irrep_id,
                         'C3' : C3_irrep_id}

MATRIX_2_CHI = {'D3' : D3_matrix_2_chi,
                'D5' : D5_matrix_2_chi,
                'C3' : C3_matrix_2_chi}

EQUAL_BASIS = {'D3' : D3_equal_basis,
               'D5' : D5_equal_basis,
               'C3' : C3_equal_basis}

EQUAL_BASIS_SHIFT = {'D3' : D3_equal_basis_shift,
                     'D5' : D5_equal_basis_shift,
                     'C3' : C3_equal_basis_shift}

GROUP_2_VORTEX_GROUP = {'D3' : 'C3'}