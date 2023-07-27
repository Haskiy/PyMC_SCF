#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import numpy
import time
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import eval_mat, _dot_ao_ao, _dot_ao_dm
from pyscf.dft.numint import _scale_ao, _contract_rho
from pyscf.dft.numint import _rks_gga_wv0, _rks_gga_wv1
from pyscf.dft.numint import _uks_gga_wv0, _uks_gga_wv1
from pyscf.dft.numint import OCCDROP
from pyscf.pbc.dft.gen_grid import make_mask, BLKSIZE
from pyscf.pbc.lib.kpts_helper import member


def eval_ao(cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0, shls_slice=None,
            non0tab=None, out=None, verbose=None):
    '''Collocate AO crystal orbitals (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Cell`

        coords : (nx*ny*nz, 3) ndarray
            The real-space grid point coordinates.

    Kwargs:
        kpt : (3,) ndarray
            The k-point corresponding to the crystal AO.
        deriv : int
            AO derivative order.  It affects the shape of the return array.
            If deriv=0, the returned AO values are stored in a (N,nao) array.
            Otherwise the AO values are stored in an array of shape (M,N,nao).
            Here N is the number of grids, nao is the number of AO functions,
            M is the size associated to the derivative deriv.

    Returns:
        aoR : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray
            The value of the AO crystal orbitals on the real-space grid by default.
            If deriv=1, also contains the value of the orbitals gradient in the
            x, y, and z directions.  It can be either complex or float array,
            depending on the kpt argument.  If kpt is not given (gamma point),
            aoR is a float array.

    See Also:
        pyscf.dft.numint.eval_ao

    '''
    ao_kpts = eval_ao_kpts(cell, coords, numpy.reshape(kpt, (-1,3)), deriv,
                           relativity, shls_slice, non0tab, out, verbose)
    return ao_kpts[0]

def uks_gga_wv0_intbypart_noweight(rho, vxc, fxc):
    """Calculate 

    Args:
        rho (2D numpy array): density
        vxc (tuple consist of 2 2D numpy arrays): [description]
        fxc (tuple consist of 3 2D numpy arrays): [description]
        weight (1D numpy array): numerical weights

    Returns:
        wva, wvb (2D numpy array): 
        wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
        wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
        ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
        bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
    """
    rhoa, rhob = rho
    u, d = vxc[0].T
    uu, ud, dd = vxc[1].T
    # u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    ngrid = uu.size
    wva = numpy.empty((4,ngrid))
    wvb = numpy.empty((4,ngrid))
    wvrho_nrho = numpy.empty((12,ngrid))
    wvnrho_nrho = numpy.empty((21,ngrid))
    wva[0]  = u  
    wva[1:] = rhoa[1:4] * (uu * 2)  # sigma_uu
    wva[1:]+= rhob[1:4] * (ud)      # sigma_ud
    wvb[0]  = d  
    wvb[1:] = rhob[1:4] * (dd * 2)  # sigma_dd
    wvb[1:]+= rhoa[1:4] * (ud)      # sigma_ud

    wvrho_nrho[0:3] = rhoa[1:4] * (u_uu * 2)
    wvrho_nrho[0:3]+= rhob[1:4] * (u_ud)
    wvrho_nrho[3:6] = rhob[1:4] * (u_dd * 2)
    wvrho_nrho[3:6]+= rhoa[1:4] * (u_ud)
    wvrho_nrho[6:9] = rhoa[1:4] * (d_uu * 2)
    wvrho_nrho[6:9]+= rhob[1:4] * (d_ud)
    wvrho_nrho[9:12] = rhob[1:4] * (d_dd * 2)
    wvrho_nrho[9:12]+= rhoa[1:4] * (d_ud)

    wvnrho_nrho[0:3] = 4*uu_uu*rhoa[1]*rhoa[1:4] + 2*uu_ud*rhoa[1]*rhob[1:4] \
                    + 2*uu_ud*rhob[1]*rhoa[1:4] + ud_ud*rhob[1]*rhob[1:4]
    wvnrho_nrho[0]  += 2*uu
    wvnrho_nrho[3:5] = 4*uu_uu*rhoa[2]*rhoa[2:4] + 2*uu_ud*rhoa[2]*rhob[2:4]\
                    + 2*uu_ud*rhob[2]*rhoa[2:4] + ud_ud*rhob[2]*rhob[2:4]
    wvnrho_nrho[3]  += 2*uu
    wvnrho_nrho[5] = 4*uu_uu*rhoa[3]*rhoa[3] + 2*uu_ud*rhoa[3]*rhob[3] \
                    + 2*uu_ud*rhob[3]*rhoa[3] + ud_ud*rhob[3]*rhob[3]
    wvnrho_nrho[5]+= 2*uu
    wvnrho_nrho[6:9] = 4*uu_dd*rhoa[1]*rhob[1:4] + 2*uu_ud*rhoa[1]*rhoa[1:4] \
                    + ud_ud*rhob[1]*rhoa[1:4] + 2*ud_dd*rhob[1]*rhob[1:4]
    wvnrho_nrho[6]  += ud
    wvnrho_nrho[9:12] = 4*uu_dd*rhoa[2]*rhob[1:4] + 2*uu_ud*rhoa[2]*rhoa[1:4] \
                    + ud_ud*rhob[2]*rhoa[1:4] + 2*ud_dd*rhob[2]*rhob[1:4]
    wvnrho_nrho[10]   += ud
    wvnrho_nrho[12:15] = 4*uu_dd*rhoa[3]*rhob[1:4] + 2*uu_ud*rhoa[3]*rhoa[1:4]\
                    + ud_ud*rhob[3]*rhoa[1:4] + 2*ud_dd*rhob[3]*rhob[1:4]
    wvnrho_nrho[14] += ud
    wvnrho_nrho[15:18] = 4*dd_dd*rhob[1]*rhob[1:4] + 2*ud_dd*rhob[1]*rhoa[1:4] \
                    + 2*ud_dd*rhoa[1]*rhob[1:4] + ud_ud*rhoa[1]*rhoa[1:4]
    wvnrho_nrho[15]+= 2*dd
    wvnrho_nrho[18:20] = 4*dd_dd*rhob[2]*rhob[2:4] + 2*ud_dd*rhob[2]*rhoa[2:4] \
                        + 2*ud_dd*rhoa[2]*rhob[2:4] + ud_ud*rhoa[2]*rhoa[2:4]
    wvnrho_nrho[18]   += 2*dd
    wvnrho_nrho[20] = 4*dd_dd*rhob[3]*rhob[3] + 2*ud_dd*rhob[3]*rhoa[3] \
                    + 2*ud_dd*rhoa[3]*rhob[3] + ud_ud*rhoa[3]*rhoa[3]
    wvnrho_nrho[20]+= 2*dd

    return wva, wvb, wvrho_nrho, wvnrho_nrho

def get_2d_offset():
    """
    Get the offset for 2d arrays
    Note that xx, xy, xz, yy, yz, zz
              0   1   2   3   4   5
    """
    offset2 = numpy.zeros((3,3),dtype = numpy.int8)
    offset2[0,0] = 0
    offset2[0,1] = 1
    offset2[0,2] = 2
    offset2[1,1] = 3
    offset2[1,2] = 4
    offset2[2,2] = 5
    # get the identical part
    offset2[1,0] = 1
    offset2[2,0] = 2
    offset2[2,1] = 4
    return offset2

def get_3d_offset():
    """
    Get the offset for 2d arrays
     xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
      0   1   2   3   4   5   6   7   8   9
    """
    offset3 = numpy.zeros((3,3,3),dtype = numpy.int8)
    offset3[0,0,0] = 0
    offset3[0,0,1] = offset3[0,1,0] = offset3[1,0,0] = 1
    offset3[0,0,2] = offset3[0,2,0] = offset3[2,0,0] = 2
    offset3[0,1,1] = offset3[1,1,0] = offset3[1,0,1] = 3
    offset3[0,1,2] = offset3[0,2,1] = offset3[1,0,2] = offset3[1,2,0] = offset3[2,0,1] = offset3[2,1,0] = 4
    offset3[0,2,2] = offset3[2,2,0] = offset3[2,0,2] = 5
    offset3[1,1,1] = 6
    offset3[1,1,2] = offset3[1,2,1] = offset3[2,1,1] = 7
    offset3[1,2,2] = offset3[2,1,2] = offset3[2,2,1] = 8
    offset3[2,2,2] = 9
    return offset3

def get_kxc_in_s_n(rho, v2rhosigma, v2sigma2, kxc):
    """This subroutine calculates the integral by part potential, using 
       AGEC (new AGC method, also known as multi-collinear appraoch) for GGA functionals

    Args:
        rho (numpy.array): ((den_u,grad_xu,grad_yu,grad_zu, xxu, xyu, xzu, yyu, yzu, zzu)
                            (den_d,grad_xd,grad_yd,grad_zd, xxd, xyd, xzd, yyd, yzd, zzd))
        v2rhosigma ([type]): v2rhosigma[:,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
        v2sigma2 ([type]): [description]
        kxc ([type]): [description]

    Returns:
        n_s_Ns : (3, ngrid) x y z
        s_s_Ns : (3, ngrid) x y z
        n_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
        s_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
        s_Nn_Ns : (3, 3, ngrid) (x y z) times (x y z)
        Nn_Ns_Ns : (3,6,ngrid) (x y z) times (xx xy xz yy yz zz)
        Ns_Ns_Ns : (10, ngrid) xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
    """
    # * get some important paremeters
    ngrid = rho[0].shape[-1]
    
    # * unpack all the paremeters
    rhoa, rhob = rho
    # 2D array of (10,N) to store density and "density derivatives" for x,y,z components
    #   and "2nd-order density derivatives" xx,xy,xz,yy,yz,zz if xctype = GGA
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = v2rhosigma.T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = v2sigma2.T
    u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd = kxc[1].T
    u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, \
        d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd =  kxc[2].T
    uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, \
        ud_ud_dd, ud_dd_dd, dd_dd_dd = kxc[3].T  
        
    # * First get the kxc in n+ and n-
    # * concerning the memory using, all the following parts are done by part
    # * First one including only one Nabla part.
    # ! init some of the temperate paremeters
    u_u_Nu = numpy.zeros((3, ngrid))
    u_d_Nu = numpy.zeros((3, ngrid))
    d_d_Nu = numpy.zeros((3, ngrid))
    u_u_Nd = numpy.zeros((3, ngrid))
    u_d_Nd = numpy.zeros((3, ngrid))
    d_d_Nd = numpy.zeros((3, ngrid))
    # ! init some of the output paremeters
    n_s_Ns = numpy.zeros((3, ngrid))
    s_s_Ns = numpy.zeros((3, ngrid))
    n_s_Nn = numpy.zeros((3, ngrid))
    s_s_Nn = numpy.zeros((3, ngrid))
    
    # ! calculate all the temp paremerters
    u_u_Nu[0] = 2*u_u_uu*rhoa[1] + u_u_ud*rhob[1]
    u_u_Nu[1] = 2*u_u_uu*rhoa[2] + u_u_ud*rhob[2] 
    u_u_Nu[2] = 2*u_u_uu*rhoa[3] + u_u_ud*rhob[3]
    u_d_Nu[0] = 2*u_d_uu*rhoa[1] + u_d_ud*rhob[1]
    u_d_Nu[1] = 2*u_d_uu*rhoa[2] + u_d_ud*rhob[2] 
    u_d_Nu[2] = 2*u_d_uu*rhoa[3] + u_d_ud*rhob[3]
    d_d_Nu[0] = 2*d_d_uu*rhoa[1] + d_d_ud*rhob[1]
    d_d_Nu[1] = 2*d_d_uu*rhoa[2] + d_d_ud*rhob[2] 
    d_d_Nu[2] = 2*d_d_uu*rhoa[3] + d_d_ud*rhob[3]
    
    u_u_Nd[0] = 2*u_u_dd*rhob[1] + u_u_ud*rhoa[1]
    u_u_Nd[1] = 2*u_u_dd*rhob[2] + u_u_ud*rhoa[2] 
    u_u_Nd[2] = 2*u_u_dd*rhob[3] + u_u_ud*rhoa[3]
    u_d_Nd[0] = 2*u_d_dd*rhob[1] + u_d_ud*rhoa[1]
    u_d_Nd[1] = 2*u_d_dd*rhob[2] + u_d_ud*rhoa[2] 
    u_d_Nd[2] = 2*u_d_dd*rhob[3] + u_d_ud*rhoa[3]
    d_d_Nd[0] = 2*d_d_dd*rhob[1] + d_d_ud*rhoa[1]
    d_d_Nd[1] = 2*d_d_dd*rhob[2] + d_d_ud*rhoa[2] 
    d_d_Nd[2] = 2*d_d_dd*rhob[3] + d_d_ud*rhoa[3]
    
    # ! get the final output paremeters
    n_s_Ns[0] = (u_u_Nu[0] - u_u_Nd[0] - d_d_Nu[0] + d_d_Nd[0])*0.125
    n_s_Ns[1] = (u_u_Nu[1] - u_u_Nd[1] - d_d_Nu[1] + d_d_Nd[1])*0.125
    n_s_Ns[2] = (u_u_Nu[2] - u_u_Nd[2] - d_d_Nu[2] + d_d_Nd[2])*0.125
    s_s_Ns[0] = (u_u_Nu[0] - u_u_Nd[0] - 2*u_d_Nu[0] + 2*u_d_Nd[0] + d_d_Nu[0] - d_d_Nd[0])*0.125
    s_s_Ns[1] = (u_u_Nu[1] - u_u_Nd[1] - 2*u_d_Nu[1] + 2*u_d_Nd[1] + d_d_Nu[1] - d_d_Nd[1])*0.125
    s_s_Ns[2] = (u_u_Nu[2] - u_u_Nd[2] - 2*u_d_Nu[2] + 2*u_d_Nd[2] + d_d_Nu[2] - d_d_Nd[2])*0.125
    # following from rho dependent potential
    n_s_Nn[0] = (u_u_Nu[0] + u_u_Nd[0] - d_d_Nu[0] - d_d_Nd[0])*0.125
    n_s_Nn[1] = (u_u_Nu[1] + u_u_Nd[1] - d_d_Nu[1] - d_d_Nd[1])*0.125
    n_s_Nn[2] = (u_u_Nu[2] + u_u_Nd[2] - d_d_Nu[2] - d_d_Nd[2])*0.125
    s_s_Nn[0] = (u_u_Nu[0] + u_u_Nd[0] - 2*u_d_Nu[0] - 2*u_d_Nd[0] + d_d_Nu[0] + d_d_Nd[0])*0.125
    s_s_Nn[1] = (u_u_Nu[1] + u_u_Nd[1] - 2*u_d_Nu[1] - 2*u_d_Nd[1] + d_d_Nu[1] + d_d_Nd[1])*0.125
    s_s_Nn[2] = (u_u_Nu[2] + u_u_Nd[2] - 2*u_d_Nu[2] - 2*u_d_Nd[2] + d_d_Nu[2] + d_d_Nd[2])*0.125
    
    # ! relase the memeroty
    u_u_Nu = u_d_Nu = d_d_Nu = u_u_Nd = u_d_Nd = d_d_Nd = None
    
    # ~ Now turn to the second part
    # * Second one including two Nabla part.
    # ! init some of the temperate paremeters
    u_Nu_Nu = numpy.zeros((6, ngrid))
    d_Nu_Nu = numpy.zeros((6, ngrid))
    u_Nu_Nd = numpy.zeros((3, 3, ngrid))
    d_Nu_Nd = numpy.zeros((3, 3, ngrid))
    u_Nd_Nd = numpy.zeros((6, ngrid))
    d_Nd_Nd = numpy.zeros((6, ngrid))
    # ! init some of the output paremeters
    n_Ns_Ns = numpy.zeros((6, ngrid))
    s_Ns_Ns = numpy.zeros((6, ngrid))
    s_Nn_Ns = numpy.zeros((3, 3, ngrid))
    n_Nn_Ns = numpy.zeros((3, 3, ngrid))
    s_Nn_Nn = numpy.zeros((6, ngrid))
    # Note that xx, xy, xz, yy, yz, zz
    #           0   1   2   3   4   5
        
    # ! calculate all the temp paremerters
    u_Nu_Nu[0] = 2*u_uu + 4*u_uu_uu*rhoa[1]*rhoa[1] + 2*u_uu_ud*rhoa[1]*rhob[1] + 2*u_uu_ud*rhob[1]*rhoa[1]\
        + u_ud_ud*rhob[1]*rhob[1]
    u_Nu_Nu[1] = 4*u_uu_uu*rhoa[1]*rhoa[2] + 2*u_uu_ud*rhoa[1]*rhob[2] + 2*u_uu_ud*rhob[1]*rhoa[2]\
        + u_ud_ud*rhob[1]*rhob[2]
    u_Nu_Nu[2] = 4*u_uu_uu*rhoa[1]*rhoa[3] + 2*u_uu_ud*rhoa[1]*rhob[3] + 2*u_uu_ud*rhob[1]*rhoa[3]\
        + u_ud_ud*rhob[1]*rhob[3]
    u_Nu_Nu[3] = 2*u_uu + 4*u_uu_uu*rhoa[2]*rhoa[2] + 2*u_uu_ud*rhoa[2]*rhob[2] + 2*u_uu_ud*rhob[2]*rhoa[2]\
        + u_ud_ud*rhob[2]*rhob[2]
    u_Nu_Nu[4] = 4*u_uu_uu*rhoa[2]*rhoa[3] + 2*u_uu_ud*rhoa[2]*rhob[3] + 2*u_uu_ud*rhob[2]*rhoa[3]\
        + u_ud_ud*rhob[2]*rhob[3]
    u_Nu_Nu[5] = 2*u_uu + 4*u_uu_uu*rhoa[3]*rhoa[3] + 2*u_uu_ud*rhoa[3]*rhob[3] + 2*u_uu_ud*rhob[3]*rhoa[3]\
        + u_ud_ud*rhob[3]*rhob[3]
        
    d_Nu_Nu[0] = 2*d_uu + 4*d_uu_uu*rhoa[1]*rhoa[1] + 2*d_uu_ud*rhoa[1]*rhob[1] + 2*d_uu_ud*rhob[1]*rhoa[1]\
        + d_ud_ud*rhob[1]*rhob[1]
    d_Nu_Nu[1] = 4*d_uu_uu*rhoa[1]*rhoa[2] + 2*d_uu_ud*rhoa[1]*rhob[2] + 2*d_uu_ud*rhob[1]*rhoa[2]\
        + d_ud_ud*rhob[1]*rhob[2]
    d_Nu_Nu[2] = 4*d_uu_uu*rhoa[1]*rhoa[3] + 2*d_uu_ud*rhoa[1]*rhob[3] + 2*d_uu_ud*rhob[1]*rhoa[3]\
        + d_ud_ud*rhob[1]*rhob[3]
    d_Nu_Nu[3] = 2*d_uu + 4*d_uu_uu*rhoa[2]*rhoa[2] + 2*d_uu_ud*rhoa[2]*rhob[2] + 2*d_uu_ud*rhob[2]*rhoa[2]\
        + d_ud_ud*rhob[2]*rhob[2]
    d_Nu_Nu[4] = 4*d_uu_uu*rhoa[2]*rhoa[3] + 2*d_uu_ud*rhoa[2]*rhob[3] + 2*d_uu_ud*rhob[2]*rhoa[3]\
        + d_ud_ud*rhob[2]*rhob[3]
    d_Nu_Nu[5] = 2*d_uu + 4*d_uu_uu*rhoa[3]*rhoa[3] + 2*d_uu_ud*rhoa[3]*rhob[3] + 2*d_uu_ud*rhob[3]*rhoa[3]\
        + d_ud_ud*rhob[3]*rhob[3]
        
    u_Nu_Nd[0,0] = 4*u_uu_dd*rhoa[1]*rhob[1] + 2*u_uu_ud*rhoa[1]*rhoa[1] + u_ud + u_ud_ud*rhob[1]*rhoa[1] \
        +2*u_ud_dd*rhob[1]*rhob[1]
    u_Nu_Nd[0,1] = 4*u_uu_dd*rhoa[1]*rhob[2] + 2*u_uu_ud*rhoa[1]*rhoa[2] + u_ud_ud*rhob[1]*rhoa[2] \
        +2*u_ud_dd*rhob[1]*rhob[2]
    u_Nu_Nd[0,2] = 4*u_uu_dd*rhoa[1]*rhob[3] + 2*u_uu_ud*rhoa[1]*rhoa[3] + u_ud_ud*rhob[1]*rhoa[3] \
        +2*u_ud_dd*rhob[1]*rhob[3]
    u_Nu_Nd[1,0] = 4*u_uu_dd*rhoa[2]*rhob[1] + 2*u_uu_ud*rhoa[2]*rhoa[1] + u_ud_ud*rhob[2]*rhoa[1] \
        +2*u_ud_dd*rhob[2]*rhob[1]
    u_Nu_Nd[1,1] = 4*u_uu_dd*rhoa[2]*rhob[2] + 2*u_uu_ud*rhoa[2]*rhoa[2] + u_ud + u_ud_ud*rhob[2]*rhoa[2] \
        +2*u_ud_dd*rhob[2]*rhob[2]
    u_Nu_Nd[1,2] = 4*u_uu_dd*rhoa[2]*rhob[3] + 2*u_uu_ud*rhoa[2]*rhoa[3] + u_ud_ud*rhob[2]*rhoa[3] \
        +2*u_ud_dd*rhob[2]*rhob[3]
    u_Nu_Nd[2,0] = 4*u_uu_dd*rhoa[3]*rhob[1] + 2*u_uu_ud*rhoa[3]*rhoa[1] + u_ud_ud*rhob[3]*rhoa[1] \
        +2*u_ud_dd*rhob[3]*rhob[1]
    u_Nu_Nd[2,1] = 4*u_uu_dd*rhoa[3]*rhob[2] + 2*u_uu_ud*rhoa[3]*rhoa[2] + u_ud_ud*rhob[3]*rhoa[2] \
        +2*u_ud_dd*rhob[3]*rhob[2]
    u_Nu_Nd[2,2] = 4*u_uu_dd*rhoa[3]*rhob[3] + 2*u_uu_ud*rhoa[3]*rhoa[3] + u_ud + u_ud_ud*rhob[3]*rhoa[3] \
        +2*u_ud_dd*rhob[3]*rhob[3]
        
    d_Nu_Nd[0,0] = 4*d_uu_dd*rhoa[1]*rhob[1] + 2*d_uu_ud*rhoa[1]*rhoa[1] + d_ud + d_ud_ud*rhob[1]*rhoa[1] \
        +2*d_ud_dd*rhob[1]*rhob[1]
    d_Nu_Nd[0,1] = 4*d_uu_dd*rhoa[1]*rhob[2] + 2*d_uu_ud*rhoa[1]*rhoa[2] + d_ud_ud*rhob[1]*rhoa[2] \
        +2*d_ud_dd*rhob[1]*rhob[2]
    d_Nu_Nd[0,2] = 4*d_uu_dd*rhoa[1]*rhob[3] + 2*d_uu_ud*rhoa[1]*rhoa[3] + d_ud_ud*rhob[1]*rhoa[3] \
        +2*d_ud_dd*rhob[1]*rhob[3]
    d_Nu_Nd[1,0] = 4*d_uu_dd*rhoa[2]*rhob[1] + 2*d_uu_ud*rhoa[2]*rhoa[1] + d_ud_ud*rhob[2]*rhoa[1] \
        +2*d_ud_dd*rhob[2]*rhob[1]
    d_Nu_Nd[1,1] = 4*d_uu_dd*rhoa[2]*rhob[2] + 2*d_uu_ud*rhoa[2]*rhoa[2] + d_ud + d_ud_ud*rhob[2]*rhoa[2] \
        +2*d_ud_dd*rhob[2]*rhob[2]
    d_Nu_Nd[1,2] = 4*d_uu_dd*rhoa[2]*rhob[3] + 2*d_uu_ud*rhoa[2]*rhoa[3] + d_ud_ud*rhob[2]*rhoa[3] \
        +2*d_ud_dd*rhob[2]*rhob[3]
    d_Nu_Nd[2,0] = 4*d_uu_dd*rhoa[3]*rhob[1] + 2*d_uu_ud*rhoa[3]*rhoa[1] + d_ud_ud*rhob[3]*rhoa[1] \
        +2*d_ud_dd*rhob[3]*rhob[1]
    d_Nu_Nd[2,1] = 4*d_uu_dd*rhoa[3]*rhob[2] + 2*d_uu_ud*rhoa[3]*rhoa[2] + d_ud_ud*rhob[3]*rhoa[2] \
        +2*d_ud_dd*rhob[3]*rhob[2]
    d_Nu_Nd[2,2] = 4*d_uu_dd*rhoa[3]*rhob[3] + 2*d_uu_ud*rhoa[3]*rhoa[3] + d_ud + d_ud_ud*rhob[3]*rhoa[3] \
        +2*d_ud_dd*rhob[3]*rhob[3]
        
    u_Nd_Nd[0] = 2*u_dd + 4*u_dd_dd*rhob[1]*rhob[1] + 2*u_ud_dd*rhob[1]*rhoa[1] + 2*u_ud_dd*rhoa[1]*rhob[1] \
        + u_ud_ud*rhoa[1]*rhoa[1]
    u_Nd_Nd[1] = 4*u_dd_dd*rhob[1]*rhob[2] + 2*u_ud_dd*rhob[1]*rhoa[2] + 2*u_ud_dd*rhoa[1]*rhob[2] \
        + u_ud_ud*rhoa[1]*rhoa[2]
    u_Nd_Nd[2] = 4*u_dd_dd*rhob[1]*rhob[3] + 2*u_ud_dd*rhob[1]*rhoa[3] + 2*u_ud_dd*rhoa[1]*rhob[3] \
        + u_ud_ud*rhoa[1]*rhoa[3]
    u_Nd_Nd[3] = 2*u_dd + 4*u_dd_dd*rhob[2]*rhob[2] + 2*u_ud_dd*rhob[2]*rhoa[2] + 2*u_ud_dd*rhoa[2]*rhob[2] \
        + u_ud_ud*rhoa[2]*rhoa[2]
    u_Nd_Nd[4] = 4*u_dd_dd*rhob[2]*rhob[3] + 2*u_ud_dd*rhob[2]*rhoa[3] + 2*u_ud_dd*rhoa[2]*rhob[3] \
        + u_ud_ud*rhoa[2]*rhoa[3]
    u_Nd_Nd[5] = 2*u_dd + 4*u_dd_dd*rhob[3]*rhob[3] + 2*u_ud_dd*rhob[3]*rhoa[3] + 2*u_ud_dd*rhoa[3]*rhob[3] \
        + u_ud_ud*rhoa[3]*rhoa[3]
        
    d_Nd_Nd[0] = 2*d_dd + 4*d_dd_dd*rhob[1]*rhob[1] + 2*d_ud_dd*rhob[1]*rhoa[1] + 2*d_ud_dd*rhoa[1]*rhob[1] \
        + d_ud_ud*rhoa[1]*rhoa[1]
    d_Nd_Nd[1] = 4*d_dd_dd*rhob[1]*rhob[2] + 2*d_ud_dd*rhob[1]*rhoa[2] + 2*d_ud_dd*rhoa[1]*rhob[2] \
        + d_ud_ud*rhoa[1]*rhoa[2]
    d_Nd_Nd[2] = 4*d_dd_dd*rhob[1]*rhob[3] + 2*d_ud_dd*rhob[1]*rhoa[3] + 2*d_ud_dd*rhoa[1]*rhob[3] \
        + d_ud_ud*rhoa[1]*rhoa[3]
    d_Nd_Nd[3] = 2*d_dd + 4*d_dd_dd*rhob[2]*rhob[2] + 2*d_ud_dd*rhob[2]*rhoa[2] + 2*d_ud_dd*rhoa[2]*rhob[2] \
        + d_ud_ud*rhoa[2]*rhoa[2]
    d_Nd_Nd[4] = 4*d_dd_dd*rhob[2]*rhob[3] + 2*d_ud_dd*rhob[2]*rhoa[3] + 2*d_ud_dd*rhoa[2]*rhob[3] \
        + d_ud_ud*rhoa[2]*rhoa[3]
    d_Nd_Nd[5] = 2*d_dd + 4*d_dd_dd*rhob[3]*rhob[3] + 2*d_ud_dd*rhob[3]*rhoa[3] + 2*d_ud_dd*rhoa[3]*rhob[3] \
        + d_ud_ud*rhoa[3]*rhoa[3]
        
        
    # ! get the final output paremeters
    n_Ns_Ns[0] = 0.125*( u_Nu_Nu[0] - u_Nu_Nd[0,0] - u_Nu_Nd[0,0] + u_Nd_Nd[0]
                        +d_Nu_Nu[0] - d_Nu_Nd[0,0] - d_Nu_Nd[0,0] + d_Nd_Nd[0])
    n_Ns_Ns[1] = 0.125*( u_Nu_Nu[1] - u_Nu_Nd[0,1] - u_Nu_Nd[1,0] + u_Nd_Nd[1]
                        +d_Nu_Nu[1] - d_Nu_Nd[0,1] - d_Nu_Nd[1,0] + d_Nd_Nd[1])
    n_Ns_Ns[2] = 0.125*( u_Nu_Nu[2] - u_Nu_Nd[0,2] - u_Nu_Nd[2,0] + u_Nd_Nd[2]
                        +d_Nu_Nu[2] - d_Nu_Nd[0,2] - d_Nu_Nd[2,0] + d_Nd_Nd[2])
    n_Ns_Ns[3] = 0.125*( u_Nu_Nu[3] - u_Nu_Nd[1,1] - u_Nu_Nd[1,1] + u_Nd_Nd[3]
                        +d_Nu_Nu[3] - d_Nu_Nd[1,1] - d_Nu_Nd[1,1] + d_Nd_Nd[3])
    n_Ns_Ns[4] = 0.125*( u_Nu_Nu[4] - u_Nu_Nd[1,2] - u_Nu_Nd[2,1] + u_Nd_Nd[4]
                        +d_Nu_Nu[4] - d_Nu_Nd[1,2] - d_Nu_Nd[2,1] + d_Nd_Nd[4])
    n_Ns_Ns[5] = 0.125*( u_Nu_Nu[5] - u_Nu_Nd[2,2] - u_Nu_Nd[2,2] + u_Nd_Nd[5]
                        +d_Nu_Nu[5] - d_Nu_Nd[2,2] - d_Nu_Nd[2,2] + d_Nd_Nd[5])
    
    s_Ns_Ns[0] = 0.125*(u_Nu_Nu[0] - u_Nu_Nd[0,0] - u_Nu_Nd[0,0] + u_Nd_Nd[0]
                      -d_Nu_Nu[0] + d_Nu_Nd[0,0] + d_Nu_Nd[0,0] - d_Nd_Nd[0])
    s_Ns_Ns[1] = 0.125*(u_Nu_Nu[1] - u_Nu_Nd[0,1] - u_Nu_Nd[1,0] + u_Nd_Nd[1]
                      -d_Nu_Nu[1] + d_Nu_Nd[0,1] + d_Nu_Nd[1,0] - d_Nd_Nd[1])
    s_Ns_Ns[2] = 0.125*(u_Nu_Nu[2] - u_Nu_Nd[0,2] - u_Nu_Nd[2,0] + u_Nd_Nd[2]
                      -d_Nu_Nu[2] + d_Nu_Nd[0,2] + d_Nu_Nd[2,0] - d_Nd_Nd[2])
    s_Ns_Ns[3] = 0.125*(u_Nu_Nu[3] - u_Nu_Nd[1,1] - u_Nu_Nd[1,1] + u_Nd_Nd[3]
                      -d_Nu_Nu[3] + d_Nu_Nd[1,1] + d_Nu_Nd[1,1] - d_Nd_Nd[3])
    s_Ns_Ns[4] = 0.125*(u_Nu_Nu[4] - u_Nu_Nd[1,2] - u_Nu_Nd[2,1] + u_Nd_Nd[4]
                      -d_Nu_Nu[4] + d_Nu_Nd[1,2] + d_Nu_Nd[2,1] - d_Nd_Nd[4])
    s_Ns_Ns[5] = 0.125*(u_Nu_Nu[5] - u_Nu_Nd[2,2] - u_Nu_Nd[2,2] + u_Nd_Nd[5]
                      -d_Nu_Nu[5] + d_Nu_Nd[2,2] + d_Nu_Nd[2,2] - d_Nd_Nd[5])
    
    s_Nn_Ns[0,0] = 0.125*(u_Nu_Nu[0] - u_Nu_Nd[0,0] + u_Nu_Nd[0,0] - u_Nd_Nd[0]
                        -d_Nu_Nu[0] + d_Nu_Nd[0,0] - d_Nu_Nd[0,0] + d_Nd_Nd[0])
    s_Nn_Ns[0,1] = 0.125*(u_Nu_Nu[1] - u_Nu_Nd[0,1] + u_Nu_Nd[1,0] - u_Nd_Nd[1]
                        -d_Nu_Nu[1] + d_Nu_Nd[0,1] - d_Nu_Nd[1,0] + d_Nd_Nd[1])
    s_Nn_Ns[0,2] = 0.125*(u_Nu_Nu[2] - u_Nu_Nd[0,2] + u_Nu_Nd[2,0] - u_Nd_Nd[2]
                        -d_Nu_Nu[2] + d_Nu_Nd[0,2] - d_Nu_Nd[2,0] + d_Nd_Nd[2])
    s_Nn_Ns[1,0] = 0.125*(u_Nu_Nu[1] - u_Nu_Nd[1,0] + u_Nu_Nd[0,1] - u_Nd_Nd[1]
                        -d_Nu_Nu[1] + d_Nu_Nd[1,0] - d_Nu_Nd[0,1] + d_Nd_Nd[1])
    s_Nn_Ns[1,1] = 0.125*(u_Nu_Nu[3] - u_Nu_Nd[1,1] + u_Nu_Nd[1,1] - u_Nd_Nd[3]
                        -d_Nu_Nu[3] + d_Nu_Nd[1,1] - d_Nu_Nd[1,1] + d_Nd_Nd[3])
    s_Nn_Ns[1,2] = 0.125*(u_Nu_Nu[4] - u_Nu_Nd[1,2] + u_Nu_Nd[2,1] - u_Nd_Nd[4]
                        -d_Nu_Nu[4] + d_Nu_Nd[1,2] - d_Nu_Nd[2,1] + d_Nd_Nd[4])
    s_Nn_Ns[2,0] = 0.125*(u_Nu_Nu[2] - u_Nu_Nd[2,0] + u_Nu_Nd[0,2] - u_Nd_Nd[2]
                        -d_Nu_Nu[2] + d_Nu_Nd[2,0] - d_Nu_Nd[0,2] + d_Nd_Nd[2])
    s_Nn_Ns[2,1] = 0.125*(u_Nu_Nu[4] - u_Nu_Nd[2,1] + u_Nu_Nd[1,2] - u_Nd_Nd[4]
                        -d_Nu_Nu[4] + d_Nu_Nd[2,1] - d_Nu_Nd[1,2] + d_Nd_Nd[4])
    s_Nn_Ns[2,2] = 0.125*(u_Nu_Nu[5] - u_Nu_Nd[2,2] + u_Nu_Nd[2,2] - u_Nd_Nd[5]
                        -d_Nu_Nu[5] + d_Nu_Nd[2,2] - d_Nu_Nd[2,2] + d_Nd_Nd[5])
    
    n_Nn_Ns[0,0] = 0.125*(u_Nu_Nu[0] - u_Nu_Nd[0,0] + u_Nu_Nd[0,0] - u_Nd_Nd[0]
                         +d_Nu_Nu[0] - d_Nu_Nd[0,0] + d_Nu_Nd[0,0] - d_Nd_Nd[0])
    n_Nn_Ns[0,1] = 0.125*(u_Nu_Nu[1] - u_Nu_Nd[0,1] + u_Nu_Nd[1,0] - u_Nd_Nd[1]
                         +d_Nu_Nu[1] - d_Nu_Nd[0,1] + d_Nu_Nd[1,0] - d_Nd_Nd[1])
    n_Nn_Ns[0,2] = 0.125*(u_Nu_Nu[2] - u_Nu_Nd[0,2] + u_Nu_Nd[2,0] - u_Nd_Nd[2]
                         +d_Nu_Nu[2] - d_Nu_Nd[0,2] + d_Nu_Nd[2,0] - d_Nd_Nd[2])
    n_Nn_Ns[1,0] = 0.125*(u_Nu_Nu[1] - u_Nu_Nd[1,0] + u_Nu_Nd[0,1] - u_Nd_Nd[1]
                         +d_Nu_Nu[1] - d_Nu_Nd[1,0] + d_Nu_Nd[0,1] - d_Nd_Nd[1])
    n_Nn_Ns[1,1] = 0.125*(u_Nu_Nu[3] - u_Nu_Nd[1,1] + u_Nu_Nd[1,1] - u_Nd_Nd[3]
                         +d_Nu_Nu[3] - d_Nu_Nd[1,1] + d_Nu_Nd[1,1] - d_Nd_Nd[3])
    n_Nn_Ns[1,2] = 0.125*(u_Nu_Nu[4] - u_Nu_Nd[1,2] + u_Nu_Nd[2,1] - u_Nd_Nd[4]
                         +d_Nu_Nu[4] - d_Nu_Nd[1,2] + d_Nu_Nd[2,1] - d_Nd_Nd[4])
    n_Nn_Ns[2,0] = 0.125*(u_Nu_Nu[2] - u_Nu_Nd[2,0] + u_Nu_Nd[0,2] - u_Nd_Nd[2]
                         +d_Nu_Nu[2] - d_Nu_Nd[2,0] + d_Nu_Nd[0,2] - d_Nd_Nd[2])
    n_Nn_Ns[2,1] = 0.125*(u_Nu_Nu[4] - u_Nu_Nd[2,1] + u_Nu_Nd[1,2] - u_Nd_Nd[4]
                         +d_Nu_Nu[4] - d_Nu_Nd[2,1] + d_Nu_Nd[1,2] - d_Nd_Nd[4])
    n_Nn_Ns[2,2] = 0.125*(u_Nu_Nu[5] - u_Nu_Nd[2,2] + u_Nu_Nd[2,2] - u_Nd_Nd[5]
                         +d_Nu_Nu[5] - d_Nu_Nd[2,2] + d_Nu_Nd[2,2] - d_Nd_Nd[5])
    
    s_Nn_Nn[0] = 0.125*( u_Nu_Nu[0] + u_Nu_Nd[0,0] + u_Nu_Nd[0,0] + u_Nd_Nd[0]
                        -d_Nu_Nu[0] - d_Nu_Nd[0,0] - d_Nu_Nd[0,0] - d_Nd_Nd[0])
    s_Nn_Nn[1] = 0.125*( u_Nu_Nu[1] + u_Nu_Nd[0,1] + u_Nu_Nd[1,0] + u_Nd_Nd[1]
                        -d_Nu_Nu[1] - d_Nu_Nd[0,1] - d_Nu_Nd[1,0] - d_Nd_Nd[1])
    s_Nn_Nn[2] = 0.125*( u_Nu_Nu[2] + u_Nu_Nd[0,2] + u_Nu_Nd[2,0] + u_Nd_Nd[2]
                        -d_Nu_Nu[2] - d_Nu_Nd[0,2] - d_Nu_Nd[2,0] - d_Nd_Nd[2])
    s_Nn_Nn[3] = 0.125*( u_Nu_Nu[3] + u_Nu_Nd[1,1] + u_Nu_Nd[1,1] + u_Nd_Nd[3]
                        -d_Nu_Nu[3] - d_Nu_Nd[1,1] - d_Nu_Nd[1,1] - d_Nd_Nd[3])
    s_Nn_Nn[4] = 0.125*( u_Nu_Nu[4] + u_Nu_Nd[1,2] + u_Nu_Nd[2,1] + u_Nd_Nd[4]
                        -d_Nu_Nu[4] - d_Nu_Nd[1,2] - d_Nu_Nd[2,1] - d_Nd_Nd[4])
    s_Nn_Nn[5] = 0.125*( u_Nu_Nu[5] + u_Nu_Nd[2,2] + u_Nu_Nd[2,2] + u_Nd_Nd[5]
                        -d_Nu_Nu[5] - d_Nu_Nd[2,2] - d_Nu_Nd[2,2] - d_Nd_Nd[5])
    
    # ! relase the memeroty
    u_Nu_Nu = d_Nu_Nu = u_Nu_Nd = d_Nu_Nd = u_Nd_Nd = d_Nd_Nd = None
    
    # ~ Now turn to the third part
    # * Third one including three Nabla part.
    # ! init some of the temperate paremeters
    # Note that for special case, many of them are equivelent, thus many are ignored
    # xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
    #  0   1   2   3   4   5   6   7   8   9
    offset2 = get_2d_offset()
    offset3 = get_3d_offset()
    Nu_Nu_Nu = numpy.zeros((10, ngrid))
    Nu_Nu_Nd = numpy.zeros((6,3,ngrid))
    Nd_Nd_Nu = numpy.zeros((6,3,ngrid))
    Nd_Nd_Nd = numpy.zeros((10, ngrid))
    # ! init some of the output paremeters
    Nn_Ns_Ns = numpy.zeros((3,6,ngrid))
    Ns_Ns_Ns = numpy.zeros((10, ngrid))
    Nn_Nn_Ns = numpy.zeros((6,3,ngrid))
    
    # ! calculate all the temp paremerters
    # This part is too difficult to write as before.
    for u in range(3):
        for v in range(u,3):
            for w in range(v,3):
                ui = u + 1
                vi = v + 1
                wi = w + 1
                Nu_Nu_Nu[offset3[u,v,w]] = 8*uu_uu_uu*rhoa[ui]*rhoa[vi]*rhoa[wi] + 4*uu_uu_ud*rhoa[ui]*rhoa[vi]*rhob[wi] \
                                         + 4*uu_uu_ud*rhoa[ui]*rhob[vi]*rhoa[wi] + 2*uu_ud_ud*rhoa[ui]*rhob[vi]*rhob[wi] \
                                         + 4*uu_uu_ud*rhob[ui]*rhoa[vi]*rhoa[wi] + 2*uu_ud_ud*rhob[ui]*rhoa[vi]*rhob[wi] \
                                         + 2*uu_ud_ud*rhob[ui]*rhob[vi]*rhoa[wi] +   ud_ud_ud*rhob[ui]*rhob[vi]*rhob[wi]
                Nd_Nd_Nd[offset3[u,v,w]] = 8*dd_dd_dd*rhob[ui]*rhob[vi]*rhob[wi] + 4*ud_dd_dd*rhob[ui]*rhob[vi]*rhoa[wi] \
                                         + 4*ud_dd_dd*rhob[ui]*rhoa[vi]*rhob[wi] + 2*ud_ud_dd*rhob[ui]*rhoa[vi]*rhoa[wi] \
                                         + 4*ud_dd_dd*rhoa[ui]*rhob[vi]*rhob[wi] + 2*ud_ud_dd*rhoa[ui]*rhob[vi]*rhoa[wi] \
                                         + 2*ud_ud_dd*rhoa[ui]*rhoa[vi]*rhob[wi] +   ud_ud_ud*rhoa[ui]*rhoa[vi]*rhoa[wi]
                if u == v:
                    Nu_Nu_Nu[offset3[u,v,w]] += 4*uu_uu*rhoa[wi] + 2*uu_ud*rhob[wi]
                    Nd_Nd_Nd[offset3[u,v,w]] += 4*dd_dd*rhob[wi] + 2*ud_dd*rhoa[wi]
                if u == w:
                    Nu_Nu_Nu[offset3[u,v,w]] += 4*uu_uu*rhoa[vi] + 2*uu_ud*rhob[vi]
                    Nd_Nd_Nd[offset3[u,v,w]] += 4*dd_dd*rhob[vi] + 2*ud_dd*rhoa[vi]
                if v == w:
                    Nu_Nu_Nu[offset3[u,v,w]] += 4*uu_uu*rhoa[ui] + 2*uu_ud*rhob[ui]
                    Nd_Nd_Nd[offset3[u,v,w]] += 4*dd_dd*rhob[ui] + 2*ud_dd*rhoa[ui]
    for u in range(3):
        for v in range(u,3):
            for w in range(3):
                ui = u + 1
                vi = v + 1
                wi = w + 1
                Nu_Nu_Nd[offset2[u,v],w] = 8*uu_uu_dd*rhoa[ui]*rhoa[vi]*rhob[wi] + 4*uu_uu_ud*rhoa[ui]*rhoa[vi]*rhoa[wi] \
                                         + 4*uu_ud_dd*rhoa[ui]*rhob[vi]*rhob[wi] + 2*uu_ud_ud*rhoa[ui]*rhob[vi]*rhoa[wi] \
                                         + 4*uu_ud_dd*rhob[ui]*rhoa[vi]*rhob[wi] + 2*uu_ud_ud*rhob[ui]*rhoa[vi]*rhoa[wi] \
                                         + 2*ud_ud_dd*rhob[ui]*rhob[vi]*rhob[wi] +   ud_ud_ud*rhob[ui]*rhob[vi]*rhoa[wi]
                Nd_Nd_Nu[offset2[u,v],w] = 8*uu_dd_dd*rhob[ui]*rhob[vi]*rhoa[wi] + 4*ud_dd_dd*rhob[ui]*rhob[vi]*rhob[wi] \
                                         + 4*uu_ud_dd*rhob[ui]*rhoa[vi]*rhoa[wi] + 2*ud_ud_dd*rhob[ui]*rhoa[vi]*rhob[wi] \
                                         + 4*uu_ud_dd*rhoa[ui]*rhob[vi]*rhoa[wi] + 2*ud_ud_dd*rhoa[ui]*rhob[vi]*rhob[wi] \
                                         + 2*uu_ud_ud*rhoa[ui]*rhoa[vi]*rhoa[wi] +   ud_ud_ud*rhoa[ui]*rhoa[vi]*rhob[wi]
                if u == v:
                    Nu_Nu_Nd[offset2[u,v],w] += 4*uu_dd*rhob[wi] + 2*uu_ud*rhoa[wi]
                    Nd_Nd_Nu[offset2[u,v],w] += 4*uu_dd*rhoa[wi] + 2*ud_dd*rhob[wi]
                if u == w:
                    Nu_Nu_Nd[offset2[u,v],w] += 2*uu_ud*rhoa[vi] + ud_ud*rhob[vi]
                    Nd_Nd_Nu[offset2[u,v],w] += 2*ud_dd*rhob[vi] + ud_ud*rhoa[vi]
                if v == w:
                    Nu_Nu_Nd[offset2[u,v],w] += 2*uu_ud*rhoa[ui] + ud_ud*rhob[ui]
                    Nd_Nd_Nu[offset2[u,v],w] += 2*ud_dd*rhob[ui] + ud_ud*rhoa[ui]
    for u in range(3):
        for v in range(3):
            for w in range(v,3):
                Nn_Ns_Ns[u, offset2[v,w]] = 0.125*( Nu_Nu_Nu[offset3[u,v,w]] - Nu_Nu_Nd[offset2[u,v],w]
                                                   -Nu_Nu_Nd[offset2[u,w],v] + Nd_Nd_Nu[offset2[v,w],u]
                                                   +Nu_Nu_Nd[offset2[v,w],u] - Nd_Nd_Nu[offset2[u,w],v]
                                                   -Nd_Nd_Nu[offset2[u,v],w] + Nd_Nd_Nd[offset3[u,v,w]])
                Ns_Ns_Ns[offset3[u,v,w]] = 0.125*( Nu_Nu_Nu[offset3[u,v,w]] - Nu_Nu_Nd[offset2[u,v],w]
                                                  -Nu_Nu_Nd[offset2[u,w],v] + Nd_Nd_Nu[offset2[v,w],u]
                                                  -Nu_Nu_Nd[offset2[v,w],u] + Nd_Nd_Nu[offset2[u,w],v]
                                                  +Nd_Nd_Nu[offset2[u,v],w] - Nd_Nd_Nd[offset3[u,v,w]])              
    
    for u in range(3):
        for v in range(u,3):
            for w in range(3):
                Nn_Nn_Ns[offset2[u,v],w] = 0.125*( Nu_Nu_Nu[offset3[u,v,w]] - Nu_Nu_Nd[offset2[u,v],w]
                                                  +Nu_Nu_Nd[offset2[u,w],v] - Nd_Nd_Nu[offset2[v,w],u]
                                                  +Nu_Nu_Nd[offset2[v,w],u] - Nd_Nd_Nu[offset2[u,w],v]
                                                  +Nd_Nd_Nu[offset2[u,v],w] - Nd_Nd_Nd[offset3[u,v,w]])
    
       
    # ! relase the memeroty
    Nu_Nu_Nu = Nu_Nu_Nd = Nd_Nd_Nu = Nd_Nd_Nd = None
    
    return n_s_Ns, s_s_Ns, n_s_Nn, s_s_Nn, n_Ns_Ns, s_Ns_Ns, s_Nn_Ns, \
        n_Nn_Ns, s_Nn_Nn, Nn_Ns_Ns, Ns_Ns_Ns, Nn_Nn_Ns

def eval_mat_hss(mol, ao, weight, vxc,
             non0tab=None, xctype='LDA', spin=0, verbose=None):
    r'''Calculate XC potential matrix.

    Args:
        mol : an instance of :class:`Mole`

        ao : ([4/10,] ngrids, nao) ndarray
            2D array of shape (N,nao) for LDA,
            3D array of shape (4,N,nao) for GGA
            or (10,N,nao) for meta-GGA.
            N is the number of grids, nao is the number of AO functions.
            If xctype is GGA, ao[0] is AO value and ao[1:3] are the real space
            gradients.  If xctype is meta-GGA, ao[4:10] are second derivatives
            of ao values.
        weight : 1D array
            Integral weights on grids.
        rho : ([4/6,] ngrids) ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))
        vxc : ([4,] ngrids) ndarray
            XC potential value on each grid = (vrho, vsigma, vlapl, vtau)
            vsigma is GGA potential value on each grid.
            If the kwarg spin != 0, a list [vsigma_uu,vsigma_ud] is required.

    Kwargs:
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of `ao` and `rho`
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        spin : int
            If not 0, the returned matrix is the Vxc matrix of alpha-spin.  It
            is computed with the spin non-degenerated UKS formula.

    Returns:
        XC potential matrix in 2D array of shape (nao,nao) where nao is the
        number of AO functions.
    '''
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    transpose_for_uks = False
    if xctype == 'LDA':
        vrho = vxc
        # *.5 because return mat + mat.T
        #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
        aow = _scale_ao(ao, .5*weight*vrho)
        mat = _dot_ao_ao(mol, ao, aow, non0tab, shls_slice, ao_loc)
    else:
        vxc = weight * vxc
        #:aow = numpy.einsum('npi,np->pi', ao[:4], wv)
        aow = _scale_ao(ao[:4], vxc)
        mat = _dot_ao_ao(mol, ao[0], aow, non0tab, shls_slice, ao_loc)

# JCP 138, 244108 (2013); DOI:10.1063/1.4811270
# JCP 112, 7002 (2000); DOI:10.1063/1.481298
    # if xctype == 'MGGA':
    #     vlapl, vtau = vxc[2:]

    #     if vlapl is None:
    #         vlapl = 0
    #     else:
    #         if spin != 0:
    #             if transpose_for_uks:
    #                 vlapl = vlapl.T
    #             vlapl = vlapl[0]
    #         XX, YY, ZZ = 4, 7, 9
    #         ao2 = ao[XX] + ao[YY] + ao[ZZ]
    #         #:aow = numpy.einsum('pi,p->pi', ao2, .5 * weight * vlapl, out=aow)
    #         aow = _scale_ao(ao2, .5 * weight * vlapl, out=aow)
    #         mat += _dot_ao_ao(mol, ao[0], aow, non0tab, shls_slice, ao_loc)

    #     if spin != 0:
    #         if transpose_for_uks:
    #             vtau = vtau.T
    #         vtau = vtau[0]
    #     wv = weight * (.25*vtau + vlapl)
    #     #:aow = numpy.einsum('pi,p->pi', ao[1], wv, out=aow)
    #     aow = _scale_ao(ao[1], wv, out=aow)
    #     mat += _dot_ao_ao(mol, ao[1], aow, non0tab, shls_slice, ao_loc)
    #     #:aow = numpy.einsum('pi,p->pi', ao[2], wv, out=aow)
    #     aow = _scale_ao(ao[2], wv, out=aow)
    #     mat += _dot_ao_ao(mol, ao[2], aow, non0tab, shls_slice, ao_loc)
    #     #:aow = numpy.einsum('pi,p->pi', ao[3], wv, out=aow)
    #     aow = _scale_ao(ao[3], wv, out=aow)
    #     mat += _dot_ao_ao(mol, ao[3], aow, non0tab, shls_slice, ao_loc)

    return mat + mat.T.conj()


def eval_ao_kpts(cell, coords, kpts=None, deriv=0, relativity=0,
                 shls_slice=None, non0tab=None, out=None, verbose=None, **kwargs):
    '''
    Returns:
        ao_kpts: (nkpts, [comp], ngrids, nao) ndarray
            AO values at each k-point
    '''
    if kpts is None:
        if 'kpt' in kwargs:
            sys.stderr.write('WARN: KNumInt.eval_ao function finds keyword '
                             'argument "kpt" and converts it to "kpts"\n')
            kpts = kwargs['kpt']
        else:
            kpts = numpy.zeros((1,3))
    kpts = numpy.reshape(kpts, (-1,3))

    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    if cell.cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv
    return cell.pbc_eval_gto(feval, coords, comp, kpts,
                             shls_slice=shls_slice, non0tab=non0tab, out=out)


def eval_rho(cell, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Mole` or :class:`Cell`

        ao : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray
            The value of the AO crystal orbitals on the real-space grid by default.
            If xctype='GGA', also contains the value of the gradient in the x, y,
            and z directions.

    Returns:
        rho : ([4,] nx*ny*nz) ndarray
            The value of the density on the real-space grid. If xctype='GGA',
            also contains the value of the gradient in the x, y, and z
            directions.

    See Also:
        pyscf.dft.numint.eval_rho

    '''

    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE, cell.nbas),
                              dtype=numpy.uint8)
        non0tab[:] = 0xff

    # complex orbitals or density matrix
    if numpy.iscomplexobj(ao) or numpy.iscomplexobj(dm):
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        dm = dm.astype(numpy.complex128)
# For GGA, function eval_rho returns   real(|\nabla i> D_ij <j| + |i> D_ij <\nabla j|)
#       = real(|\nabla i> D_ij <j| + |i> D_ij <\nabla j|)
#       = real(|\nabla i> D_ij <j| + conj(|\nabla j> conj(D_ij) < i|))
#       = real(|\nabla i> D_ij <j|) + real(|\nabla j> conj(D_ij) < i|)
#       = real(|\nabla i> [D_ij + (D^\dagger)_ij] <j|)
# symmetrization dm (D + D.conj().T) then /2 because the code below computes
#       2*real(|\nabla i> D_ij <j|)
        if not hermi:
            dm = (dm + dm.conj().T) * .5

        def dot_bra(bra, aodm):
            # rho = numpy.einsum('pi,pi->p', bra.conj(), aodm).real
            #:rho  = numpy.einsum('pi,pi->p', bra.real, aodm.real)
            #:rho += numpy.einsum('pi,pi->p', bra.imag, aodm.imag)
            #:return rho
            return _contract_rho(bra, aodm)

        if xctype == 'LDA' or xctype == 'HF':
            c0 = _dot_ao_dm(cell, ao, dm, non0tab, shls_slice, ao_loc)
            rho = dot_bra(ao, c0)

        elif xctype == 'GGA':
            rho = numpy.empty((4,ngrids))
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            rho[0] = dot_bra(ao[0], c0)
            for i in range(1, 4):
                rho[i] = dot_bra(ao[i], c0) * 2

        else:
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = numpy.empty((6,ngrids))
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            rho[0] = dot_bra(ao[0], c0)
            rho[5] = 0
            for i in range(1, 4):
                rho[i] = dot_bra(ao[i], c0) * 2  # *2 for +c.c.
                c1 = _dot_ao_dm(cell, ao[i], dm, non0tab, shls_slice, ao_loc)
                rho[5] += dot_bra(ao[i], c1)
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            rho[4] = dot_bra(ao2, c0)
            rho[4] += rho[5]
            rho[4] *= 2 # *2 for +c.c.
            rho[5] *= .5
    else:
        # real orbitals and real DM
        rho = numint.eval_rho(cell, ao, dm, non0tab, xctype, hermi, verbose)
    return rho


def eval_rho_ibp(cell, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Mole` or :class:`Cell`

        ao : ([10,] nx*ny*nz, nao=cell.nao_nr()) ndarray
            The value of the AO crystal orbitals on the real-space grid by default.
            If xctype='GGA', also contains the value of the gradient in the x, y,
            and z directions, and the following dimensions are xx, xy, xz, yy, yz, zz.

    Returns:
        rho : ([4,] nx*ny*nz) ndarray
            The value of the density on the real-space grid. If xctype='GGA',
            also contains the value of the gradient in the x, y, and z
            directions.

    See Also:
        pyscf.dft.numint.eval_rho

    '''
    ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE, cell.nbas),
                              dtype=numpy.uint8)
        non0tab[:] = 0xff

    # complex orbitals or density matrix
    if numpy.iscomplexobj(ao) or numpy.iscomplexobj(dm):
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        dm = dm.astype(numpy.complex128)
# For GGA, function eval_rho returns   real(|\nabla i> D_ij <j| + |i> D_ij <\nabla j|)
#       = real(|\nabla i> D_ij <j| + |i> D_ij <\nabla j|)
#       = real(|\nabla i> D_ij <j| + conj(|\nabla j> conj(D_ij) < i|))
#       = real(|\nabla i> D_ij <j|) + real(|\nabla j> conj(D_ij) < i|)
#       = real(|\nabla i> [D_ij + (D^\dagger)_ij] <j|)
# symmetrization dm (D + D.conj().T) then /2 because the code below computes
#       2*real(|\nabla i> D_ij <j|)
        if not hermi:
            dm = (dm + dm.conj().T) * .5

        def dot_bra(bra, aodm):
            # rho = numpy.einsum('pi,pi->p', bra.conj(), aodm).real
            #:rho  = numpy.einsum('pi,pi->p', bra.real, aodm.real)
            #:rho += numpy.einsum('pi,pi->p', bra.imag, aodm.imag)
            #:return rho
            return _contract_rho(bra, aodm)

        if xctype == 'LDA' or xctype == 'HF':
            raise ValueError("Should not reach here")

        elif xctype == 'GGA':
            offset2 = get_2d_offset()
            
            rho = numpy.zeros((10, ngrids))
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            rho[0] = dot_bra(ao[0], c0)
            for i in range(1, 4):
                rho[i] = dot_bra(ao[i], c0) * 2  # *2 for +c.c.
                c1 = _dot_ao_dm(cell, ao[i], dm, non0tab, shls_slice, ao_loc)
                for j in range(i,4):
                    rho[4+offset2[i-1,j-1]]+= dot_bra(ao[j], c1)*2
                
            for i in range(4,10):
                rho[i] += dot_bra(ao[i], c0)*2
        else:
            raise ValueError("Should not reach here")
    else:
        raise NotImplementedError("will not reach here.")
        # real orbitals and real DM
        # rho = eval_rho_intbypart(cell, ao, dm, non0tab, xctype, hermi, verbose)
    # import pdb
    # pdb.set_trace()
    return rho

def eval_rho2(cell, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              verbose=None):
    '''Refer to `pyscf.dft.numint.eval_rho2` for full documentation.
    '''
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                              dtype=numpy.uint8)
        non0tab[:] = 0xff

    # complex orbitals or density matrix
    if numpy.iscomplexobj(ao) or numpy.iscomplexobj(mo_coeff):
        def dot(bra, ket):
            #:rho  = numpy.einsum('pi,pi->p', bra.real, ket.real)
            #:rho += numpy.einsum('pi,pi->p', bra.imag, ket.imag)
            #:return rho
            return _contract_rho(bra, ket)

        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        pos = mo_occ > OCCDROP
        cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))

        if pos.sum() > 0:
            if xctype == 'LDA' or xctype == 'HF':
                c0 = _dot_ao_dm(cell, ao, cpos, non0tab, shls_slice, ao_loc)
                rho = dot(c0, c0)
            elif xctype == 'GGA':
                rho = numpy.empty((4,ngrids))
                c0 = _dot_ao_dm(cell, ao[0], cpos, non0tab, shls_slice, ao_loc)
                rho[0] = dot(c0, c0)
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cpos, non0tab, shls_slice, ao_loc)
                    rho[i] = dot(c0, c1) * 2  # *2 for +c.c.
            else: # meta-GGA
                # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
                rho = numpy.empty((6,ngrids))
                c0 = _dot_ao_dm(cell, ao[0], cpos, non0tab, shls_slice, ao_loc)
                rho[0] = dot(c0, c0)
                rho[5] = 0
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cpos, non0tab, shls_slice, ao_loc)
                    rho[i] = dot(c0, c1) * 2  # *2 for +c.c.
                    rho[5]+= dot(c1, c1)
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao[XX] + ao[YY] + ao[ZZ]
                c1 = _dot_ao_dm(cell, ao2, cpos, non0tab, shls_slice, ao_loc)
                rho[4] = dot(c0, c1)
                rho[4]+= rho[5]
                rho[4]*= 2
                rho[5]*= .5
        else:
            if xctype == 'LDA' or xctype == 'HF':
                rho = numpy.zeros(ngrids)
            elif xctype == 'GGA':
                rho = numpy.zeros((4,ngrids))
            else:
                rho = numpy.zeros((6,ngrids))

        neg = mo_occ < -OCCDROP
        if neg.sum() > 0:
            cneg = numpy.einsum('ij,j->ij', mo_coeff[:,neg], numpy.sqrt(-mo_occ[neg]))
            if xctype == 'LDA' or xctype == 'HF':
                c0 = _dot_ao_dm(cell, ao, cneg, non0tab, shls_slice, ao_loc)
                rho -= dot(c0, c0)
            elif xctype == 'GGA':
                c0 = _dot_ao_dm(cell, ao[0], cneg, non0tab, shls_slice, ao_loc)
                rho[0] -= dot(c0, c0)
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cneg, non0tab, shls_slice, ao_loc)
                    rho[i] -= dot(c0, c1) * 2  # *2 for +c.c.
            else:
                c0 = _dot_ao_dm(cell, ao[0], cneg, non0tab, shls_slice, ao_loc)
                rho[0] -= dot(c0, c0)
                rho5 = 0
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cneg, non0tab, shls_slice, ao_loc)
                    rho[i] -= dot(c0, c1) * 2  # *2 for +c.c.
                    rho5 -= dot(c1, c1)
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao[XX] + ao[YY] + ao[ZZ]
                c1 = _dot_ao_dm(cell, ao2, cneg, non0tab, shls_slice, ao_loc)
                rho[4] -= dot(c0, c1) * 2
                rho[4] -= rho5 * 2
                rho[5] -= rho5 * .5
    else:
        rho = numint.eval_rho2(cell, ao, mo_coeff, mo_occ, non0tab, xctype, verbose)
    return rho


def nr_rks(ni, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    '''Calculate RKS XC functional and potential matrix for given meshgrids and density matrix

    Note: This is a replica of pyscf.dft.numint.nr_rks_vxc with kpts added.
    This implemented uses slow function in numint, which only calls eval_rho, eval_mat.
    Faster function uses eval_rho2 which is not yet implemented.

    Args:
        ni : an instance of :class:`NumInt` or :class:`KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D/3D array or a list of 2D/3D arrays
            Density matrices (2D) / density matrices for k-points (3D)

    Kwargs:
        spin : int
            spin polarized if spin = 1
        relativity : int
            No effects.
        hermi : int
            No effects
        max_memory : int or float
            The maximum size of cache to use (in MB).
        verbose : int or object of :class:`Logger`
            No effects.
        kpts : (3,) ndarray or (nkpts,3) ndarray
            Single or multiple k-points sampled for the DM.  Default is gamma point.
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evaluate the XC matrix.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))

    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi)

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = [0]*nset
    if xctype == 'LDA':
        ao_deriv = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, spin=0,
                                      relativity=relativity, deriv=1)[:2]
                den = rho*weight
                nelec[i] += den.sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, spin=0,
                                      relativity=relativity, deriv=1)[:2]
                den = rho[0]*weight
                nelec[i] += den.sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, spin=0,
                                      relativity=relativity, deriv=1)[:2]
                den = rho[0]*weight
                nelec[i] += den.sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, numpy.asarray(vmat)

def nr_uks(ni, cell, grids, xc_code, dms, spin=1, relativity=0, hermi=0,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    '''Calculate UKS XC functional and potential matrix for given meshgrids and density matrix

    Note: This is a replica of pyscf.dft.numint.nr_rks_vxc with kpts added.
    This implemented uses slow function in numint, which only calls eval_rho, eval_mat.
    Faster function uses eval_rho2 which is not yet implemented.

    Args:
        ni : an instance of :class:`NumInt` or :class:`KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms :
            Density matrices

    Kwargs:
        spin : int
            spin polarized if spin = 1
        relativity : int
            No effects.
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).
        verbose : int or object of :class:`Logger`
            No effects.
        kpts : (3,) ndarray or (nkpts,3) ndarray
            Single or multiple k-points sampled for the DM.  Default is gamma point.
            kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evaluate the XC matrix.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))

    xctype = ni._xc_type(xc_code)
    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dma, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(cell, dmb, hermi)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmata = [0]*nset
    vmatb = [0]*nset
    if xctype == 'LDA':
        ao_deriv = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype)
                rho_b = make_rhob(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                vrho = vxc[0]
                den = rho_a * weight
                nelec[0,i] += den.sum()
                excsum[i] += (den*exc).sum()
                den = rho_b * weight
                nelec[1,i] += den.sum()
                excsum[i] += (den*exc).sum()

                vmata[i] += ni.eval_mat(cell, ao_k1, weight, rho_a, vrho[:,0],
                                        mask, xctype, 1, verbose)
                vmatb[i] += ni.eval_mat(cell, ao_k1, weight, rho_b, vrho[:,1],
                                        mask, xctype, 1, verbose)
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts,
                                 kpts_band, max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype)
                rho_b = make_rhob(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                vrho, vsigma = vxc[:2]
                den = rho_a[0]*weight
                nelec[0,i] += den.sum()
                excsum[i] += (den*exc).sum()
                den = rho_b[0]*weight
                nelec[1,i] += den.sum()
                excsum[i] += (den*exc).sum()

                vmata[i] += ni.eval_mat(cell, ao_k1, weight, (rho_a,rho_b),
                                        (vrho[:,0], (vsigma[:,0],vsigma[:,1])),
                                        mask, xctype, 1, verbose)
                vmatb[i] += ni.eval_mat(cell, ao_k1, weight, (rho_b,rho_a),
                                        (vrho[:,1], (vsigma[:,2],vsigma[:,1])),
                                        mask, xctype, 1, verbose)
    elif xctype == 'MGGA':
        assert(all(x not in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')))
        ao_deriv = 2
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype)
                rho_b = make_rhob(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                vrho, vsigma, vlapl, vtau = vxc
                den = rho_a[0]*weight
                nelec[0,i] += den.sum()
                excsum[i] += (den*exc).sum()
                den = rho_b[0]*weight
                nelec[1,i] += den.sum()
                excsum[i] += (den*exc).sum()

                v = (vrho[:,0], (vsigma[:,0],vsigma[:,1]), None, vtau[:,0])
                vmata[i] += ni.eval_mat(cell, ao_k1, weight, (rho_a,rho_b), v,
                                        mask, xctype, 1, verbose)
                v = (vrho[:,1], (vsigma[:,2],vsigma[:,1]), None, vtau[:,1])
                vmatb[i] += ni.eval_mat(cell, ao_k1, weight, (rho_b,rho_a), v,
                                        mask, xctype, 1, verbose)
                v = None

    if dma.ndim == vmata[0].ndim:  # One set of DMs in the input
        nelec = nelec[:,0]
        excsum = excsum[0]
        vmata = vmata[0]
        vmatb = vmatb[0]
    return nelec, excsum, numpy.asarray((vmata,vmatb))

def _format_uks_dm(dms):
    dma, dmb = dms
    if getattr(dms, 'mo_coeff', None) is not None:
        #TODO: test whether dm.mo_coeff matching dm
        mo_coeff = dms.mo_coeff
        mo_occ = dms.mo_occ
        if (isinstance(mo_coeff[0], numpy.ndarray) and
            mo_coeff[0].ndim < dma.ndim): # handle ROKS
            mo_occa = [numpy.array(occ> 0, dtype=numpy.double) for occ in mo_occ]
            mo_occb = [numpy.array(occ==2, dtype=numpy.double) for occ in mo_occ]
            dma = lib.tag_array(dma, mo_coeff=mo_coeff, mo_occ=mo_occa)
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff, mo_occ=mo_occb)
        else:
            dma = lib.tag_array(dma, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1])
    return dma, dmb


def AGEC_LDA_parallel_kernel(Mx, My, Mz, NX, rhop, xc_code, ni, index
                             , relativity, verbose, weight, ngrid, factor):
    init, finish = index
    exctot = 0.0
    vaa = numpy.zeros((ngrid))
    vbb = numpy.zeros((ngrid))
    vab_r = numpy.zeros((ngrid))
    vab_i = numpy.zeros((ngrid))
    Bxc = numpy.zeros((3,ngrid))
    for idrct in range(init,finish):
        s = 0.5*(Mx*NX[idrct,0]
           + My*NX[idrct,1]
           + Mz*NX[idrct,2])
        rho_ahss = rhop + s
        rho_bhss = rhop - s
        exc, exc_cor, vxcn, vxcs = ni.eval_xc_new_ASDP(xc_code, (rho_ahss, rho_bhss), spin=1,
                            relativity=relativity, deriv=1,
                            verbose=verbose)
        den = rho_ahss * weight
        # import pdb
        # pdb.set_trace()
        # ! Note that for energy, factor must product here!
        exctot += numpy.dot(den, exc)*factor[idrct]
        den = rho_bhss * weight
        exctot += numpy.dot(den, exc)*factor[idrct]
        exctot += numpy.dot(weight, exc_cor)*factor[idrct]
        
        vaa+= (vxcn + vxcs*NX[idrct,2])*factor[idrct]
        vbb+= (vxcn - vxcs*NX[idrct,2])*factor[idrct]
        vab_r+= vxcs*NX[idrct,0]*factor[idrct]
        vab_i+= vxcs*NX[idrct,1]*factor[idrct]
        
        Bxc[0] += vxcs*NX[idrct,0]*factor[idrct]
        Bxc[1] += vxcs*NX[idrct,1]*factor[idrct]
        Bxc[2] += vxcs*NX[idrct,2]*factor[idrct]

    return exctot, vaa, vbb, vab_r, vab_i, Bxc


def AGEC_GGA_parallel_kernel(Mx, My, Mz, NX, rhop, xc_code, ni, index
                             , relativity, verbose, weight, ngrid, factor):
    init, finish = index
    exctot = 0.0
    s = numpy.zeros((4, ngrid))
    wvaahss = numpy.zeros((4,ngrid))
    wvbbhss = numpy.zeros((4,ngrid))
    wvabhss_i = numpy.zeros((4,ngrid))
    wvabhss_r = numpy.zeros((4,ngrid))
    for idrct in range(init, finish):
        s[:,0:ngrid] = 0.5*(Mx[:,0:ngrid]*NX[idrct,0]
            + My[:,0:ngrid]*NX[idrct,1]
            + Mz[:,0:ngrid]*NX[idrct,2])
        rho_ahss = rhop + s
        rho_bhss = rhop - s
        exc, exc_cor, vxcn, vxcs = ni.eval_xc_new_ASDP(xc_code, (rho_ahss, rho_bhss), spin=1,
                            relativity=relativity, deriv=1,
                            verbose=verbose)
        # if exc.max() > 1E10:
        #     print(exc.max() + '!!!')
        den = rho_ahss[0]*weight
        # ! Note that for energy, factor must product here!
        exctot += numpy.dot(den, exc)*factor[idrct]
        den = rho_bhss[0]*weight
        exctot += numpy.dot(den, exc)*factor[idrct]
        exctot += numpy.dot(weight, exc_cor)*factor[idrct]
        
        # wva, wvb = _uks_gga_wv0((rho_ahss,rho_bhss), vxc, weight)
        # wvm = (wva - wvb)*0.5
        # wvp = (wva + wvb)*0.5
        vxcs[0] = vxcs[0]*0.5
        vxcn[0] = vxcn[0]*0.5
        # vxcn[1:4] = 0.0*vxcn[1:4]
        # vxcs[1:4] = 0.0*vxcs[1:4]
        wvaahss+= (vxcn[:,:] + vxcs[0:4,:]*NX[idrct,2])*factor[idrct]
        wvbbhss+= (vxcn[:,:] - vxcs[0:4,:]*NX[idrct,2])*factor[idrct]
        wvabhss_i+= vxcs*NX[idrct,1]*factor[idrct]
        wvabhss_r+= vxcs*NX[idrct,0]*factor[idrct]
        
    return exctot, wvaahss, wvbbhss, wvabhss_r, wvabhss_i

def nr_new_ASDP_parallel(ni, cell, grids, xc_code, dms, spin=1, relativity=0, hermi=0,
           kpts=None, kpts_band=None, max_memory=2000, NX=numpy.array([[0.0, 0.0, 1.0]]), THRESHOLD=1.0E-10, 
           THRESHOLD_lc = 0.99, verbose=None):
    '''Calculate UKS XC functional with Multi-directions framework
    and potential matrix on given meshgrids for a set of density matrices.
    This subroutine is general subroutine gksm_g which handles both collinear, 
    Multidirections and Tridirections, which is controled by attributes of numint 'NX':
        NX.ndim == 1 : collinear;
        Nx.ndim == 2 : Multidirections;
        NX.ndim == 3 : Tri-directions;              

    Args:
        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : a list of 2D arrays, which should contains 4 parts
            A list of density matrices, stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).
        NX : Input projection vectors.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of (alpha,beta) electrons generated by numerical integration.
        excsum is the XC functional value.
        vmat is the XC potential matrix for (alpha_alpha,alpha_beta,beta_alpha,beta_beta) spin.

    Examples:

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> grids = dft.gen_grid.Grids(mol)
    >>> grids.coords = numpy.random.random((100,3))  # 100 random points
    >>> grids.weights = numpy.random.random(100)
    >>> nao = mol.nao_nr()
    >>> dm = numpy.random.random((2,nao,nao))
    >>> ni = dft.numint.NumInt()
    >>> nelec, exc, vxc = ni.nr_uks(mol, grids, 'lda,vwn', dm)
    '''
    NX = numpy.load('D1454.npy')
    factor = numpy.load('W1454.npy')
    
    start = time.process_time()
    xctype = ni._xc_type(xc_code)
    ndirect = NX.shape[0]
    
    
    if xctype == 'NLC':
        NotImplementedError
    nso = dms.shape[-1]
    nao = nso // 2
    dmaa = dms[...,:nao,:nao]   
    dmab = dms[...,:nao,nao:]   
    dmba = dms[...,nao:,:nao]   
    dmbb = dms[...,nao:,nao:]   

    #print(dmaa, dmab, dmba, dmbb)
    nao = dmaa.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dmaa, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(cell, dmbb, hermi)[0]
    # make_rho        = ni._gen_rho_evaluator(mol, dmaa+dmbb, hermi)[0]
    # make_rhoMz      = ni._gen_rho_evaluator(mol, dmaa-dmbb, hermi)[0]
    make_rhoMx      = ni._gen_rho_evaluator(cell, (dmba+dmab), hermi)[0]
    make_rhoMy      = ni._gen_rho_evaluator(cell, (-dmba*1.0j+dmab*1.0j), hermi)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    # vmat will save in the order(alpha_alpha,alpha_beta,beta_alpha,beta_beta).
    vmat = numpy.zeros((4,nset,nao,nao), dtype=numpy.complex128)
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        ipart = 0
        numpy.save('coords',grids.coords)
        numpy.save('weights',grids.weights)
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory):
            # ao = ao + 0.0j
            ipart += 1
            for idm in range(nset):
                # calculate densities and M vector
                # Cause we need \nabla M informations, so we use 
                rho_aa = make_rhoa(idm, ao_k2, mask, xctype)
                rho_bb = make_rhob(idm, ao_k2, mask, xctype)
                Mx = make_rhoMx(idm, ao_k2, mask, xctype)
                My = make_rhoMy(idm, ao_k2, mask, xctype)
                Mz = numpy.real(rho_aa - rho_bb)

                ngrid = rho_aa.shape[0]
                ni.M = numpy.zeros((3,ngrid))
                ni.M[0] = Mx
                ni.M[1] = My
                ni.M[2] = Mz

                numpy.save('Mx_part'+str(ipart),Mx)
                numpy.save('My_part'+str(ipart),My)
                numpy.save('Mz_part'+str(ipart),Mz)
                # ! Debug use

                rhop = 0.5*(rho_aa + rho_bb)
                vaa_factor = numpy.zeros((ngrid))
                vbb_factor = numpy.zeros((ngrid))
                vab_r_factor = numpy.zeros((ngrid))
                vab_i_factor = numpy.zeros((ngrid))
                Bxc_tot = numpy.zeros((3,ngrid))

                # ~ import Package
                import multiprocessing
                import math
                # ~ init some parameters in parallel.
                ncpu = multiprocessing.cpu_count()
                nsbatch = math.ceil(ndirect/ncpu)
                NX_list = [(i, i+nsbatch) for i in range(0, ndirect-nsbatch, nsbatch)]
                if NX_list[-1][-1] < ndirect:
                    NX_list.append((NX_list[-1][-1], ndirect))
                pool = multiprocessing.Pool()
                para_results = []
                
                # ~ parallel run spherical average
                for para in NX_list:
                    para_results.append(pool.apply_async(AGEC_LDA_parallel_kernel,
                                                         (Mx, My, Mz, NX, rhop, xc_code, ni, para
                                                        , relativity, verbose, weight, ngrid, factor)))
                # ~ finisht the parallel part.
                pool.close()
                pool.join()
                # ~ get the final result
                for result_para in para_results:
                    result = result_para.get()
                    excsum[idm] += result[0]
                    Bxc_tot += result[5]
                    vaa_factor+= result[1]
                    vbb_factor+= result[2]
                    vab_r_factor+= result[3]
                    vab_i_factor+= result[4]
                # ~ DEBYG
                # exctot, vaa, vbb, vab_r, vab_i, Bxc = AGEC_LDA_parallel_kernel(Mx, My, Mz, NX, rhop, xc_code, ni, para
                #         , relativity, verbose, weight, ngrid, factor)
                # Bxc_tot += result[5]
                #     vaa_factor+= result[1]
                #     vbb_factor+= result[2]
                #     vab_r_factor+= result[3]
                #     vab_i_factor+= result[4]
                # ~ DEBYG
                Bxc_tot = Bxc_tot
                # import pdb
                # pdb.set_trace()
                numpy.save('Bxc_tot_part'+str(ipart),Bxc_tot)
                # ! Note that for Vmat, factor can be product here!
                # !     different from energy.
                
                # contraction
                vmat[0, idm] += ni.eval_mat_hss(cell, ao_k1, weight, vaa_factor,
                                        mask, xctype, 1, verbose)
                vabr_M = ni.eval_mat_hss(cell, ao_k1, weight, vab_r_factor,
                                        mask, xctype, 1, verbose)
                vabi_M = ni.eval_mat_hss(cell, ao_k1, weight, vab_i_factor,
                                        mask, xctype, 1, verbose)*1.0j
                vmat[1, idm] += vabr_M
                vmat[2, idm] += vabr_M
                vmat[1, idm] -= vabi_M
                vmat[2, idm] += vabi_M
                vmat[3, idm] += ni.eval_mat_hss(cell, ao_k1, weight, vbb_factor,
                                        mask, xctype, 1, verbose)
                
                rho_ahss = rho_bhss = exc = vxcn = vxcs = None
                # Nelectron
                den = rho_aa*weight
                nelec[0,idm] += den.sum()
                den = rho_bb*weight
                nelec[1,idm] += den.sum()
    elif xctype == 'GGA':
        ao_deriv = 1
        ipart = 0
        numpy.save('coords',grids.coords)
        numpy.save('weights',grids.weights)
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts,
                                 kpts_band, max_memory):
            ipart += 1
            for idm in range(nset):
                # calculate densities and M vector
                rho_aa = make_rhoa(idm, ao_k2, mask, 'GGA')
                rho_bb = make_rhob(idm, ao_k2, mask, 'GGA')
                Mx = make_rhoMx(idm, ao_k2, mask, 'GGA')
                My = make_rhoMy(idm, ao_k2, mask, 'GGA')
                # Mx (4,ngrid) Mx, nablax Mx, nablay Mx, nablaz Mx
                Mz = numpy.real(rho_aa - rho_bb)
                ngrid = rho_aa.shape[1]
                ni.M = numpy.zeros((3,ngrid))
                ni.M[0] = Mx[0]
                ni.M[1] = My[0]
                ni.M[2] = Mz[0]
                
                numpy.save('Mx_part'+str(ipart),Mx[:4,:])
                numpy.save('My_part'+str(ipart),My[:4,:])
                numpy.save('Mz_part'+str(ipart),Mz[:4,:])
                rhop = 0.5*(rho_aa + rho_bb)
                
                wvaahss_factor = numpy.zeros((4,ngrid))
                wvbbhss_factor = numpy.zeros((4,ngrid))
                wvabhss_i_factor = numpy.zeros((4,ngrid))
                wvabhss_r_factor = numpy.zeros((4,ngrid))
                # ~ import Package
                import multiprocessing
                # ~ init some parameters in parallel.
                ncpu = multiprocessing.cpu_count()
                import math
                nsbatch = math.ceil(ndirect/ncpu)
                NX_list = [(i, i+nsbatch) for i in range(0, ndirect-nsbatch, nsbatch)]
                if NX_list[-1][-1] < ndirect:
                    NX_list.append((NX_list[-1][-1], ndirect))
                pool = multiprocessing.Pool()
                para_results = []
                
                # ~ parallel run spherical average
                for para in NX_list:
                    para_results.append(pool.apply_async(AGEC_GGA_parallel_kernel,
                                                         (Mx, My, Mz, NX, rhop, xc_code, ni, para
                                                        , relativity, verbose, weight, ngrid, factor)))
                # ~ finisht the parallel part.
                pool.close()
                pool.join()
                # ~ get the final result
                for result_para in para_results:
                    result = result_para.get()
                    excsum[idm] += result[0]
                    wvaahss_factor+= result[1]
                    wvbbhss_factor+= result[2]
                    wvabhss_r_factor+= result[3]
                    wvabhss_i_factor+= result[4]
                # ! Note that for Vmat, factor can be product here!
                # !     different from energy.
                
                # Nelectron
                den = rho_aa[0]*weight
                nelec[0,idm] += den.sum()
                den = rho_bb[0]*weight
                nelec[1,idm] += den.sum()
                # contraction
                vmat[0,idm] += ni.eval_mat_hss(cell, ao_k1, weight, wvaahss_factor,
                                        mask, xctype, 1, verbose)
                
                vabr_M = ni.eval_mat_hss(cell, ao_k1, weight, wvabhss_r_factor,
                                        mask, xctype, 1, verbose)
                vabi_M = ni.eval_mat_hss(cell, ao_k1, weight, wvabhss_i_factor,
                                        mask, xctype, 1, verbose)*1.0j
                
                vmat[1, idm] += vabr_M
                vmat[2, idm] += vabr_M
                vmat[1, idm] -= vabi_M
                vmat[2, idm] += vabi_M
                vmat[3, idm] += ni.eval_mat_hss(cell, ao_k1, weight, wvbbhss_factor,
                                        mask, xctype, 1, verbose)
                

                rho_ahss = rho_bhss = exc = vxc = wva = wvb = None
                rho_ahss = rho_bhss = exc = vxc = wva = wvb = aow = None
                rho_aa = rho_bb = Mx = My = Mz = s = rhop = None
    elif xctype == 'MGGA':
        raise NotImplementedError("There is no meta-GGA fuctionals for AGEC")

    vmat = vmat[:,0]
    nelec = nelec.reshape(2)
    excsum = excsum[0]

    end = time.process_time()
    print('Running time for uksm: %s Seconds'%(end-start))
    return nelec, excsum, vmat

def AGEC_GGA_intbypart_parallel_kernel(Mx, My, Mz, NX, rhop, xc_code, ni, index
                             , relativity, verbose, weight, ngrid, factor):
    init, finish = index
    exctot = 0.0
    s = numpy.zeros((10, ngrid))
    vaa = numpy.zeros((ngrid))
    vbb = numpy.zeros((ngrid))
    vab_r = numpy.zeros((ngrid))
    vab_i = numpy.zeros((ngrid))
    Bxc = numpy.zeros((3,ngrid))
    
    for idrct in range(init, finish):
        s[:,0:ngrid] = 0.5*(Mx[:,0:ngrid]*NX[idrct,0]
            + My[:,0:ngrid]*NX[idrct,1]
            + Mz[:,0:ngrid]*NX[idrct,2])
        rho_ahss = rhop + s
        rho_bhss = rhop - s
        exc, exc_cor, vxcn, vxcs = ni.eval_xc_new_ASDP(xc_code, (rho_ahss, rho_bhss), spin=1,
                        relativity=relativity, deriv=2,
                        verbose=verbose, ibp = True)
        den = rho_ahss[0]*weight
        # ! Note that for energy, factor must product here!
        exctot += numpy.dot(den, exc)*factor[idrct]
        den = rho_bhss[0]*weight
        exctot += numpy.dot(den, exc)*factor[idrct]
        exctot += numpy.dot(weight, exc_cor)*factor[idrct]

        # numpy.savetxt('Bxc_d'+str(idrct)+'_part'+str(ipart)+'.txt',Bxc)
        Bxc[0] += vxcs*NX[idrct,0]*factor[idrct]
        Bxc[1] += vxcs*NX[idrct,1]*factor[idrct]
        Bxc[2] += vxcs*NX[idrct,2]*factor[idrct]
        
        vaa += (vxcn + vxcs*NX[idrct,2])*factor[idrct]
        vbb += (vxcn - vxcs*NX[idrct,2])*factor[idrct]
        vab_r += vxcs*NX[idrct,0]*factor[idrct]
        vab_i += vxcs*NX[idrct,1]*factor[idrct]
        
    return exctot, vaa, vbb, vab_r, vab_i, Bxc

def nr_new_ASDP_parallel_ibp(ni, cell, grids, xc_code, dms, spin=1, relativity=0, hermi=0,
           kpts=None, kpts_band=None, max_memory=2000, NX=numpy.array([[0.0, 0.0, 1.0]]), verbose=None):
    '''Calculate UKS XC functional with Multi-directions framework
    and potential matrix on given meshgrids for a set of density matrices.
    This subroutine is general subroutine gksm_g which handles both collinear, 
    Multidirections and Tridirections, which is controled by attributes of numint 'NX':
        NX.ndim == 1 : collinear;
        Nx.ndim == 2 : Multidirections;
        NX.ndim == 3 : Tri-directions;              

    Args:
        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : a list of 2D arrays, which should contains 4 parts
            A list of density matrices, stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).
        NX : Input projection vectors.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of (alpha,beta) electrons generated by numerical integration.
        excsum is the XC functional value.
        vmat is the XC potential matrix for (alpha_alpha,alpha_beta,beta_alpha,beta_beta) spin.

    Examples:

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> grids = dft.gen_grid.Grids(mol)
    >>> grids.coords = numpy.random.random((100,3))  # 100 random points
    >>> grids.weights = numpy.random.random(100)
    >>> nao = mol.nao_nr()
    >>> dm = numpy.random.random((2,nao,nao))
    >>> ni = dft.numint.NumInt()
    >>> nelec, exc, vxc = ni.nr_uks(mol, grids, 'lda,vwn', dm)
    '''
    NX = numpy.load('D1454.npy')
    factor = numpy.load('W1454.npy')
    
    start = time.process_time()
    xctype = ni._xc_type(xc_code)
    ndirect = NX.shape[0]
    
    
    if xctype == 'NLC':
        NotImplementedError
    nso = dms.shape[-1]
    nao = nso // 2
    dmaa = dms[...,:nao,:nao]   
    dmab = dms[...,:nao,nao:]   
    dmba = dms[...,nao:,:nao]   
    dmbb = dms[...,nao:,nao:]   

    #print(dmaa, dmab, dmba, dmbb)
    nao = dmaa.shape[-1]
    # import pdb
    # pdb.set_trace()
    make_rhoa, nset = ni._gen_rho_evaluator_ibp_pbc(cell, dmaa, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator_ibp_pbc(cell, dmbb, hermi)[0]
    make_rhoMx      = ni._gen_rho_evaluator_ibp_pbc(cell, (dmba+dmab), hermi)[0]
    make_rhoMy      = ni._gen_rho_evaluator_ibp_pbc(cell, (-dmba*1.0j+dmab*1.0j), hermi)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    # vmat will save in the order(alpha_alpha,alpha_beta,beta_alpha,beta_beta).
    vmat = numpy.zeros((4,nset,nao,nao), dtype=numpy.complex128)
    if xctype == 'LDA':
        raise NotImplementedError("IBP should be only applied to GGA functional")
    elif xctype == 'GGA':
        ao_deriv = 2
        ipart = 0
        numpy.save('coords',grids.coords)
        numpy.save('weights',grids.weights)
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts,
                                 kpts_band, max_memory):
            ipart += 1
            for idm in range(nset):
                # calculate densities and M vector

                rho_aa = make_rhoa(idm, ao_k2, mask, 'GGA')
                rho_bb = make_rhob(idm, ao_k2, mask, 'GGA')
                Mx = make_rhoMx(idm, ao_k2, mask, 'GGA')
                My = make_rhoMy(idm, ao_k2, mask, 'GGA')
                # Mx (4,ngrid) Mx, nablax Mx, nablay Mx, nablaz Mx
                Mz = numpy.real(rho_aa - rho_bb)
                ngrid = rho_aa.shape[1]
                ni.M = numpy.zeros((3,ngrid))
                ni.M[0] = Mx[0]
                ni.M[1] = My[0]
                ni.M[2] = Mz[0]
                
                numpy.save('Mx_part'+str(ipart),Mx[:4,:])
                numpy.save('My_part'+str(ipart),My[:4,:])
                numpy.save('Mz_part'+str(ipart),Mz[:4,:])
                rhop = 0.5*(rho_aa + rho_bb)
                
                vaa_factor = numpy.zeros((ngrid))
                vbb_factor = numpy.zeros((ngrid))
                vabi_factor = numpy.zeros((ngrid))
                vabr_factor = numpy.zeros((ngrid))
                Bxc_tot = numpy.zeros((3,ngrid))
                # ~ import Package
                import multiprocessing
                # ~ init some parameters in parallel.
                ncpu = multiprocessing.cpu_count()
                import math
                nsbatch = math.ceil(ndirect/ncpu)
                NX_list = [(i, i+nsbatch) for i in range(0, ndirect-nsbatch, nsbatch)]
                if NX_list[-1][-1] < ndirect:
                    NX_list.append((NX_list[-1][-1], ndirect))
                pool = multiprocessing.Pool()
                para_results = []
                # NX_list = [(0,50)]
                # import pdb
                # pdb.set_trace()                
                # para = (0,50)
                # exctot, vaa, vbb, vab_r, vab_i, Bxc = AGEC_GGA_intbypart_parallel_kernel(Mx, My, Mz, NX, rhop, xc_code, ni, para
                #         , relativity, verbose, weight, ngrid, factor)
                # excsum[idm] += exctot
                # Bxc_tot += Bxc
                # vaa_factor+= vaa
                # vbb_factor+= vbb
                # vabr_factor+= vab_r
                # vabi_factor+= vab_i
                
                # ~ parallel run spherical average
                for para in NX_list:
                    para_results.append(pool.apply_async(AGEC_GGA_intbypart_parallel_kernel,
                                                         (Mx, My, Mz, NX, rhop, xc_code, ni, para
                                                        , relativity, verbose, weight, ngrid, factor)))
                # ~ finisht the parallel part.
                pool.close()
                pool.join()
                # ~ get the final result
                for result_para in para_results:
                    result = result_para.get()
                    excsum[idm] += result[0]
                    Bxc_tot += result[5]
                    vaa_factor+= result[1]
                    vbb_factor+= result[2]
                    vabr_factor+= result[3]
                    vabi_factor+= result[4]
                # ! Note that for Vmat, factor can be product here!
                # !     different from energy.
                numpy.save('Bxc_tot_part'+str(ipart),Bxc_tot)
                # Nelectron
                den = rho_aa[0]*weight
                nelec[0,idm] += den.sum()
                den = rho_bb[0]*weight
                nelec[1,idm] += den.sum()
                # contraction
                vmat[0,idm] += ni.eval_mat_hss(cell, ao_k1[0], weight, vaa_factor,
                                        mask, 'LDA', 1, verbose)
                
                vabr_M = ni.eval_mat_hss(cell, ao_k1[0], weight, vabr_factor,
                                        mask, 'LDA', 1, verbose)
                vabi_M = ni.eval_mat_hss(cell, ao_k1[0], weight, vabi_factor,
                                        mask, 'LDA', 1, verbose)*1.0j
                # import pdb
                # pdb.set_trace()
                vmat[1, idm] += vabr_M
                vmat[2, idm] += vabr_M
                vmat[1, idm] -= vabi_M
                vmat[2, idm] += vabi_M
                vmat[3, idm] += ni.eval_mat_hss(cell, ao_k1[0], weight, vbb_factor,
                                        mask, 'LDA', 1, verbose)
                

                rho_ahss = rho_bhss = exc = vxc = wva = wvb = None
                rho_ahss = rho_bhss = exc = vxc = wva = wvb = aow = None
                rho_aa = rho_bb = Mx = My = Mz = s = rhop = None
    elif xctype == 'MGGA':
        raise NotImplementedError("There is no meta-GGA fuctionals for AGEC")

    vmat = vmat[:,0]
    nelec = nelec.reshape(2)
    excsum = excsum[0]

    end = time.process_time()
    print('Running time for uksm: %s Seconds'%(end-start))
    return nelec, excsum, vmat

nr_rks_vxc = nr_rks
nr_uks_vxc = nr_uks

def nr_rks_fxc(ni, cell, grids, xc_code, dm0, dms, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, kpts=None, max_memory=2000,
               verbose=None):
    '''Contract RKS XC kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt` or :class:`KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D/3D array or a list of 2D/3D arrays
            Density matrices (2D) / density matrices for k-points (3D)

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).
        rho0 : float array
            Zero-order density (and density derivative for GGA).  Giving kwargs rho0,
            vxc and fxc to improve better performance.
        vxc : float array
            First order XC derivatives
        fxc : float array
            Second order XC derivatives

    Examples:

    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)

    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi)
    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        make_rho0 = ni._gen_rho_evaluator(cell, dm0, 1)[0]

    ao_loc = cell.ao_loc_nr()
    vmat = [0] * nset
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if fxc is None:
                rho = make_rho0(0, ao_k1, mask, xctype)
                fxc0 = ni.eval_xc(xc_code, rho, spin=0,
                                  relativity=relativity, deriv=2,
                                  verbose=verbose)[2]
                frr = fxc0[0]
            else:
                frr = fxc[0][ip:ip+ngrid]
                ip += ngrid

            for i in range(nset):
                rho1 = make_rho(i, ao_k1, mask, xctype)
                wv = weight * frr * rho1
                vmat[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if rho0 is None:
                rho = make_rho0(0, ao_k1, mask, xctype)
            else:
                rho = numpy.asarray(rho0[:,ip:ip+ngrid], order='C')

            if vxc is None or fxc is None:
                vxc0, fxc0 = ni.eval_xc(xc_code, rho, spin=0,
                                        relativity=relativity, deriv=2,
                                        verbose=verbose)[1:3]
            else:
                vxc0 = (None, vxc[1][ip:ip+ngrid])
                fxc0 = (fxc[0][ip:ip+ngrid], fxc[1][ip:ip+ngrid], fxc[2][ip:ip+ngrid])
                ip += ngrid

            for i in range(nset):
                rho1 = make_rho(i, ao_k1, mask, xctype)
                wv = _rks_gga_wv1(rho, rho1, vxc0, fxc0, weight)
                vmat[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

        # call swapaxes method to swap last two indices because vmat may be a 3D
        # array (nset,nao,nao) in single k-point mode or a 4D array
        # (nset,nkpts,nao,nao) in k-points mode
        for i in range(nset):  # for (\nabla\mu) \nu + \mu (\nabla\nu)
            vmat[i] = vmat[i] + vmat[i].swapaxes(-2,-1).conj()

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if isinstance(dms, numpy.ndarray) and dms.ndim == vmat[0].ndim:
        # One set of DMs in the input
        vmat = vmat[0]
    return numpy.asarray(vmat)

def nr_rks_fxc_st(ni, cell, grids, xc_code, dm0, dms_alpha, relativity=0, singlet=True,
                  rho0=None, vxc=None, fxc=None, kpts=None, max_memory=2000,
                  verbose=None):
    '''Associated to singlet or triplet Hessian
    Note the difference to nr_rks_fxc, dms_alpha is the response density
    matrices of alpha spin, alpha+/-beta DM is applied due to singlet/triplet
    coupling

    Ref. CPL, 256, 454
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)

    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms_alpha)
    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        make_rho0 = ni._gen_rho_evaluator(cell, dm0, 1)[0]

    ao_loc = cell.ao_loc_nr()
    vmat = [0] * nset
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if fxc is None:
                rho = make_rho0(0, ao_k1, mask, xctype)
                rho *= .5  # alpha density
                fxc0 = ni.eval_xc(xc_code, (rho,rho), spin=1, deriv=2)[2]
                u_u, u_d, d_d = fxc0[0].T
            else:
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T
                ip += ngrid
            if singlet:
                frho = u_u + u_d
            else:
                frho = u_u - u_d

            for i in range(nset):
                rho1 = make_rho(i, ao_k1, mask, xctype)
                wv = weight * frho * rho1
                vmat[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if vxc is None or fxc is None:
                rho = make_rho0(0, ao_k1, mask, xctype)
                rho *= .5  # alpha density
                vxc0, fxc0 = ni.eval_xc(xc_code, (rho,rho), spin=1, deriv=2)[1:3]

                vsigma = vxc0[1].T
                u_u, u_d, d_d = fxc0[0].T  # v2rho2
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc0[1].T  # v2rhosigma
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc0[2].T  # v2sigma2
            else:
                rho = rho0[0][:,ip:ip+ngrid]
                vsigma = vxc[1][ip:ip+ngrid].T
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T  # v2rho2
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1][ip:ip+ngrid].T  # v2rhosigma
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2][ip:ip+ngrid].T  # v2sigma2

            if singlet:
                fgamma = vsigma[0] + vsigma[1] * .5
                frho = u_u + u_d
                fgg = uu_uu + .5*ud_ud + 2*uu_ud + uu_dd
                frhogamma = u_uu + u_dd + u_ud
            else:
                fgamma = vsigma[0] - vsigma[1] * .5
                frho = u_u - u_d
                fgg = uu_uu - uu_dd
                frhogamma = u_uu - u_dd

            for i in range(nset):
                # rho1[0 ] = |b><j| z_{bj}
                # rho1[1:] = \nabla(|b><j|) z_{bj}
                rho1 = make_rho(i, ao_k1, mask, xctype)
                wv = _rks_gga_wv1(rho, rho1, (None,fgamma),
                                  (frho,frhogamma,fgg), weight)
                vmat[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

        for i in range(nset):  # for (\nabla\mu) \nu + \mu (\nabla\nu)
            vmat[i] = vmat[i] + vmat[i].swapaxes(-2,-1).conj()

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if isinstance(dms_alpha, numpy.ndarray) and dms_alpha.ndim == vmat[0].ndim:
        vmat = vmat[0]
    return numpy.asarray(vmat)


def nr_uks_fxc(ni, cell, grids, xc_code, dm0, dms, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, kpts=None, max_memory=2000,
               verbose=None):
    '''Contract UKS XC kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt` or :class:`KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D array a list of 2D arrays
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).
        rho0 : float array
            Zero-order density (and density derivative for GGA).  Giving kwargs rho0,
            vxc and fxc to improve better performance.
        vxc : float array
            First order XC derivatives
        fxc : float array
            Second order XC derivatives

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dma, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(cell, dmb, hermi)[0]

    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        dm0a, dm0b = _format_uks_dm(dm0)
        make_rho0a = ni._gen_rho_evaluator(cell, dm0a, 1)[0]
        make_rho0b = ni._gen_rho_evaluator(cell, dm0b, 1)[0]

    ao_loc = cell.ao_loc_nr()

    vmata = [0] * nset
    vmatb = [0] * nset
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if fxc is None:
                rho0a = make_rho0a(0, ao_k1, mask, xctype)
                rho0b = make_rho0b(0, ao_k1, mask, xctype)
                fxc0 = ni.eval_xc(xc_code, (rho0a,rho0b), spin=1,
                                  relativity=relativity, deriv=2,
                                  verbose=verbose)[2]
                u_u, u_d, d_d = fxc0[0].T
            else:
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T
                ip += ngrid

            for i in range(nset):
                rho1a = make_rhoa(i, ao_k1, mask, xctype)
                rho1b = make_rhob(i, ao_k1, mask, xctype)
                wv = u_u * rho1a + u_d * rho1b
                wv *= weight
                vmata[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)
                wv = u_d * rho1a + d_d * rho1b
                wv *= weight
                vmatb[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if rho0 is None:
                rho0a = make_rho0a(0, ao_k1, mask, xctype)
                rho0b = make_rho0b(0, ao_k1, mask, xctype)
            else:
                rho0a = rho0[0][:,ip:ip+ngrid]
                rho0b = rho0[1][:,ip:ip+ngrid]
            if vxc is None or fxc is None:
                vxc0, fxc0 = ni.eval_xc(xc_code, (rho0a,rho0b), spin=1,
                                        relativity=relativity, deriv=2,
                                        verbose=verbose)[1:3]
            else:
                vxc0 = (None, vxc[1][ip:ip+ngrid])
                fxc0 = (fxc[0][ip:ip+ngrid], fxc[1][ip:ip+ngrid], fxc[2][ip:ip+ngrid])
                ip += ngrid

            for i in range(nset):
                rho1a = make_rhoa(i, ao_k1, mask, xctype)
                rho1b = make_rhob(i, ao_k1, mask, xctype)
                wva, wvb = _uks_gga_wv1((rho0a,rho0b), (rho1a,rho1b),
                                        vxc0, fxc0, weight)
                vmata[i] += ni._fxc_mat(cell, ao_k1, wva, mask, xctype, ao_loc)
                vmatb[i] += ni._fxc_mat(cell, ao_k1, wvb, mask, xctype, ao_loc)

        for i in range(nset):  # for (\nabla\mu) \nu + \mu (\nabla\nu)
            vmata[i] = vmata[i] + vmata[i].swapaxes(-1,-2).conj()
            vmatb[i] = vmatb[i] + vmatb[i].swapaxes(-1,-2).conj()
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if dma.ndim == vmata[0].ndim:  # One set of DMs in the input
        vmata = vmata[0]
        vmatb = vmatb[0]
    return numpy.asarray((vmata,vmatb))

def _fxc_mat(cell, ao, wv, non0tab, xctype, ao_loc):
    shls_slice = (0, cell.nbas)

    if xctype == 'LDA' or xctype == 'HF':
        #:aow = numpy.einsum('pi,p->pi', ao, wv)
        aow = _scale_ao(ao, wv)
        mat = _dot_ao_ao(cell, ao, aow, non0tab, shls_slice, ao_loc)
    else:
        #:aow = numpy.einsum('npi,np->pi', ao, wv)
        aow = _scale_ao(ao, wv)
        mat = _dot_ao_ao(cell, ao[0], aow, non0tab, shls_slice, ao_loc)
    return mat

def cache_xc_kernel(ni, cell, grids, xc_code, mo_coeff, mo_occ, spin=0,
                    kpts=None, max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)
    ao_deriv = 0
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    nao = cell.nao_nr()
    if spin == 0:
        rho = []
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            rho.append(ni.eval_rho2(cell, ao_k1, mo_coeff, mo_occ, mask, xctype))
        rho = numpy.hstack(rho)
    else:
        rhoa = []
        rhob = []
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            rhoa.append(ni.eval_rho2(cell, ao_k1, mo_coeff[0], mo_occ[0], mask, xctype))
            rhob.append(ni.eval_rho2(cell, ao_k1, mo_coeff[1], mo_occ[1], mask, xctype))
        rho = (numpy.hstack(rhoa), numpy.hstack(rhob))
    vxc, fxc = ni.eval_xc(xc_code, rho, spin=spin, relativity=0, deriv=2,
                          verbose=0)[1:3]
    return rho, vxc, fxc


def get_rho(ni, cell, dm, grids, kpts=numpy.zeros((1,3)), max_memory=2000):
    '''Density in real space
    '''
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm)
    assert(nset == 1)
    rho = numpy.empty(grids.weights.size)
    p1 = 0
    for ao_k1, ao_k2, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, 0, kpts, None, max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = make_rho(0, ao_k1, mask, 'LDA')
    return rho


class NumInt(numint.NumInt):
    '''Generalization of pyscf's NumInt class for a single k-point shift and
    periodic images.
    '''
    def eval_ao(self, cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0,
                shls_slice=None, non0tab=None, out=None, verbose=None):
        return eval_ao(cell, coords, kpt, deriv, relativity, shls_slice,
                       non0tab, out, verbose)

    @lib.with_doc(make_mask.__doc__)
    def make_mask(self, cell, coords, relativity=0, shls_slice=None,
                  verbose=None):
        return make_mask(cell, coords, relativity, shls_slice, verbose)

    @lib.with_doc(eval_rho.__doc__)
    def eval_rho(self, cell, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
        return eval_rho(cell, ao, dm, non0tab, xctype, hermi, verbose)

    def eval_rho2(self, cell, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  verbose=None):
        return eval_rho2(cell, ao, mo_coeff, mo_occ, non0tab, xctype, verbose)

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
               kpt=None, kpts_band=None, max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix.
        See :func:`nr_rks` and :func:`nr_uks` for more details.
        '''
        if spin == 0:
            return self.nr_rks(cell, grids, xc_code, dms, hermi,
                               kpt, kpts_band, max_memory, verbose)
        else:
            return self.nr_uks(cell, grids, xc_code, dms, hermi,
                               kpt, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, hermi=0,
               kpt=numpy.zeros(3), kpts_band=None, max_memory=2000, verbose=None):
        if kpts_band is not None:
            # To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = KNumInt()
            ni.__dict__.update(self.__dict__)
            nao = dms.shape[-1]
            return ni.nr_rks(cell, grids, xc_code, dms.reshape(-1,1,nao,nao),
                             hermi, kpt.reshape(1,3), kpts_band, max_memory,
                             verbose)
        return nr_rks(self, cell, grids, xc_code, dms,
                      0, 0, hermi, kpt, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(self, cell, grids, xc_code, dms, hermi=0,
               kpt=numpy.zeros(3), kpts_band=None, max_memory=2000, verbose=None):
        if kpts_band is not None:
            # To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = KNumInt()
            ni.__dict__.update(self.__dict__)
            nao = dms[0].shape[-1]
            return ni.nr_uks(cell, grids, xc_code, dms.reshape(-1,1,nao,nao),
                             hermi, kpt.reshape(1,3), kpts_band, max_memory,
                             verbose)
        return nr_uks(self, cell, grids, xc_code, dms,
                      1, 0, hermi, kpt, kpts_band, max_memory, verbose)

    def eval_mat(self, cell, ao, weight, rho, vxc,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
        # Guess whether ao is evaluated for kpts_band.  When xctype is LDA, ao on grids
        # should be a 2D array.  For other xc functional, ao should be a 3D array.
        if ao.ndim == 2 or (xctype != 'LDA' and ao.ndim == 3):
            mat = eval_mat(cell, ao, weight, rho, vxc, non0tab, xctype, spin, verbose)
        else:
            nkpts = len(ao)
            nao = ao[0].shape[-1]
            mat = numpy.empty((nkpts,nao,nao), dtype=numpy.complex128)
            for k in range(nkpts):
                mat[k] = eval_mat(cell, ao[k], weight, rho, vxc,
                                  non0tab, xctype, spin, verbose)
        return mat
    
    def eval_mat_hss(self, cell, ao, weight, vxc,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
        # Guess whether ao is evaluated for kpts_band.  When xctype is LDA, ao on grids
        # should be a 2D array.  For other xc functional, ao should be a 3D array.
        if ao.ndim == 2 or (xctype != 'LDA' and ao.ndim == 3):
            mat = eval_mat_hss(cell, ao, weight, vxc, non0tab, xctype, spin, verbose)
        else:
            nkpts = len(ao)
            nao = ao[0].shape[-1]
            mat = numpy.empty((nkpts,nao,nao), dtype=numpy.complex128)
            for k in range(nkpts):
                mat[k] = eval_mat_hss(cell, ao[k], weight, vxc,
                                  non0tab, xctype, spin, verbose)
        return mat
    
    def eval_rho_ibp(self, cell, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
        return eval_rho_ibp(cell, ao, dm, non0tab, xctype, hermi, verbose)
    
    def _gen_rho_evaluator_ibp_pbc(self, mol, dms, hermi=0):
        if getattr(dms, 'mo_coeff', None) is not None:
            #TODO: test whether dm.mo_coeff matching dm
            raise NotImplementedError("Not implemented yet")
            mo_coeff = dms.mo_coeff
            mo_occ = dms.mo_occ
            if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
                mo_coeff = [mo_coeff]
                mo_occ = [mo_occ]
            nao = mo_coeff[0].shape[0]
            ndms = len(mo_occ)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho2(mol, ao, mo_coeff[idm], mo_occ[idm],
                                      non0tab, xctype)
        else:
            if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
                dms = [dms]
            if not hermi:
                # For eval_rho when xctype==GGA, which requires hermitian DMs
                dms = [(dm+dm.conj().T)*.5 for dm in dms]
            nao = dms[0].shape[0]
            ndms = len(dms)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho_ibp(mol, ao, dms[idm], non0tab, xctype, hermi=1)
        return make_rho, ndms, nao
    
    # def _gen_rho_evaluator_ibp_pbc(self, cell, dms, hermi=0):
    #     if getattr(dms, 'mo_coeff', None) is not None:
    #         raise NotImplementedError("Not implemented yet")
    #         mo_coeff = dms.mo_coeff
    #         mo_occ = dms.mo_occ
    #         if isinstance(dms[0], numpy.ndarray) and dms[0].ndim == 2:
    #             mo_coeff = [mo_coeff]
    #             mo_occ = [mo_occ]
    #         nao = cell.nao_nr()
    #         ndms = len(mo_occ)

    #         def make_rho(idm, ao, non0tab, xctype):
    #             return self.eval_rho2(cell, ao, mo_coeff[idm], mo_occ[idm],
    #                                   non0tab, xctype)
    #     else:
    #         if isinstance(dms[0], numpy.ndarray) and dms[0].ndim == 2:
    #             dms = [numpy.stack(dms)]
    #         #if not hermi:
    #         # Density (or response of density) is always real for DFT.
    #         # Symmetrizing DM for gamma point should not change the value of
    #         # density. However, when k-point is considered, unless dm and
    #         # dm.conj().transpose produce the same real part of density, the
    #         # symmetrization code below may be incorrect (proof is needed).
    #         #    # dm.shape = (nkpts, nao, nao)
    #         #    dms = [(dm+dm.conj().transpose(0,2,1))*.5 for dm in dms]
    #         nao = dms[0].shape[-1]
    #         ndms = len(dms)
    #         import pdb
    #         pdb.set_trace()
    #         def make_rho(idm, ao_kpts, non0tab, xctype):
    #             return self.eval_rho_ibp(cell, ao_kpts, dms[idm], non0tab, xctype,
    #                                  hermi=hermi)
    #     return make_rho, ndms, nao

    def _fxc_mat(self, cell, ao, wv, non0tab, xctype, ao_loc):
        return _fxc_mat(cell, ao, wv, non0tab, xctype, ao_loc)

    def block_loop(self, cell, grids, nao=None, deriv=0, kpt=numpy.zeros(3),
                   kpts_band=None, max_memory=2000, non0tab=None, blksize=None):
        '''Define this macro to loop over grids by blocks.
        '''
        # For UniformGrids, grids.coords does not indicate whehter grids are initialized
        if grids.non0tab is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = cell.nao
        grids_coords = grids.coords
        grids_weights = grids.weights
        ngrids = grids_coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index grids.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/(comp*2*nao*16*BLKSIZE))*BLKSIZE
            blksize = max(BLKSIZE, min(blksize, ngrids, BLKSIZE*1200))
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = 0xff
        # import pdb
        # pdb.set_trace()
        kpt = numpy.reshape(kpt, 3)
        if kpts_band is None:
            kpt1 = kpt2 = kpt
        else:
            kpt1 = kpts_band
            kpt2 = kpt

        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids_coords[ip0:ip1]
            weight = grids_weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao_k2 = self.eval_ao(cell, coords, kpt2, deriv=deriv, non0tab=non0)
            if abs(kpt1-kpt2).sum() < 1e-9:
                ao_k1 = ao_k2
            else:
                ao_k1 = self.eval_ao(cell, coords, kpt1, deriv=deriv)
            yield ao_k1, ao_k2, non0, weight, coords
            ao_k1 = ao_k2 = None

    def _gen_rho_evaluator(self, cell, dms, hermi=0):
        return numint.NumInt._gen_rho_evaluator(self, cell, dms, hermi)

    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    cache_xc_kernel  = cache_xc_kernel
    get_rho = get_rho
    nr_new_ASDP_parallel = nr_new_ASDP_parallel
    nr_new_ASDP_parallel_ibp = nr_new_ASDP_parallel_ibp
    
    def eval_xc_new_ASDP(self, xc_code, rho, spin=1, relativity=0, deriv=1, omega=None,
                verbose=None, ibp = False):
        """calculate the xc functional used in new ASDP formalism.

        Args:
            xc_code (str): xc functional name
            rho (tuple): Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            spin (int, optional): wheter spin to be polarised. Defaults to 0.
            relativity (int, optional): whether use relativity, no effect. Defaults to 0.
            deriv (int, optional): derivatives of the density matrix. Defaults to 1.
            omega ([type], optional): [description]. Defaults to None.
            verbose ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        xctype = self._xc_type(xc_code)
        deriv_new = deriv + 1
        if omega is None: omega = self.omega
        # calculate and  split the functional
        exc, vxc, fxc, kxc= self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv_new,
                                  omega, verbose)[:4]
        # vrho, vsigma, vlapl, vtau = vxc
        # v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2
        #    , v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau = fxc
        
        # transformations of old variables
        # 1st order
        vrho = vxc[0]
        v2rho2 = fxc[0]
        u, d = vrho.T
        # 2nd order
        u_u, u_d, d_d = v2rho2.T
        # import pdb
        # pdb.set_trace()
        if ibp:
            # * GGA potetianl same part!
            vsigma = vxc[1]
            v2rhosigma = fxc[1]
            v2sigma2 = fxc[2]
            s = rho[0][0] - rho[1][0]
            ngrid = rho[0][0].shape[-1]
            nablas = rho[0][1:4] - rho[1][1:4]
            nablan = rho[0][1:4] + rho[1][1:4]
            # xx,xy,xz,yy,yz,zz
            nabla2s = rho[0][4:10] - rho[1][4:10]
            nabla2n = rho[0][4:10] + rho[1][4:10]
            # initiate some output arrays
            out_n = numpy.zeros((ngrid)) 
            out_s = numpy.zeros((ngrid))  
            
            offset2 = get_2d_offset()
            offset3 = get_3d_offset()
            # ~ Third order part BEGIND
            # One of the most troblesome codes are done in get_kxc_in_s_n
            # n_s_Ns : (3, ngrid) x y z
            # s_s_Ns : (3, ngrid) x y z
            # n_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
            # s_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
            # s_Nn_Ns : (3, 3, ngrid) (x y z) times (x y z)
            # Nn_Ns_Ns : (3,6,ngrid) (x y z) times (xx xy xz yy yz zz)
            # Ns_Ns_Ns : (10, ngrid) xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
            n_s_Ns, s_s_Ns, n_s_Nn, s_s_Nn, n_Ns_Ns, s_Ns_Ns, s_Nn_Ns, \
                n_Nn_Ns, s_Nn_Nn, Nn_Ns_Ns, Ns_Ns_Ns, Nn_Nn_Ns = get_kxc_in_s_n(rho, v2rhosigma, v2sigma2, kxc) 

            # init some temeperate paremeters!  
            # This part is also troublesome!        
            # ~ Third order part Finish
            
            # ~ Second order part BEGIND
            # calculate part of the derivatives
            # wva, wvb (2D numpy array): 
            # wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
            # wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
            #                               0      1    2      3     4     5
            # ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
            #   6     7     8      9    10   11      12    13    14   
            # bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
            #  15    16    17    18    19    20   
            wva, wvb, wvrho_nrho, wvnrho_nrho =\
                uks_gga_wv0_intbypart_noweight(rho, (vrho, vsigma), (v2rho2, v2rhosigma, v2sigma2))
            # initiate some temperate variables.
            pn_Ns = numpy.zeros((3,ngrid))
            pn_Nn = numpy.zeros((3,ngrid))
            ps_Ns = numpy.zeros((3,ngrid))
            ps_Nn = numpy.zeros((3,ngrid))
            pNn_Ns = numpy.zeros((3,3,ngrid))
            pNn_Nn = numpy.zeros((6,ngrid))
            pNs_Ns = numpy.zeros((6,ngrid))
            
            # construct temporary variables
            pn = 0.5*(wva + wvb) # * include the nabla part.
            ps = 0.5*(wva - wvb) # * include the nabla part.
            pn_s = 0.25*(u_u - d_d)
            ps_s = 0.25*(u_u - 2*u_d + d_d)
            
            pn_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] + wvrho_nrho[6] + wvrho_nrho[9] )
            pn_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] + wvrho_nrho[7] + wvrho_nrho[10])
            pn_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] + wvrho_nrho[8] + wvrho_nrho[11])
            
            pn_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] + wvrho_nrho[6] - wvrho_nrho[9] )
            pn_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] + wvrho_nrho[7] - wvrho_nrho[10])
            pn_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] + wvrho_nrho[8] - wvrho_nrho[11])
            
            ps_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] - wvrho_nrho[6] + wvrho_nrho[9] )
            ps_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] - wvrho_nrho[7] + wvrho_nrho[10])
            ps_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] - wvrho_nrho[8] + wvrho_nrho[11])
            
            ps_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] - wvrho_nrho[6] - wvrho_nrho[9] )
            ps_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] - wvrho_nrho[7] - wvrho_nrho[10])
            ps_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] - wvrho_nrho[8] - wvrho_nrho[11])
            
            pNn_Ns[0,0] = (wvnrho_nrho[0] - wvnrho_nrho[15])*0.25
            pNn_Ns[0,1] = (wvnrho_nrho[1] - wvnrho_nrho[7] + wvnrho_nrho[9]  - wvnrho_nrho[16])*0.25
            pNn_Ns[0,2] = (wvnrho_nrho[2] - wvnrho_nrho[8] + wvnrho_nrho[12] - wvnrho_nrho[17])*0.25
            pNn_Ns[1,0] = (wvnrho_nrho[1] - wvnrho_nrho[9] + wvnrho_nrho[7]  - wvnrho_nrho[16])*0.25
            pNn_Ns[1,1] = (wvnrho_nrho[3] - wvnrho_nrho[18])*0.25
            pNn_Ns[1,2] = (wvnrho_nrho[4] - wvnrho_nrho[11] + wvnrho_nrho[13] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,0] = (wvnrho_nrho[2] - wvnrho_nrho[12] + wvnrho_nrho[8]  - wvnrho_nrho[17])*0.25
            pNn_Ns[2,1] = (wvnrho_nrho[4] - wvnrho_nrho[13] + wvnrho_nrho[11] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,2] = (wvnrho_nrho[5] - wvnrho_nrho[20])*0.25
            
            pNs_Ns[0] = (wvnrho_nrho[0] - wvnrho_nrho[6 ] - wvnrho_nrho[6 ] + wvnrho_nrho[15])*0.25 # xx
            pNs_Ns[1] = (wvnrho_nrho[1] - wvnrho_nrho[7 ] - wvnrho_nrho[9 ] + wvnrho_nrho[16])*0.25 # xy
            pNs_Ns[2] = (wvnrho_nrho[2] - wvnrho_nrho[8 ] - wvnrho_nrho[12] + wvnrho_nrho[17])*0.25 # xz
            pNs_Ns[3] = (wvnrho_nrho[3] - wvnrho_nrho[10] - wvnrho_nrho[10] + wvnrho_nrho[18])*0.25 # yy
            pNs_Ns[4] = (wvnrho_nrho[4] - wvnrho_nrho[11] - wvnrho_nrho[13] + wvnrho_nrho[19])*0.25 # yz
            pNs_Ns[5] = (wvnrho_nrho[5] - wvnrho_nrho[14] - wvnrho_nrho[14] + wvnrho_nrho[20])*0.25 # zz
            
            pNn_Nn[0] = (wvnrho_nrho[0] + wvnrho_nrho[6 ] + wvnrho_nrho[6 ] + wvnrho_nrho[15])*0.25 # xx
            pNn_Nn[1] = (wvnrho_nrho[1] + wvnrho_nrho[7 ] + wvnrho_nrho[9 ] + wvnrho_nrho[16])*0.25 # xy
            pNn_Nn[2] = (wvnrho_nrho[2] + wvnrho_nrho[8 ] + wvnrho_nrho[12] + wvnrho_nrho[17])*0.25 # xz
            pNn_Nn[3] = (wvnrho_nrho[3] + wvnrho_nrho[10] + wvnrho_nrho[10] + wvnrho_nrho[18])*0.25 # yy
            pNn_Nn[4] = (wvnrho_nrho[4] + wvnrho_nrho[11] + wvnrho_nrho[13] + wvnrho_nrho[19])*0.25 # yz
            pNn_Nn[5] = (wvnrho_nrho[5] + wvnrho_nrho[14] + wvnrho_nrho[14] + wvnrho_nrho[20])*0.25 # zz
            wva = wvb = wvrho_nrho = wvnrho_nrho =None
            # ~ Second order part FINISH
            
            # ~ ~ ~
            # ~ Combine the second and third part to generate final potential derivatives
            # ~ ~ ~
            # The following part will use some temerate arrays
            # This part is the potential independent of gradient
            out_n = pn[0] + s*pn_s + nablas[0]*pn_Ns[0]+ nablas[1]*pn_Ns[1]+ nablas[2]*pn_Ns[2]
            out_s = 2*ps[0] + s*ps_s + nablas[0]*ps_Ns[0]+ nablas[1]*ps_Ns[1]+ nablas[2]*ps_Ns[2]
            # ~ Frist rho dependent part.
            # ~ temp1N is a temperate array to save paremeters.
            temp1N = numpy.zeros((ngrid))
            for u in range(3):
                # ~ 1st part
                # temp1N = gt_n_Nn
                temp1N = pn_Nn[u] + s*n_s_Nn[u] + nablas[0]*n_Nn_Ns[u,0] \
                    + nablas[1]*n_Nn_Ns[u,1] + nablas[2]*n_Nn_Ns[u,2]
                # ~ n_Nn part
                out_n -= temp1N*nablan[u]
                
                # ~ 2nd part
                # temp1N = gt_s_Nn
                temp1N = 2*ps_Nn[u] + s*s_s_Nn[u] + nablas[0]*s_Nn_Ns[u,0] \
                    + nablas[1]*s_Nn_Ns[u,1] + nablas[2]*s_Nn_Ns[u,2]
                # ~ s_Nn part
                out_n -= temp1N*nablas[u]
                
                # ~ 3rd part
                for v in range(3):
                    # temp1N = gt_Nn_Nn [u,v]
                    temp1N = pNn_Nn[offset2[u,v]] + s*s_Nn_Nn[offset2[u,v]] + nablas[0]*Nn_Nn_Ns[offset2[u,v],0] \
                        + nablas[1]*Nn_Nn_Ns[offset2[u,v],1] + nablas[2]*Nn_Nn_Ns[offset2[u,v],2]

                    out_n -= temp1N*nabla2n[offset2[u,v]]
                    
                    # ~ 4th part This part should be correct.
                    # temp1N = gt_Nn_Ns [u,v]
                    temp1N = 2*pNn_Ns[u,v] + s*s_Nn_Ns[u,v] + nablas[0]*Nn_Ns_Ns[u,offset2[v,0]] \
                        + nablas[1]*Nn_Ns_Ns[u,offset2[v,1]] + nablas[2]*Nn_Ns_Ns[u,offset2[v,2]]
                        
                    out_n -= temp1N*nabla2s[offset2[u,v]]
            # ~ spin dependent part
            for u in range(3):
                # ~ 1st part 
                # temp1N = n_Ns
                temp1N = 2*pn_Ns[u] + s*n_s_Ns[u] + nablas[0]*n_Ns_Ns[offset2[u,0]] \
                    + nablas[1]*n_Ns_Ns[offset2[u,1]] + nablas[2]*n_Ns_Ns[offset2[u,2]]
                out_s -= temp1N*nablan[u]
                
                # ~ 2nd part
                # temp1N = s_Ns
                temp1N = 3*ps_Ns[u] + s*s_s_Ns[u] + nablas[0]*s_Ns_Ns[offset2[u,0]] \
                    + nablas[1]*s_Ns_Ns[offset2[u,1]] + nablas[2]*s_Ns_Ns[offset2[u,2]]
                out_s -= temp1N*nablas[u]
                
                # ~ 3rd part
                # temp1N = Nn_Ns
                for v in range(3):
                    temp1N = 2*pNn_Ns[u,v] + s*s_Nn_Ns[u,v] + nablas[0]*Nn_Ns_Ns[u,offset2[v,0]] \
                        + nablas[1]*Nn_Ns_Ns[u,offset2[v,1]] + nablas[2]*Nn_Ns_Ns[u,offset2[v,2]]
                    out_s -= temp1N*nabla2n[offset2[u,v]]
                    
                    # ~ 4th part
                    # temp1N = Ns_Ns
                    temp1N = 3*pNs_Ns[offset2[u,v]] + s*s_Ns_Ns[offset2[u,v]] + nablas[0]*Ns_Ns_Ns[offset3[u,v,0]] \
                        + nablas[1]*Ns_Ns_Ns[offset3[u,v,1]] + nablas[2]*Ns_Ns_Ns[offset3[u,v,2]]
                    out_s -= temp1N*nabla2s[offset2[u,v]]
            
            return exc, s*ps[0] + nablas[0]*ps[1] + nablas[1]*ps[2] + nablas[2]*ps[3],\
                    out_n, out_s
                        
        else:
            if xctype == 'LDA':
                # transform to s    
                s = rho[0] - rho[1]
                # construct temporary variables
                pn = 0.5*(u + d)
                ps = 0.5*(u - d)
                pn_s = 0.25*(u_u - d_d)
                ps_s = 0.25*(u_u - 2*u_d + d_d)
                # construct new variables
                n_new = pn + s*pn_s
                s_new = 2*ps + s*ps_s
                # import pdb
                # pdb.set_trace()
                return exc, s*ps, n_new, s_new
                # return exc, pn, ps
            elif xctype == 'GGA':
                # transform to s 
                vsigma = vxc[1]
                v2rhosigma = fxc[1]
                v2sigma2 = fxc[2]
                s = rho[0][0] - rho[1][0]
                ngrid = rho[0][0].shape[-1]
                nablas = rho[0][1:4] - rho[1][1:4]
                # initiate some output arrays
                out_n = numpy.zeros((4,ngrid)) 
                out_s = numpy.zeros((4,ngrid))         
                # calculate part of the derivatives
                # wva, wvb (2D numpy array): 
                # wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
                # wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
                #                               0      1    2      3     4     5
                # ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
                #   6     7     8      9    10   11      12    13    14   
                # bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
                #  15    16    17    18    19    20   
                wva, wvb, wvrho_nrho, wvnrho_nrho =\
                    uks_gga_wv0_intbypart_noweight(rho, (vrho, vsigma), (v2rho2, v2rhosigma, v2sigma2))
                # initiate some temperate variables.
                pn_Ns = numpy.zeros((3,ngrid))
                ps_Ns = numpy.zeros((3,ngrid))
                ps_Nn = numpy.zeros((3,ngrid))
                pNn_Ns = numpy.zeros((3,3,ngrid))
                pNs_Ns = numpy.zeros((6,ngrid))
                # construct temporary variables
                pn = 0.5*(wva + wvb) # * include the nabla part.
                ps = 0.5*(wva - wvb) # * include the nabla part.
                pn_s = 0.25*(u_u - d_d)
                ps_s = 0.25*(u_u - 2*u_d + d_d)
                pn_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] + wvrho_nrho[6] - wvrho_nrho[9] )
                pn_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] + wvrho_nrho[7] - wvrho_nrho[10])
                pn_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] + wvrho_nrho[8] - wvrho_nrho[11])
                
                ps_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] - wvrho_nrho[6] + wvrho_nrho[9] )
                ps_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] - wvrho_nrho[7] + wvrho_nrho[10])
                ps_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] - wvrho_nrho[8] + wvrho_nrho[11])
                
                ps_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] - wvrho_nrho[6] - wvrho_nrho[9] )
                ps_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] - wvrho_nrho[7] - wvrho_nrho[10])
                ps_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] - wvrho_nrho[8] - wvrho_nrho[11])
                
                pNn_Ns[0,0] = (wvnrho_nrho[0] - wvnrho_nrho[15])*0.25
                pNn_Ns[0,1] = (wvnrho_nrho[1] - wvnrho_nrho[7] + wvnrho_nrho[9]  - wvnrho_nrho[16])*0.25
                pNn_Ns[0,2] = (wvnrho_nrho[2] - wvnrho_nrho[8] + wvnrho_nrho[12] - wvnrho_nrho[17])*0.25
                pNn_Ns[1,0] = (wvnrho_nrho[1] - wvnrho_nrho[9] + wvnrho_nrho[7]  - wvnrho_nrho[16])*0.25
                pNn_Ns[1,1] = (wvnrho_nrho[3] - wvnrho_nrho[18])*0.25
                pNn_Ns[1,2] = (wvnrho_nrho[4] - wvnrho_nrho[11] + wvnrho_nrho[13] - wvnrho_nrho[19])*0.25
                pNn_Ns[2,0] = (wvnrho_nrho[2] - wvnrho_nrho[12] + wvnrho_nrho[8]  - wvnrho_nrho[17])*0.25
                pNn_Ns[2,1] = (wvnrho_nrho[4] - wvnrho_nrho[13] + wvnrho_nrho[11] - wvnrho_nrho[19])*0.25
                pNn_Ns[2,2] = (wvnrho_nrho[5] - wvnrho_nrho[20])*0.25
                
                pNs_Ns[0] = (wvnrho_nrho[0] - wvnrho_nrho[6 ] - wvnrho_nrho[6 ] + wvnrho_nrho[15])*0.25 # xx
                pNs_Ns[1] = (wvnrho_nrho[1] - wvnrho_nrho[7 ] - wvnrho_nrho[9 ] + wvnrho_nrho[16])*0.25 # xy
                pNs_Ns[2] = (wvnrho_nrho[2] - wvnrho_nrho[8 ] - wvnrho_nrho[12] + wvnrho_nrho[17])*0.25 # xz
                pNs_Ns[3] = (wvnrho_nrho[3] - wvnrho_nrho[10] - wvnrho_nrho[10] + wvnrho_nrho[18])*0.25 # yy
                pNs_Ns[4] = (wvnrho_nrho[4] - wvnrho_nrho[11] - wvnrho_nrho[13] + wvnrho_nrho[19])*0.25 # yz
                pNs_Ns[5] = (wvnrho_nrho[5] - wvnrho_nrho[14] - wvnrho_nrho[14] + wvnrho_nrho[20])*0.25 # zz
                
                # * construct the final functional derivatives!
                
                out_n[0] = pn[0] + s*pn_s + nablas[0]*pn_Ns[0]+ nablas[1]*pn_Ns[1]+ nablas[2]*pn_Ns[2]
                out_s[0] = 2*ps[0] + s*ps_s + nablas[0]*ps_Ns[0]+ nablas[1]*ps_Ns[1]+ nablas[2]*ps_Ns[2]
                
                out_n[1] = pn[1] + s*ps_Nn[0] + nablas[0]*pNn_Ns[0,0]+ nablas[1]*pNn_Ns[0,1]+ nablas[2]*pNn_Ns[0,2]
                out_n[2] = pn[2] + s*ps_Nn[1] + nablas[0]*pNn_Ns[1,0]+ nablas[1]*pNn_Ns[1,1]+ nablas[2]*pNn_Ns[1,2]
                out_n[3] = pn[3] + s*ps_Nn[2] + nablas[0]*pNn_Ns[2,0]+ nablas[1]*pNn_Ns[2,1]+ nablas[2]*pNn_Ns[2,2]
                out_s[1] = 2*ps[1] + s*ps_Ns[0] + nablas[0]*pNs_Ns[0] + nablas[1]*pNs_Ns[1] + nablas[2]*pNs_Ns[2]
                out_s[2] = 2*ps[2] + s*ps_Ns[1] + nablas[0]*pNs_Ns[1] + nablas[1]*pNs_Ns[3] + nablas[2]*pNs_Ns[4]
                out_s[3] = 2*ps[3] + s*ps_Ns[2] + nablas[0]*pNs_Ns[2] + nablas[1]*pNs_Ns[4] + nablas[2]*pNs_Ns[5]
                
                
                # return exc, numpy.zeros((ngrid)), pn, ps
                
                return exc, s*ps[0] + nablas[0]*ps[1] + nablas[1]*ps[2] + nablas[2]*ps[3],\
                    out_n, out_s
            
            elif xctype == 'MGGA':
                raise NotImplementedError("Meta-GGA is not implemented")
        
        

_NumInt = NumInt


class KNumInt(numint.NumInt):
    '''Generalization of pyscf's NumInt class for k-point sampling and
    periodic images.
    '''
    def __init__(self, kpts=numpy.zeros((1,3))):
        numint.NumInt.__init__(self)
        self.kpts = numpy.reshape(kpts, (-1,3))

    def eval_ao(self, cell, coords, kpts=numpy.zeros((1,3)), deriv=0, relativity=0,
                shls_slice=None, non0tab=None, out=None, verbose=None, **kwargs):
        return eval_ao_kpts(cell, coords, kpts, deriv,
                            relativity, shls_slice, non0tab, out, verbose)

    @lib.with_doc(make_mask.__doc__)
    def make_mask(self, cell, coords, relativity=0, shls_slice=None,
                  verbose=None):
        return make_mask(cell, coords, relativity, shls_slice, verbose)

    def eval_rho(self, cell, ao_kpts, dm_kpts, non0tab=None, xctype='LDA',
                 hermi=0, verbose=None):
        '''Collocate the *real* density (opt. gradients) on the real-space grid.

        Args:
            cell : Mole or Cell object
            ao_kpts : (nkpts, ngrids, nao) ndarray
                AO values at each k-point
            dm_kpts: (nkpts, nao, nao) ndarray
                Density matrix at each k-point

        Returns:
           rhoR : (ngrids,) ndarray
        '''
        nkpts = len(ao_kpts)
        rhoR = 0
        for k in range(nkpts):
            rhoR += eval_rho(cell, ao_kpts[k], dm_kpts[k], non0tab, xctype,
                             hermi, verbose)
        rhoR *= 1./nkpts
        return rhoR
    
    
    def eval_rho2(self, cell, ao_kpts, mo_coeff_kpts, mo_occ_kpts,
                  non0tab=None, xctype='LDA', verbose=None):
        nkpts = len(ao_kpts)
        rhoR = 0
        for k in range(nkpts):
            rhoR += eval_rho2(cell, ao_kpts[k], mo_coeff_kpts[k],
                              mo_occ_kpts[k], non0tab, xctype, verbose)
        rhoR *= 1./nkpts
        return rhoR

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix.
        See :func:`nr_rks` and :func:`nr_uks` for more details.
        '''
        if spin == 0:
            return self.nr_rks(cell, grids, xc_code, dms, hermi,
                               kpts, kpts_band, max_memory, verbose)
        else:
            return self.nr_uks(cell, grids, xc_code, dms, hermi,
                               kpts, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, hermi=0, kpts=None, kpts_band=None,
               max_memory=2000, verbose=None, **kwargs):
        if kpts is None:
            if 'kpt' in kwargs:
                sys.stderr.write('WARN: KNumInt.nr_rks function finds keyword '
                                 'argument "kpt" and converts it to "kpts"\n')
                kpts = kwargs['kpt']
            else:
                kpts = self.kpts
        kpts = kpts.reshape(-1,3)

        return nr_rks(self, cell, grids, xc_code, dms, 0, 0,
                      hermi, kpts, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(self, cell, grids, xc_code, dms, hermi=0, kpts=None, kpts_band=None,
               max_memory=2000, verbose=None, **kwargs):
        if kpts is None:
            if 'kpt' in kwargs:
                sys.stderr.write('WARN: KNumInt.nr_uks function finds keyword '
                                 'argument "kpt" and converts it to "kpts"\n')
                kpts = kwargs['kpt']
            else:
                kpts = self.kpts
        kpts = kpts.reshape(-1,3)

        return nr_uks(self, cell, grids, xc_code, dms, 1, 0,
                      hermi, kpts, kpts_band, max_memory, verbose)

    def eval_mat(self, cell, ao_kpts, weight, rho, vxc,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
        nkpts = len(ao_kpts)
        nao = ao_kpts[0].shape[-1]
        dtype = numpy.result_type(*ao_kpts)
        mat = numpy.empty((nkpts,nao,nao), dtype=dtype)
        for k in range(nkpts):
            mat[k] = eval_mat(cell, ao_kpts[k], weight, rho, vxc,
                              non0tab, xctype, spin, verbose)
        return mat

    def _fxc_mat(self, cell, ao_kpts, wv, non0tab, xctype, ao_loc):
        nkpts = len(ao_kpts)
        nao = ao_kpts[0].shape[-1]
        dtype = numpy.result_type(*ao_kpts)
        mat = numpy.empty((nkpts,nao,nao), dtype=dtype)
        for k in range(nkpts):
            mat[k] = _fxc_mat(cell, ao_kpts[k], wv, non0tab, xctype, ao_loc)
        return mat

    def block_loop(self, cell, grids, nao=None, deriv=0, kpts=numpy.zeros((1,3)),
                   kpts_band=None, max_memory=2000, non0tab=None, blksize=None):
        '''Define this macro to loop over grids by blocks.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = cell.nao
        grids_coords = grids.coords
        grids_weights = grids.weights
        ngrids = grids_coords.shape[0]
        nkpts = len(kpts)
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index grids.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/(comp*2*nkpts*nao*16*BLKSIZE))*BLKSIZE
            blksize = max(BLKSIZE, min(blksize, ngrids, BLKSIZE*1200))
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = 0xff
        if kpts_band is not None:
            kpts_band = numpy.reshape(kpts_band, (-1,3))
            where = [member(k, kpts) for k in kpts_band]
            where = [k_id[0] if len(k_id)>0 else None for k_id in where]

        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids_coords[ip0:ip1]
            weight = grids_weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao_k1 = ao_k2 = self.eval_ao(cell, coords, kpts, deriv=deriv, non0tab=non0)
            if kpts_band is not None:
                ao_k1 = self.eval_ao(cell, coords, kpts_band, deriv=deriv, non0tab=non0)
            yield ao_k1, ao_k2, non0, weight, coords
            ao_k1 = ao_k2 = None

    def _gen_rho_evaluator(self, cell, dms, hermi=0):
        if getattr(dms, 'mo_coeff', None) is not None:
            mo_coeff = dms.mo_coeff
            mo_occ = dms.mo_occ
            if isinstance(dms[0], numpy.ndarray) and dms[0].ndim == 2:
                mo_coeff = [mo_coeff]
                mo_occ = [mo_occ]
            nao = cell.nao_nr()
            ndms = len(mo_occ)

            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho2(cell, ao, mo_coeff[idm], mo_occ[idm],
                                      non0tab, xctype)
        else:
            if isinstance(dms[0], numpy.ndarray) and dms[0].ndim == 2:
                dms = [numpy.stack(dms)]
            #if not hermi:
            # Density (or response of density) is always real for DFT.
            # Symmetrizing DM for gamma point should not change the value of
            # density. However, when k-point is considered, unless dm and
            # dm.conj().transpose produce the same real part of density, the
            # symmetrization code below may be incorrect (proof is needed).
            #    # dm.shape = (nkpts, nao, nao)
            #    dms = [(dm+dm.conj().transpose(0,2,1))*.5 for dm in dms]
            nao = dms[0].shape[-1]
            ndms = len(dms)

            def make_rho(idm, ao_kpts, non0tab, xctype):
                return self.eval_rho(cell, ao_kpts, dms[idm], non0tab, xctype,
                                     hermi=hermi)
        return make_rho, ndms, nao
    
    
    
    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    cache_xc_kernel  = cache_xc_kernel
    get_rho = get_rho
    nr_new_ASDP_parallel = nr_new_ASDP_parallel
    
    def eval_xc_new_ASDP(self, xc_code, rho, spin=1, relativity=0, deriv=1, omega=None,
                verbose=None, ibp = False):
        """calculate the xc functional used in new ASDP formalism.

        Args:
            xc_code (str): xc functional name
            rho (tuple): Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            spin (int, optional): wheter spin to be polarised. Defaults to 0.
            relativity (int, optional): whether use relativity, no effect. Defaults to 0.
            deriv (int, optional): derivatives of the density matrix. Defaults to 1.
            omega ([type], optional): [description]. Defaults to None.
            verbose ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        xctype = self._xc_type(xc_code)
        deriv_new = deriv + 1
        if omega is None: omega = self.omega
        # calculate and  split the functional
        exc, vxc, fxc, kxc= self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv_new,
                                  omega, verbose)[:4]
        # vrho, vsigma, vlapl, vtau = vxc
        # v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2
        #    , v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau = fxc
        
        # transformations of old variables
        # 1st order
        vrho = vxc[0]
        v2rho2 = fxc[0]
        u, d = vrho.T
        # 2nd order
        u_u, u_d, d_d = v2rho2.T
        # import pdb
        # pdb.set_trace()
        if ibp:
            # * GGA potetianl same part!
            vsigma = vxc[1]
            v2rhosigma = fxc[1]
            v2sigma2 = fxc[2]
            s = rho[0][0] - rho[1][0]
            ngrid = rho[0][0].shape[-1]
            nablas = rho[0][1:4] - rho[1][1:4]
            nablan = rho[0][1:4] + rho[1][1:4]
            # xx,xy,xz,yy,yz,zz
            nabla2s = rho[0][4:10] - rho[1][4:10]
            nabla2n = rho[0][4:10] + rho[1][4:10]
            # initiate some output arrays
            out_n = numpy.zeros((ngrid)) 
            out_s = numpy.zeros((ngrid))  
            
            offset2 = get_2d_offset()
            offset3 = get_3d_offset()
            # ~ Third order part BEGIND
            # One of the most troblesome codes are done in get_kxc_in_s_n
            # n_s_Ns : (3, ngrid) x y z
            # s_s_Ns : (3, ngrid) x y z
            # n_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
            # s_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
            # s_Nn_Ns : (3, 3, ngrid) (x y z) times (x y z)
            # Nn_Ns_Ns : (3,6,ngrid) (x y z) times (xx xy xz yy yz zz)
            # Ns_Ns_Ns : (10, ngrid) xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
            n_s_Ns, s_s_Ns, n_s_Nn, s_s_Nn, n_Ns_Ns, s_Ns_Ns, s_Nn_Ns, \
                n_Nn_Ns, s_Nn_Nn, Nn_Ns_Ns, Ns_Ns_Ns, Nn_Nn_Ns = get_kxc_in_s_n(rho, v2rhosigma, v2sigma2, kxc) 

            # init some temeperate paremeters!  
            # This part is also troublesome!        
            # ~ Third order part Finish
            
            # ~ Second order part BEGIND
            # calculate part of the derivatives
            # wva, wvb (2D numpy array): 
            # wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
            # wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
            #                               0      1    2      3     4     5
            # ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
            #   6     7     8      9    10   11      12    13    14   
            # bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
            #  15    16    17    18    19    20   
            wva, wvb, wvrho_nrho, wvnrho_nrho =\
                uks_gga_wv0_intbypart_noweight(rho, (vrho, vsigma), (v2rho2, v2rhosigma, v2sigma2))
            # initiate some temperate variables.
            pn_Ns = numpy.zeros((3,ngrid))
            pn_Nn = numpy.zeros((3,ngrid))
            ps_Ns = numpy.zeros((3,ngrid))
            ps_Nn = numpy.zeros((3,ngrid))
            pNn_Ns = numpy.zeros((3,3,ngrid))
            pNn_Nn = numpy.zeros((6,ngrid))
            pNs_Ns = numpy.zeros((6,ngrid))
            
            # construct temporary variables
            pn = 0.5*(wva + wvb) # * include the nabla part.
            ps = 0.5*(wva - wvb) # * include the nabla part.
            pn_s = 0.25*(u_u - d_d)
            ps_s = 0.25*(u_u - 2*u_d + d_d)
            
            pn_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] + wvrho_nrho[6] + wvrho_nrho[9] )
            pn_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] + wvrho_nrho[7] + wvrho_nrho[10])
            pn_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] + wvrho_nrho[8] + wvrho_nrho[11])
            
            pn_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] + wvrho_nrho[6] - wvrho_nrho[9] )
            pn_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] + wvrho_nrho[7] - wvrho_nrho[10])
            pn_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] + wvrho_nrho[8] - wvrho_nrho[11])
            
            ps_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] - wvrho_nrho[6] + wvrho_nrho[9] )
            ps_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] - wvrho_nrho[7] + wvrho_nrho[10])
            ps_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] - wvrho_nrho[8] + wvrho_nrho[11])
            
            ps_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] - wvrho_nrho[6] - wvrho_nrho[9] )
            ps_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] - wvrho_nrho[7] - wvrho_nrho[10])
            ps_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] - wvrho_nrho[8] - wvrho_nrho[11])
            
            pNn_Ns[0,0] = (wvnrho_nrho[0] - wvnrho_nrho[15])*0.25
            pNn_Ns[0,1] = (wvnrho_nrho[1] - wvnrho_nrho[7] + wvnrho_nrho[9]  - wvnrho_nrho[16])*0.25
            pNn_Ns[0,2] = (wvnrho_nrho[2] - wvnrho_nrho[8] + wvnrho_nrho[12] - wvnrho_nrho[17])*0.25
            pNn_Ns[1,0] = (wvnrho_nrho[1] - wvnrho_nrho[9] + wvnrho_nrho[7]  - wvnrho_nrho[16])*0.25
            pNn_Ns[1,1] = (wvnrho_nrho[3] - wvnrho_nrho[18])*0.25
            pNn_Ns[1,2] = (wvnrho_nrho[4] - wvnrho_nrho[11] + wvnrho_nrho[13] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,0] = (wvnrho_nrho[2] - wvnrho_nrho[12] + wvnrho_nrho[8]  - wvnrho_nrho[17])*0.25
            pNn_Ns[2,1] = (wvnrho_nrho[4] - wvnrho_nrho[13] + wvnrho_nrho[11] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,2] = (wvnrho_nrho[5] - wvnrho_nrho[20])*0.25
            
            pNs_Ns[0] = (wvnrho_nrho[0] - wvnrho_nrho[6 ] - wvnrho_nrho[6 ] + wvnrho_nrho[15])*0.25 # xx
            pNs_Ns[1] = (wvnrho_nrho[1] - wvnrho_nrho[7 ] - wvnrho_nrho[9 ] + wvnrho_nrho[16])*0.25 # xy
            pNs_Ns[2] = (wvnrho_nrho[2] - wvnrho_nrho[8 ] - wvnrho_nrho[12] + wvnrho_nrho[17])*0.25 # xz
            pNs_Ns[3] = (wvnrho_nrho[3] - wvnrho_nrho[10] - wvnrho_nrho[10] + wvnrho_nrho[18])*0.25 # yy
            pNs_Ns[4] = (wvnrho_nrho[4] - wvnrho_nrho[11] - wvnrho_nrho[13] + wvnrho_nrho[19])*0.25 # yz
            pNs_Ns[5] = (wvnrho_nrho[5] - wvnrho_nrho[14] - wvnrho_nrho[14] + wvnrho_nrho[20])*0.25 # zz
            
            pNn_Nn[0] = (wvnrho_nrho[0] + wvnrho_nrho[6 ] + wvnrho_nrho[6 ] + wvnrho_nrho[15])*0.25 # xx
            pNn_Nn[1] = (wvnrho_nrho[1] + wvnrho_nrho[7 ] + wvnrho_nrho[9 ] + wvnrho_nrho[16])*0.25 # xy
            pNn_Nn[2] = (wvnrho_nrho[2] + wvnrho_nrho[8 ] + wvnrho_nrho[12] + wvnrho_nrho[17])*0.25 # xz
            pNn_Nn[3] = (wvnrho_nrho[3] + wvnrho_nrho[10] + wvnrho_nrho[10] + wvnrho_nrho[18])*0.25 # yy
            pNn_Nn[4] = (wvnrho_nrho[4] + wvnrho_nrho[11] + wvnrho_nrho[13] + wvnrho_nrho[19])*0.25 # yz
            pNn_Nn[5] = (wvnrho_nrho[5] + wvnrho_nrho[14] + wvnrho_nrho[14] + wvnrho_nrho[20])*0.25 # zz
            wva = wvb = wvrho_nrho = wvnrho_nrho =None
            # ~ Second order part FINISH
            
            # ~ ~ ~
            # ~ Combine the second and third part to generate final potential derivatives
            # ~ ~ ~
            # The following part will use some temerate arrays
            # This part is the potential independent of gradient
            out_n = pn[0] + s*pn_s + nablas[0]*pn_Ns[0]+ nablas[1]*pn_Ns[1]+ nablas[2]*pn_Ns[2]
            out_s = 2*ps[0] + s*ps_s + nablas[0]*ps_Ns[0]+ nablas[1]*ps_Ns[1]+ nablas[2]*ps_Ns[2]
            # ~ Frist rho dependent part.
            # ~ temp1N is a temperate array to save paremeters.
            temp1N = numpy.zeros((ngrid))
            for u in range(3):
                # ~ 1st part
                # temp1N = gt_n_Nn
                temp1N = pn_Nn[u] + s*n_s_Nn[u] + nablas[0]*n_Nn_Ns[u,0] \
                    + nablas[1]*n_Nn_Ns[u,1] + nablas[2]*n_Nn_Ns[u,2]
                # ~ n_Nn part
                out_n -= temp1N*nablan[u]
                
                # ~ 2nd part
                # temp1N = gt_s_Nn
                temp1N = 2*ps_Nn[u] + s*s_s_Nn[u] + nablas[0]*s_Nn_Ns[u,0] \
                    + nablas[1]*s_Nn_Ns[u,1] + nablas[2]*s_Nn_Ns[u,2]
                # ~ s_Nn part
                out_n -= temp1N*nablas[u]
                
                # ~ 3rd part
                for v in range(3):
                    # temp1N = gt_Nn_Nn [u,v]
                    temp1N = pNn_Nn[offset2[u,v]] + s*s_Nn_Nn[offset2[u,v]] + nablas[0]*Nn_Nn_Ns[offset2[u,v],0] \
                        + nablas[1]*Nn_Nn_Ns[offset2[u,v],1] + nablas[2]*Nn_Nn_Ns[offset2[u,v],2]

                    out_n -= temp1N*nabla2n[offset2[u,v]]
                    
                    # ~ 4th part This part should be correct.
                    # temp1N = gt_Nn_Ns [u,v]
                    temp1N = 2*pNn_Ns[u,v] + s*s_Nn_Ns[u,v] + nablas[0]*Nn_Ns_Ns[u,offset2[v,0]] \
                        + nablas[1]*Nn_Ns_Ns[u,offset2[v,1]] + nablas[2]*Nn_Ns_Ns[u,offset2[v,2]]
                        
                    out_n -= temp1N*nabla2s[offset2[u,v]]
            # ~ spin dependent part
            for u in range(3):
                # ~ 1st part 
                # temp1N = n_Ns
                temp1N = 2*pn_Ns[u] + s*n_s_Ns[u] + nablas[0]*n_Ns_Ns[offset2[u,0]] \
                    + nablas[1]*n_Ns_Ns[offset2[u,1]] + nablas[2]*n_Ns_Ns[offset2[u,2]]
                out_s -= temp1N*nablan[u]
                
                # ~ 2nd part
                # temp1N = s_Ns
                temp1N = 3*ps_Ns[u] + s*s_s_Ns[u] + nablas[0]*s_Ns_Ns[offset2[u,0]] \
                    + nablas[1]*s_Ns_Ns[offset2[u,1]] + nablas[2]*s_Ns_Ns[offset2[u,2]]
                out_s -= temp1N*nablas[u]
                
                # ~ 3rd part
                # temp1N = Nn_Ns
                for v in range(3):
                    temp1N = 2*pNn_Ns[u,v] + s*s_Nn_Ns[u,v] + nablas[0]*Nn_Ns_Ns[u,offset2[v,0]] \
                        + nablas[1]*Nn_Ns_Ns[u,offset2[v,1]] + nablas[2]*Nn_Ns_Ns[u,offset2[v,2]]
                    out_s -= temp1N*nabla2n[offset2[u,v]]
                    
                    # ~ 4th part
                    # temp1N = Ns_Ns
                    temp1N = 3*pNs_Ns[offset2[u,v]] + s*s_Ns_Ns[offset2[u,v]] + nablas[0]*Ns_Ns_Ns[offset3[u,v,0]] \
                        + nablas[1]*Ns_Ns_Ns[offset3[u,v,1]] + nablas[2]*Ns_Ns_Ns[offset3[u,v,2]]
                    out_s -= temp1N*nabla2s[offset2[u,v]]
            
            return exc, s*ps[0] + nablas[0]*ps[1] + nablas[1]*ps[2] + nablas[2]*ps[3],\
                    out_n, out_s
                        
        else:
            if xctype == 'LDA':
                # transform to s    
                s = rho[0] - rho[1]
                # construct temporary variables
                pn = 0.5*(u + d)
                ps = 0.5*(u - d)
                pn_s = 0.25*(u_u - d_d)
                ps_s = 0.25*(u_u - 2*u_d + d_d)
                # construct new variables
                n_new = pn + s*pn_s
                s_new = 2*ps + s*ps_s
                # import pdb
                # pdb.set_trace()
                return exc, s*ps, n_new, s_new
                # return exc, pn, ps
            elif xctype == 'GGA':
                # transform to s 
                vsigma = vxc[1]
                v2rhosigma = fxc[1]
                v2sigma2 = fxc[2]
                s = rho[0][0] - rho[1][0]
                ngrid = rho[0][0].shape[-1]
                nablas = rho[0][1:4] - rho[1][1:4]
                # initiate some output arrays
                out_n = numpy.zeros((4,ngrid)) 
                out_s = numpy.zeros((4,ngrid))         
                # calculate part of the derivatives
                # wva, wvb (2D numpy array): 
                # wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
                # wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
                #                               0      1    2      3     4     5
                # ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
                #   6     7     8      9    10   11      12    13    14   
                # bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
                #  15    16    17    18    19    20   
                wva, wvb, wvrho_nrho, wvnrho_nrho =\
                    uks_gga_wv0_intbypart_noweight(rho, (vrho, vsigma), (v2rho2, v2rhosigma, v2sigma2))
                # initiate some temperate variables.
                pn_Ns = numpy.zeros((3,ngrid))
                ps_Ns = numpy.zeros((3,ngrid))
                ps_Nn = numpy.zeros((3,ngrid))
                pNn_Ns = numpy.zeros((3,3,ngrid))
                pNs_Ns = numpy.zeros((6,ngrid))
                # construct temporary variables
                pn = 0.5*(wva + wvb) # * include the nabla part.
                ps = 0.5*(wva - wvb) # * include the nabla part.
                pn_s = 0.25*(u_u - d_d)
                ps_s = 0.25*(u_u - 2*u_d + d_d)
                pn_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] + wvrho_nrho[6] - wvrho_nrho[9] )
                pn_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] + wvrho_nrho[7] - wvrho_nrho[10])
                pn_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] + wvrho_nrho[8] - wvrho_nrho[11])
                
                ps_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] - wvrho_nrho[6] + wvrho_nrho[9] )
                ps_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] - wvrho_nrho[7] + wvrho_nrho[10])
                ps_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] - wvrho_nrho[8] + wvrho_nrho[11])
                
                ps_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] - wvrho_nrho[6] - wvrho_nrho[9] )
                ps_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] - wvrho_nrho[7] - wvrho_nrho[10])
                ps_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] - wvrho_nrho[8] - wvrho_nrho[11])
                
                pNn_Ns[0,0] = (wvnrho_nrho[0] - wvnrho_nrho[15])*0.25
                pNn_Ns[0,1] = (wvnrho_nrho[1] - wvnrho_nrho[7] + wvnrho_nrho[9]  - wvnrho_nrho[16])*0.25
                pNn_Ns[0,2] = (wvnrho_nrho[2] - wvnrho_nrho[8] + wvnrho_nrho[12] - wvnrho_nrho[17])*0.25
                pNn_Ns[1,0] = (wvnrho_nrho[1] - wvnrho_nrho[9] + wvnrho_nrho[7]  - wvnrho_nrho[16])*0.25
                pNn_Ns[1,1] = (wvnrho_nrho[3] - wvnrho_nrho[18])*0.25
                pNn_Ns[1,2] = (wvnrho_nrho[4] - wvnrho_nrho[11] + wvnrho_nrho[13] - wvnrho_nrho[19])*0.25
                pNn_Ns[2,0] = (wvnrho_nrho[2] - wvnrho_nrho[12] + wvnrho_nrho[8]  - wvnrho_nrho[17])*0.25
                pNn_Ns[2,1] = (wvnrho_nrho[4] - wvnrho_nrho[13] + wvnrho_nrho[11] - wvnrho_nrho[19])*0.25
                pNn_Ns[2,2] = (wvnrho_nrho[5] - wvnrho_nrho[20])*0.25
                
                pNs_Ns[0] = (wvnrho_nrho[0] - wvnrho_nrho[6 ] - wvnrho_nrho[6 ] + wvnrho_nrho[15])*0.25 # xx
                pNs_Ns[1] = (wvnrho_nrho[1] - wvnrho_nrho[7 ] - wvnrho_nrho[9 ] + wvnrho_nrho[16])*0.25 # xy
                pNs_Ns[2] = (wvnrho_nrho[2] - wvnrho_nrho[8 ] - wvnrho_nrho[12] + wvnrho_nrho[17])*0.25 # xz
                pNs_Ns[3] = (wvnrho_nrho[3] - wvnrho_nrho[10] - wvnrho_nrho[10] + wvnrho_nrho[18])*0.25 # yy
                pNs_Ns[4] = (wvnrho_nrho[4] - wvnrho_nrho[11] - wvnrho_nrho[13] + wvnrho_nrho[19])*0.25 # yz
                pNs_Ns[5] = (wvnrho_nrho[5] - wvnrho_nrho[14] - wvnrho_nrho[14] + wvnrho_nrho[20])*0.25 # zz
                
                # * construct the final functional derivatives!
                
                out_n[0] = pn[0] + s*pn_s + nablas[0]*pn_Ns[0]+ nablas[1]*pn_Ns[1]+ nablas[2]*pn_Ns[2]
                out_s[0] = 2*ps[0] + s*ps_s + nablas[0]*ps_Ns[0]+ nablas[1]*ps_Ns[1]+ nablas[2]*ps_Ns[2]
                
                out_n[1] = pn[1] + s*ps_Nn[0] + nablas[0]*pNn_Ns[0,0]+ nablas[1]*pNn_Ns[0,1]+ nablas[2]*pNn_Ns[0,2]
                out_n[2] = pn[2] + s*ps_Nn[1] + nablas[0]*pNn_Ns[1,0]+ nablas[1]*pNn_Ns[1,1]+ nablas[2]*pNn_Ns[1,2]
                out_n[3] = pn[3] + s*ps_Nn[2] + nablas[0]*pNn_Ns[2,0]+ nablas[1]*pNn_Ns[2,1]+ nablas[2]*pNn_Ns[2,2]
                out_s[1] = 2*ps[1] + s*ps_Ns[0] + nablas[0]*pNs_Ns[0] + nablas[1]*pNs_Ns[1] + nablas[2]*pNs_Ns[2]
                out_s[2] = 2*ps[2] + s*ps_Ns[1] + nablas[0]*pNs_Ns[1] + nablas[1]*pNs_Ns[3] + nablas[2]*pNs_Ns[4]
                out_s[3] = 2*ps[3] + s*ps_Ns[2] + nablas[0]*pNs_Ns[2] + nablas[1]*pNs_Ns[4] + nablas[2]*pNs_Ns[5]
                
                
                # return exc, numpy.zeros((ngrid)), pn, ps
                
                return exc, s*ps[0] + nablas[0]*ps[1] + nablas[1]*ps[2] + nablas[2]*ps[3],\
                    out_n, out_s
            
            elif xctype == 'MGGA':
                raise NotImplementedError("Meta-GGA is not implemented")
        
        

_KNumInt = KNumInt
