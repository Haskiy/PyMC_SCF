#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2021-01-18 09:05:05
LastEditTime: 2021-06-15 10:48:06
LastEditors: Pu Zhichen
Description: 
    Numerical integration utils
FilePath: \undefinedd:\PKU_msi\pyMC\gksmc\numint_gksmc.py

 May the force be with you!
'''
# from functools import wraps
import time
from pyscf.dft import numint
import numpy
from pyscf.dft.gen_grid import BLKSIZE
from pyMC import tools as tools_hss
from pyMC.lib import Spoints

def timer_no_clock(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = time.process_time(), time.perf_counter()
        return rec._t0, rec._w0
    else:
        rec._t0 = time.process_clock()
        return rec._t0

def uks_gga_wv0_intbypart(rho, vxc, fxc, weight):
    """Calculate 

    Args:
        rho (2D numpy array): density
        vxc (tuple consist of 2 2D numpy arrays): [description]
        fxc (tuple consist of 3 2D numpy arrays): [description]
        weight (1D numpy array): numerical weights

    Returns:
        wva, wvb (2D numpy array): derivatives of rhoa, rhob
        wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
        wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
        ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
        bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
    """
    wva, wvb, wvrho_nrho, wvnrho_nrho = uks_gga_wv0_intbypart_noweight(rho, vxc, fxc)
    wva[:] = wva[:]*weight
    wvb[:] = wvb[:]*weight
    wvrho_nrho[:] = wvrho_nrho[:]*weight
    wvnrho_nrho[:] = wvnrho_nrho[:]*weight
    return wva, wvb, wvrho_nrho, wvnrho_nrho

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

    if (len(vxc)==2):
        return wva, wvb, wvrho_nrho, wvnrho_nrho
    else:
        tauu,taud = vxc[2].T
        wvtaua = tauu
        wvtaub = taud
        uu_tauu, uu_taud, ud_tauu, ud_taud, dd_tauu, dd_taud = fxc[3].T
        
        wvnrho_tau = numpy.empty((12,ngrid))
        wvnrho_tau[0:3] = rhoa[1:4] * (uu_tauu * 2)
        wvnrho_tau[0:3]+= rhob[1:4] * (ud_tauu)
        wvnrho_tau[3:6] = rhoa[1:4] * (uu_taud * 2)
        wvnrho_tau[3:6]+= rhob[1:4] * (ud_taud)
        wvnrho_tau[6:9] = rhob[1:4] * (dd_tauu * 2)
        wvnrho_tau[6:9]+= rhoa[1:4] * (ud_taud)
        wvnrho_tau[9:12] = rhob[1:4] * (dd_taud * 2)
        wvnrho_tau[9:12]+= rhoa[1:4] * (ud_taud)
        return wva, wvb, wvrho_nrho, wvnrho_nrho, wvtaua,wvtaub, wvnrho_tau

def cal_Bxc(rhot, NX, Vxc):
    wva, wvb, wvrho_nrho, wvnrho_nrho = Vxc
    rho, Mx, My, Mz = rhot
    ngrid = wva.shape[-1]
    Bxc = numpy.zeros((ngrid))
    Bxc = 0.5*(wva[0]-wvb[0])
    A = rho[1]*(wvrho_nrho[0]-wvrho_nrho[3]+wvrho_nrho[6]-wvrho_nrho[9])*0.25
    A+= rho[2]*(wvrho_nrho[1]-wvrho_nrho[4]+wvrho_nrho[7]-wvrho_nrho[10])*0.25
    A+= rho[3]*(wvrho_nrho[2]-wvrho_nrho[5]+wvrho_nrho[8]-wvrho_nrho[11])*0.25

    s = NX[0]*Mx + NX[1]*My + NX[2]*Mz
    B = s[1]*(wvrho_nrho[0]-wvrho_nrho[3]-wvrho_nrho[6]+wvrho_nrho[9])*0.25
    B+= s[2]*(wvrho_nrho[1]-wvrho_nrho[4]-wvrho_nrho[7]+wvrho_nrho[10])*0.25
    B+= s[3]*(wvrho_nrho[2]-wvrho_nrho[5]-wvrho_nrho[8]+wvrho_nrho[11])*0.25

    C = rho[4]*(wvnrho_nrho[0]-wvnrho_nrho[15])*0.25
    C+= rho[5]*(wvnrho_nrho[1]+wvnrho_nrho[7]-wvnrho_nrho[9]-wvnrho_nrho[16])*0.25
    C+= rho[6]*(wvnrho_nrho[2]+wvnrho_nrho[8]-wvnrho_nrho[12]-wvnrho_nrho[17])*0.25
    C+= rho[5]*(wvnrho_nrho[1]+wvnrho_nrho[9]-wvnrho_nrho[7]-wvnrho_nrho[16])*0.25
    C+= rho[7]*(wvnrho_nrho[3]-wvnrho_nrho[18])*0.25
    C+= rho[8]*(wvnrho_nrho[4]+wvnrho_nrho[11]-wvnrho_nrho[13]-wvnrho_nrho[19])*0.25
    C+= rho[6]*(wvnrho_nrho[2]+wvnrho_nrho[12]-wvnrho_nrho[8]-wvnrho_nrho[17])*0.25
    C+= rho[8]*(wvnrho_nrho[4]+wvnrho_nrho[13]-wvnrho_nrho[11]-wvnrho_nrho[19])*0.25
    C+= rho[9]*(wvnrho_nrho[5]-wvnrho_nrho[20])*0.25

    D = s[4]*(wvnrho_nrho[0]-wvnrho_nrho[6 ]-wvnrho_nrho[6 ]+wvnrho_nrho[15])*0.25
    D+= s[5]*(wvnrho_nrho[1]-wvnrho_nrho[7 ]-wvnrho_nrho[9 ]+wvnrho_nrho[16])*0.25
    D+= s[6]*(wvnrho_nrho[2]-wvnrho_nrho[8 ]-wvnrho_nrho[12]+wvnrho_nrho[17])*0.25
    D+= s[5]*(wvnrho_nrho[1]-wvnrho_nrho[9 ]-wvnrho_nrho[7 ]+wvnrho_nrho[16])*0.25
    D+= s[7]*(wvnrho_nrho[3]-wvnrho_nrho[10]-wvnrho_nrho[10]+wvnrho_nrho[18])*0.25
    D+= s[8]*(wvnrho_nrho[4]-wvnrho_nrho[11]-wvnrho_nrho[13]+wvnrho_nrho[19])*0.25
    D+= s[6]*(wvnrho_nrho[2]-wvnrho_nrho[12]-wvnrho_nrho[8 ]+wvnrho_nrho[17])*0.25
    D+= s[8]*(wvnrho_nrho[4]-wvnrho_nrho[13]-wvnrho_nrho[11]+wvnrho_nrho[19])*0.25
    D+= s[9]*(wvnrho_nrho[5]-wvnrho_nrho[14]-wvnrho_nrho[14]+wvnrho_nrho[20])*0.25

    Bxc = Bxc - A - B - C - D
    return Bxc

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
    
def cal_Wxc(rhot, NX, Vxc): 
    wva, wvb, wvrho_nrho, wvnrho_nrho = Vxc
    rho, Mx, My, Mz = rhot
    Wxc = 0.5*(wva[0]+wvb[0])

    A = rho[1]*(wvrho_nrho[0]+wvrho_nrho[3]+wvrho_nrho[6]+wvrho_nrho[9])*0.25
    A+= rho[2]*(wvrho_nrho[1]+wvrho_nrho[4]+wvrho_nrho[7]+wvrho_nrho[10])*0.25
    A+= rho[3]*(wvrho_nrho[2]+wvrho_nrho[5]+wvrho_nrho[8]+wvrho_nrho[11])*0.25

    s = NX[0]*Mx + NX[1]*My + NX[2]*Mz
    B = s[1]*(wvrho_nrho[0]+wvrho_nrho[3]-wvrho_nrho[6]-wvrho_nrho[9])*0.25
    B+= s[2]*(wvrho_nrho[1]+wvrho_nrho[4]-wvrho_nrho[7]-wvrho_nrho[10])*0.25
    B+= s[3]*(wvrho_nrho[2]+wvrho_nrho[5]-wvrho_nrho[8]-wvrho_nrho[11])*0.25

    C = rho[4]*(wvnrho_nrho[0]+wvnrho_nrho[6 ]+wvnrho_nrho[6 ]+wvnrho_nrho[15])*0.25
    C+= rho[5]*(wvnrho_nrho[1]+wvnrho_nrho[7 ]+wvnrho_nrho[9 ]+wvnrho_nrho[16])*0.25
    C+= rho[6]*(wvnrho_nrho[2]+wvnrho_nrho[8 ]+wvnrho_nrho[12]+wvnrho_nrho[17])*0.25
    C+= rho[5]*(wvnrho_nrho[1]+wvnrho_nrho[9 ]+wvnrho_nrho[7 ]+wvnrho_nrho[16])*0.25
    C+= rho[7]*(wvnrho_nrho[3]+wvnrho_nrho[10]+wvnrho_nrho[10]+wvnrho_nrho[18])*0.25
    C+= rho[8]*(wvnrho_nrho[4]+wvnrho_nrho[11]+wvnrho_nrho[13]+wvnrho_nrho[19])*0.25
    C+= rho[6]*(wvnrho_nrho[2]+wvnrho_nrho[12]+wvnrho_nrho[8 ]+wvnrho_nrho[17])*0.25
    C+= rho[8]*(wvnrho_nrho[4]+wvnrho_nrho[13]+wvnrho_nrho[11]+wvnrho_nrho[19])*0.25
    C+= rho[9]*(wvnrho_nrho[5]+wvnrho_nrho[14]+wvnrho_nrho[14]+wvnrho_nrho[20])*0.25

    D = s[4]*(wvnrho_nrho[0]-wvnrho_nrho[6 ]+wvnrho_nrho[6 ]-wvnrho_nrho[15])*0.25
    D+= s[5]*(wvnrho_nrho[1]-wvnrho_nrho[7 ]+wvnrho_nrho[9 ]-wvnrho_nrho[16])*0.25
    D+= s[6]*(wvnrho_nrho[2]-wvnrho_nrho[8 ]+wvnrho_nrho[12]-wvnrho_nrho[17])*0.25
    D+= s[5]*(wvnrho_nrho[1]-wvnrho_nrho[9 ]+wvnrho_nrho[7 ]-wvnrho_nrho[16])*0.25
    D+= s[7]*(wvnrho_nrho[3]-wvnrho_nrho[10]+wvnrho_nrho[10]-wvnrho_nrho[18])*0.25
    D+= s[8]*(wvnrho_nrho[4]-wvnrho_nrho[11]+wvnrho_nrho[13]-wvnrho_nrho[19])*0.25
    D+= s[6]*(wvnrho_nrho[2]-wvnrho_nrho[12]+wvnrho_nrho[8 ]-wvnrho_nrho[17])*0.25
    D+= s[8]*(wvnrho_nrho[4]-wvnrho_nrho[13]+wvnrho_nrho[11]-wvnrho_nrho[19])*0.25
    D+= s[9]*(wvnrho_nrho[5]-wvnrho_nrho[14]+wvnrho_nrho[14]-wvnrho_nrho[20])*0.25

    Wxc = Wxc - A - B - C - D
    # Wxc = - A - C
    return Wxc        
        
def get_kxc_in_s_n_kernel(rho, v2rhosigma, v2sigma2, kxc, LIBXCT_factor=None):
    """This subroutine calculates the integral by part potential, using 
       AGEC (new AGC method, also known as multi-collinear appraoch) for GGA functionals

    Args:
        rho (numpy.array): ((den_u,grad_xu,grad_yu,grad_zu, xxu, xyu, xzu, yyu, yzu, zzu)
                            (den_d,grad_xd,grad_yd,grad_zd, xxd, xyd, xzd, yyd, yzd, zzd))
        v2rhosigma ([type]): v2rhosigma[:,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
        v2sigma2 ([type]): [description]
        kxc ([type]): [description]
        LIBXCT_factor: THRESHOLD for LIBXCT_factor

    Returns:
        n_n_Ns : (3, ngrid) x y z
        n_s_Ns : (3, ngrid) x y z
        s_s_Ns : (3, ngrid) x y z
        n_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
        s_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
        s_Nn_Ns : (3, 3, ngrid) (x y z) times (x y z)
        Nn_Ns_Ns : (3,6,ngrid) (x y z) times (xx xy xz yy yz zz)
        Ns_Ns_Ns : (10, ngrid) xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
    """
    
    idx_u_polar = rho[1][0] <= LIBXCT_factor
    idx_d_polar = rho[0][0] <= LIBXCT_factor
            
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
    n_n_Ns = numpy.zeros((3, ngrid))
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
    
    # ^ NUMERICASL STABLE
    u_u_Nu[:,idx_d_polar] = 0.0
    u_d_Nu[:,idx_d_polar] = 0.0
    d_d_Nu[:,idx_d_polar] = 0.0
    u_u_Nd[:,idx_d_polar] = 0.0
    u_d_Nd[:,idx_d_polar] = 0.0
    u_d_Nu[:,idx_u_polar] = 0.0
    d_d_Nu[:,idx_u_polar] = 0.0
    u_u_Nd[:,idx_u_polar] = 0.0
    u_d_Nd[:,idx_u_polar] = 0.0
    d_d_Nd[:,idx_u_polar] = 0.0
    # ^ NUMERICASL STABLE
    
    # ! get the final output paremeters
    n_n_Ns[0] = (u_u_Nu[0] - u_u_Nd[0] + 2*u_d_Nu[0] - 2*u_d_Nd[0] + d_d_Nu[0] - d_d_Nd[0])*0.125
    n_n_Ns[1] = (u_u_Nu[1] - u_u_Nd[1] + 2*u_d_Nu[1] - 2*u_d_Nd[1] + d_d_Nu[1] - d_d_Nd[1])*0.125
    n_n_Ns[2] = (u_u_Nu[2] - u_u_Nd[2] + 2*u_d_Nu[2] - 2*u_d_Nd[2] + d_d_Nu[2] - d_d_Nd[2])*0.125
    
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
        
    # ^ NUMERICASL STABLE
    u_Nu_Nu[:,idx_d_polar] = 0.0
    d_Nu_Nu[:,idx_d_polar] = 0.0
    u_Nu_Nd[:,:,idx_d_polar] = 0.0
    d_Nu_Nd[:,:,idx_d_polar] = 0.0
    u_Nd_Nd[:,idx_d_polar] = 0.0
    d_Nu_Nu[:,idx_u_polar] = 0.0
    u_Nu_Nd[:,:,idx_u_polar] = 0.0
    d_Nu_Nd[:,:,idx_u_polar] = 0.0
    u_Nd_Nd[:,idx_u_polar] = 0.0
    d_Nd_Nd[:,idx_u_polar] = 0.0
    # ^ NUMERICASL STABLE        
        
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
    # ^ NUMERICAL STABLE
    Nu_Nu_Nu[:,idx_d_polar] = 0.0
    Nu_Nu_Nd[:,:,idx_d_polar] = 0.0
    Nd_Nd_Nu[:,:,idx_d_polar] = 0.0
    Nu_Nu_Nd[:,:,idx_u_polar] = 0.0
    Nd_Nd_Nu[:,:,idx_u_polar] = 0.0
    Nd_Nd_Nd[:,idx_u_polar] = 0.0
    # ^ NUMERICAL STABLE
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
    
    return n_n_Ns, n_s_Ns, s_s_Ns, n_s_Nn, s_s_Nn, n_Ns_Ns, s_Ns_Ns, s_Nn_Ns, \
        n_Nn_Ns, s_Nn_Nn, Nn_Ns_Ns, Ns_Ns_Ns, Nn_Nn_Ns
    
def eval_rho_intbypart(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    r'''Calculate the electron density for LDA functional, and the density
    derivatives for GGA functional.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or (5,N,nao) for meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA, ao[0] is AO value
            and ao[1:3] are the AO gradients.  If xctype is meta-GGA, ao[4:10]
            are second derivatives of ao values.
        dm : 2D array
            Density matrix

    Kwargs:
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of the return density.
        hermi : bool
            dm is hermitian or not
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        2D array of (10,N) to store density and "density derivatives" for x,y,z components
        and "2nd-order density derivatives" xx,xy,xz,yy,yz,zz if xctype = GGA; 

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_ao(mol, coords, deriv=0)
    >>> dm = numpy.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> dm = dm + dm.T
    >>> rho, dx_rho, dy_rho, dz_rho = eval_rho(mol, ao, dm, xctype='LDA')
    '''
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    if not hermi:
        # (D + D.T)/2 because eval_rho computes 2*(|\nabla i> D_ij <j|) instead of
        # |\nabla i> D_ij <j| + |i> D_ij <\nabla j| for efficiency
        dm = (dm + dm.conj().T) * .5
    # import pdb
    # pdb.set_trace()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    # offset saves the offset of the rho, which is
    # offset[i,j] means i+1 --> x,y,z j+1-->x,y,z
    offset = numpy.zeros((3,3),dtype = numpy.int8)
    offset[0,0] = 4
    offset[0,1] = 5
    offset[0,2] = 6
    offset[1,1] = 7
    offset[1,2] = 8
    offset[2,2] = 9
    if xctype == 'LDA' or xctype == 'HF':
        raise NotImplementedError('LDA is not using Integrate by part')
    elif xctype in ('GGA', 'NLC'):
        rho = numpy.empty((10,ngrids))
        c0 = numint._dot_ao_dm(mol, ao[0], dm.real, non0tab, shls_slice, ao_loc)
        #:rho[0] = numpy.einsum('pi,pi->p', c0, ao[0])
        # rho[0] = numint._contract_rho(c0, ao[0].astype(numpy.complex128))
        rho[0] = numint._contract_rho(c0, ao[0])
        rho[4:10] = 0.0
        for i in range(1, 4):
            #:rho[i] = numpy.einsum('pi,pi->p', c0, ao[i])
            # rho[i] = numint._contract_rho(c0, ao[i].astype(numpy.complex128))
            rho[i] = numint._contract_rho(c0, ao[i])
            rho[i] *= 2 # *2 for +c.c. in the next two lines
            c1 = numint._dot_ao_dm(mol, ao[i], dm.T.real, non0tab, shls_slice, ao_loc)
            for j in range(i,4):
                # rho[offset[i-1,j-1]] = numint._contract_rho(c1, ao[j].astype(numpy.complex128))
                rho[offset[i-1,j-1]] = numint._contract_rho(c1, ao[j])
        for i in range(4,10):
            # rho[i] += numint._contract_rho(c0, ao[i].astype(numpy.complex128))
            rho[i] += numint._contract_rho(c0, ao[i])
            rho[i] *= 2
    else: # meta-GGA
        raise NotImplementedError('meta-GGA is not implemented in Integrate by part')
    # import pdb
    # pdb.set_trace()
    return rho

def nr_uks_intbyparts(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
           max_memory=2000, verbose=None):
    '''Calculate UKS XC functional and potential matrix on given meshgrids
    for a set of density matrices

    Args:
        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : a list of 2D arrays
            A list of density matrices, stored as (alpha,alpha,...,beta,beta,...)

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of (alpha,beta) electrons generated by numerical integration.
        excsum is the XC functional value.
        vmat is the XC potential matrix for (alpha,beta) spin.

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
    # import pdb
    # pdb.set_trace()
    xctype = ni._xc_type(xc_code)
    if xctype == 'NLC':
        raise(NotImplementedError('NLC'))

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dma, dmb = dms
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator_ibp(mol, dma, hermi, True)[:2]
    make_rhob       = ni._gen_rho_evaluator_ibp(mol, dmb, hermi, True)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((2,nset,nao,nao), dtype=numpy.result_type(dma, dmb))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                vrho = vxc[0]
                den = rho_a * weight
                nelec[0,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                den = rho_b * weight
                nelec[1,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)

                # *.5 due to +c.c. in the end
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,0], out=aow)
                aow = numint._scale_ao(ao, .5*weight*vrho[:,0], out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,1], out=aow)
                aow = numint._scale_ao(ao, .5*weight*vrho[:,1], out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = vrho = None
    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, 2, max_memory):
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, 'GGA')
                rho_b = make_rhob(idm, ao, mask, 'GGA')
                exc, vxc, fxc = ni.eval_xc(xc_code, (rho_a[:4], rho_b[:4]), spin=1,
                                      relativity=relativity, deriv=2,
                                      verbose=verbose)[:3]
                den = rho_a[0]*weight
                nelec[0,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                den = rho_b[0]*weight
                nelec[1,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                ngrid = rho_a.shape[-1]
                NX0 = numpy.zeros((3,ngrid))
                NX0[2] = 1.0
                Mx = numpy.zeros((10,ngrid))
                My = numpy.zeros((10,ngrid))
                Mz = numpy.zeros((10,ngrid))
                Mz = rho_a-rho_b

                wva, wvb, wvrho_nrho, wvnrho_nrho \
                    = ni.uks_gga_wv0_intbypart((rho_a,rho_b), vxc, fxc, weight)
                Bxc = cal_Bxc((rho_a+rho_b, Mx, My, Mz), NX0,
                    (wva, wvb, wvrho_nrho, wvnrho_nrho))
                Wxc = cal_Wxc((rho_a+rho_b, Mx, My, Mz), NX0,
                    (wva, wvb, wvrho_nrho, wvnrho_nrho))
                wvp = 0.5*(wva[0]+wvb[0])
                
                vaa = wvp + Bxc + Wxc
                vbb = wvp - Bxc + Wxc
                #:aow = numpy.einsum('npi,np->pi', ao, wva, out=aow)
                aow = numint._scale_ao(ao[0], vaa*0.5, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
                aow = numint._scale_ao(ao[0], vbb*0.5, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = wva = wvb = None
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho_a[0]*weight
                nelec[0,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                den = rho_b[0]*weight
                nelec[1,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)

                wva, wvb = numint._uks_gga_wv0((rho_a,rho_b), vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao[:4], wva, out=aow)
                aow = numint._scale_ao(ao[:4], wva, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('npi,np->pi', ao[:4], wvb, out=aow)
                aow = numint._scale_ao(ao[:4], wvb, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

# FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
# Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
                wv = (.25 * weight * vtau[:,0]).reshape(-1,1)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
                wv = (.25 * weight * vtau[:,1]).reshape(-1,1)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = vrho = wva = wvb = None

    for i in range(nset):
        vmat[0,i] = vmat[0,i] + vmat[0,i].conj().T
        vmat[1,i] = vmat[1,i] + vmat[1,i].conj().T
    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]
    return nelec, excsum, vmat

def nr_uks_lc(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
           max_memory=2000, verbose=None, LIBXCT_factor = None):
    '''Calculate UKS XC functional with Non-collinear framework
    and potential matrix on given meshgrids for a set of density matrices.
    This subroutine uses Tri-directions for non-collinear spin systems.

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
    start = time.process_time()
    xctype = ni._xc_type(xc_code)
    if xctype == 'NLC':
        NotImplementedError

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmaa, dmab, dmba, dmbb = dms
    #print(dmaa, dmab, dmba, dmbb)
    nao = dmaa.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dmaa, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmbb, hermi)[0]
    # make_rho        = ni._gen_rho_evaluator(mol, dmaa+dmbb, hermi)[0]
    # make_rhoMz      = ni._gen_rho_evaluator(mol, dmaa-dmbb, hermi)[0]
    make_rhoMx      = ni._gen_rho_evaluator(mol, dmba+dmab, hermi)[0]
    make_rhoMy      = ni._gen_rho_evaluator(mol, -dmba*1.0j+dmab*1.0j, hermi)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    # vmat will save in the order(alpha_alpha,alpha_beta,beta_alpha,beta_beta).
    vmat = numpy.zeros((4,nset,nao,nao), dtype=numpy.complex128)
    NX = None
    aow = None
    M_tot_out = 0.0
    ao_deriv = 1
    sum_M_tot = 0
    if xctype == 'LDA':
        ao_deriv = 0
        ipart = 0
        numpy.save('coords',grids.coords)
        numpy.save('weights',grids.weights)
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # ao = ao + 0.0j
            aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            ipart = ipart + 1
            sum_M = 0.0
            for idm in range(nset):
                # calculate densities and M vector
                # Cause we need \nabla M informations, so we use 
                rho_aa = make_rhoa(idm, ao, mask, xctype)
                rho_bb = make_rhob(idm, ao, mask, xctype)
                Mx = make_rhoMx(idm, ao, mask, xctype)
                My = make_rhoMy(idm, ao, mask, xctype)
                # Mx (4,ngrid) Mx, nablax Mx, nablay Mx, nablaz Mx
                Mz = numpy.real(rho_aa - rho_bb)
                
                numpy.save('Mx_part'+str(ipart),Mx)
                numpy.save('My_part'+str(ipart),My)
                numpy.save('Mz_part'+str(ipart),Mz)
                rhop = 0.5*(rho_aa + rho_bb)
                ngrid = rho_aa.shape[0]
                # get Three principle directions
                s = numpy.zeros((ngrid))
                M_norm = numpy.zeros((ngrid))
                NX = numpy.zeros((3,ngrid))
                
                # calculate functional derivatives
                M_norm = numpy.sqrt(Mx[:]*Mx[:] + My[:]*My[:] + Mz[:]*Mz[:])
                s[:] = M_norm[:]
                # s = rho_aa - rho_bb
                for i in range(Mx.shape[-1]):
                    sum_M += Mx[i]/numpy.sqrt(3) + My[i]/numpy.sqrt(3) + Mz[i]/numpy.sqrt(3)
                for icount in range(ngrid):
                    # NX[2,icount] = 1.0
                    if M_norm[icount] <= 1.0e-100:
                        continue
                    else:
                        NX[0,icount] = Mx[icount]/M_norm[icount]
                        NX[1,icount] = My[icount]/M_norm[icount]
                        NX[2,icount] = Mz[icount]/M_norm[icount]
                rho_a = rhop + 0.5*s
                rho_b = rhop - 0.5*s
                
                idx_u_polar = rho_b <= LIBXCT_factor
                idx_d_polar = rho_a <= LIBXCT_factor
                
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                    relativity=relativity, deriv=1,
                                    verbose=verbose)[:2]
                
                vrho = vxc[0]
                den = rho_a * weight
                nelec[0,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                den = rho_b * weight
                nelec[1,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                vrho[idx_d_polar,0] = 0.0
                vrho[idx_u_polar,1] = 0.0
                # M_tot
                M_tot = numpy.sqrt(Mx[0]*Mx[0] + My[0]*My[0] + Mz[0]*Mz[0])*weight
                M_tot_out+= M_tot.sum()
                numpy.save('M_tot'+str(ipart),numpy.asarray([M_tot_out]))

                vmm = (vrho[:,0]-vrho[:,1])*0.5
                vpp = (vrho[:,0]+vrho[:,1])*0.5
                vaa = vpp + vmm*NX[2,:]
                vbb = vpp - vmm*NX[2,:]
                vab_r = vmm*NX[0,:]
                vab_i = vmm*NX[1,:] 

                # *.5 due to +c.c. in the end
                aow = numint._scale_ao(ao, .5*weight*vaa, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                aow = numint._scale_ao(ao, .5*weight*vab_r, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                vmat[2,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                aow = numint._scale_ao(ao, .5*weight*vab_i, out=aow)
                vmat[1,idm] -= numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)*1.0j
                vmat[2,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)*1.0j
                aow = numint._scale_ao(ao, .5*weight*vbb, out=aow)
                vmat[3,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = vrho = None
            sum_M_tot += sum_M
        print('sum_M_tot', end='')
        print(sum_M_tot)
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # ao = ao + 0.0j
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                # calculate densities and M vector
                rho_aa = make_rhoa(idm, ao, mask, xctype)
                rho_bb = make_rhob(idm, ao, mask, xctype)
                Mx = make_rhoMx(idm, ao, mask, 'GGA')
                My = make_rhoMy(idm, ao, mask, 'GGA')
                # Mx (4,ngrid) Mx, nablax Mx, nablay Mx, nablaz Mx
                Mz = numpy.real(rho_aa - rho_bb)
                # numpy.savetxt('coords_'+'.txt',grids.coords)
                # numpy.savetxt('Mx_'+istep+'.txt',Mx)
                # numpy.savetxt('My_'+istep+'.txt',My)
                # numpy.savetxt('Mz_'+istep+'.txt',Mz)
                rhop = 0.5*(rho_aa + rho_bb)
                ngrid = rho_aa.shape[1]
                # get Three principle directions
                NX = numpy.zeros((3, ngrid))
                s = numpy.zeros((4, ngrid))
                # calculate functional derivatives
                M_norm = numpy.sqrt(Mx[0,:]*Mx[0,:] + My[0,:]*My[0,:] + Mz[0,:]*Mz[0,:])
                s[0,:] = M_norm[:]
                for icount in range(ngrid):
                    if M_norm[icount] <= 1.0e-9:
                        continue
                    else:
                        s[1,icount] = (Mx[0,icount]*Mx[1,icount] \
                            + My[0,icount]*My[1,icount] \
                            + Mz[0,icount]*Mz[1,icount])/M_norm[icount]
                        s[2,icount] = (Mx[0,icount]*Mx[2,icount] \
                            + My[0,icount]*My[2,icount] \
                            + Mz[0,icount]*Mz[2,icount])/M_norm[icount]
                        s[3,icount] = (Mx[0,icount]*Mx[3,icount] \
                            + My[0,icount]*My[3,icount] \
                            + Mz[0,icount]*Mz[3,icount])/M_norm[icount]
                        NX[0,icount] = Mx[0,icount]/M_norm[icount]
                        NX[1,icount] = My[0,icount]/M_norm[icount]
                        NX[2,icount] = Mz[0,icount]/M_norm[icount]
                rho_a = rhop + s*0.5
                rho_b = rhop - s*0.5
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                    relativity=relativity, deriv=1,
                                    verbose=verbose)[:2]
                den = rho_a[0]*weight
                excsum[idm] += numpy.dot(den, exc)
                nelec[0,idm] += den.sum()
                den = rho_b[0]*weight
                excsum[idm] += numpy.dot(den, exc)
                nelec[1,idm] += den.sum()

                wva, wvb = numint._uks_gga_wv0((rho_a,rho_b), vxc, weight)
                wvm = (wva - wvb)*0.5
                wvp = (wva + wvb)*0.5
                wvaahss1 = wvp[:,:] + wvm*NX[2,:]
                wvbbhss1 = wvp[:,:] - wvm*NX[2,:]
                wvaahss2 = numpy.zeros((3,ngrid))
                # wvbbhss2 = numpy.zeros((3,ngrid))
                for icount in range(ngrid):
                    if M_norm[icount] <= 1.0e-9:
                        continue
                    wvaahss2[:,icount] = wvm[1:4,icount]*(-s[1:4,icount]*NX[2,icount]\
                        + Mz[1:4,icount])/M_norm[icount]
                wvabhss_i_1 = -wvm*NX[1,:]
                wvabhss_r_1 = wvm*NX[0,:]
                wvabhss_r_2 = numpy.zeros((3,ngrid))
                wvabhss_i_2 = numpy.zeros((3,ngrid))
                wvaahss2[0,:] = wvaahss2[0,:] + wvaahss2[1,:] + wvaahss2[2,:]
                for icount in range(ngrid):
                    if M_norm[icount] <= 1.0e-9:
                        continue
                    wvabhss_r_2[:,icount] = wvm[1:4,icount]*(-s[1:4,icount]*NX[0,icount] \
                        + Mx[1:4,icount])/M_norm[icount]
                    wvabhss_i_2[:,icount] = wvm[1:4,icount]*( s[1:4,icount]*NX[1,icount] \
                        - My[1:4,icount])/M_norm[icount]
                wvabhss_r_2[0,:] = wvabhss_r_2[0,:] + wvabhss_r_2[1,:] + wvabhss_r_2[2,:]
                wvabhss_i_2[0,:] = wvabhss_i_2[0,:] + wvabhss_i_2[1,:] + wvabhss_i_2[2,:]
                # Rearrange the functional derivative part.
                wvaahss1[0,:] = wvaahss1[0,:] + wvaahss2[0,:]*0.5
                wvbbhss1[0,:] = wvbbhss1[0,:] - wvaahss2[0,:]*0.5
                wvabhss_r_1[0,:] = wvabhss_r_1[0,:] + wvabhss_r_2[0,:]*0.5
                wvabhss_i_1[0,:] = wvabhss_i_1[0,:] + wvabhss_i_2[0,:]*0.5
                # contraction
                aow = numint._scale_ao(ao, wvaahss1, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # real part
                aow = numint._scale_ao(ao, wvabhss_r_1, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # aow = numint._scale_ao(ao[0], wvabhss_r_2[0]*0.5, out=aow)
                # vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # imag part
                aow = numint._scale_ao(ao, wvabhss_i_1, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                vmat[2,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                # aow = numint._scale_ao(ao[0], wvabhss_i_2[0]*0.5, out=aow)
                # vmat[1,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                # vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
                aow = numint._scale_ao(ao, wvbbhss1, out=aow)
                vmat[3,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # aow = numint._scale_ao(ao[0], wvaahss2[0]*0.5, out=aow)
                # vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # vmat[3,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = wva = wvb = None
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA method in Multi-directions framework')
    for i in range(nset):
        vmat[0,i] = vmat[0,i] + vmat[0,i].conj().T
        # vtemp = vmat[1,i].copy()
        # vmat[1,i] = vmat[1,i] + vmat[2,i].conj().T
        # vmat[2,i] = vmat[2,i] + vtemp.conj().T
        vmat[1,i] = vmat[1,i] + vmat[2,i].conj().T
        vmat[2,i] = vmat[1,i].conj().T
        vmat[3,i] = vmat[3,i] + vmat[3,i].conj().T

        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

        end = time.process_time()
        print('Running time for Non-collinear UKS: %s Seconds'%(end-start))
    return nelec, excsum, vmat

def nr_uksmc_rt(ni, rt, mol, grids, xc_code, dms, relativity=0, hermi=0,
           max_memory=2000, verbose=None):
    '''Calculate UKS XC functional with Multi-directions framework
    and potential matrix on given meshgrids for a set of density matrices.
    This subroutine uses Tri-directions for non-collinear spin systems.

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
    raise NotImplementedError("The SD subroutine is not implemented yet for multi-collinear approach!")
    # ~ Real time initial
    M_atm = numpy.zeros((mol.natm,3))
    # ~ end
    xctype = ni._xc_type(xc_code)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dmaa, dmab, dmba, dmbb = dms

    nao = dmaa.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dmaa.real, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmbb.real, hermi)[0]
    make_rhoMx      = ni._gen_rho_evaluator(mol, (dmba+dmab).real, hermi)[0]
    make_rhoMy      = ni._gen_rho_evaluator(mol, (-dmba*1.0j+dmab*1.0j).real, hermi)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    # vmat will save in the order(alpha_alpha,alpha_beta,beta_alpha,beta_beta).
    vmat = numpy.zeros((4,nset,nao,nao), dtype=numpy.complex128)
    NX = None
    aow = None
    ao_deriv = 1
    if xctype == 'LDA':
        ao_deriv = 1
        ipart = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            ipart += 1
            # ~ real time
            if rt._istep == 0:
                grids2atom = tools_hss.partition2atom.partition2atom_rt(mol.atom_coords(), coords)
                for i in range(rt.rtmf.mol.natm):
                    rt._idx_part[i][ipart] = numpy.where(grids2atom == i)[0]
            # ~ end
            for idm in range(nset):
                # calculate densities and M vector
                # Cause we need \nabla M informations, so we use 
                rho_aa = make_rhoa(idm, ao, mask, 'GGA')
                rho_bb = make_rhob(idm, ao, mask, 'GGA')
                Mx = make_rhoMx(idm, ao, mask, 'GGA')
                My = make_rhoMy(idm, ao, mask, 'GGA')
                # Mx (4,ngrid) Mx, nablax Mx, nablay Mx, nablaz Mx
                Mz = numpy.real(rho_aa - rho_bb)
                ngrid = rho_aa.shape[1]
                ni.M = numpy.zeros((3,ngrid))
                ni.M[0] = Mx[0]
                ni.M[1] = My[0]
                ni.M[2] = Mz[0]
                # ~ real time
                for iatm in range(mol.natm):
                    M_atm[iatm,0]+= (Mx[0,rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                    M_atm[iatm,1]+= (My[0,rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                    M_atm[iatm,2]+= (Mz[0,rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                # ~ end
                # numpy.save('Mx_part'+str(ipart),Mx[:4,:])
                # numpy.save('My_part'+str(ipart),My[:4,:])
                # numpy.save('Mz_part'+str(ipart),Mz[:4,:])
                rhop = 0.5*(rho_aa + rho_bb)
                # get Three principle directions
                NX = numpy.zeros((3, 3, ngrid))
                NX = ni.eval_NX_SA(Mx, My, Mz, weight)
                s = numpy.zeros((4, ngrid))
                vaa_factor = numpy.zeros((ngrid))
                vbb_factor = numpy.zeros((ngrid))
                vab_r_factor = numpy.zeros((ngrid))
                vab_i_factor = numpy.zeros((ngrid))
                # calculate functional derivatives
                # numpy.save('NXI'  +str(ipart),NX[0,:,:])
                # numpy.save('NXII' +str(ipart),NX[1,:,:])
                # numpy.save('NXIII'+str(ipart),NX[2,:,:])
                Bxc_tot = numpy.zeros((3,ngrid))
                for dhss in range(0,3):
                    s[0,0:ngrid] = 0.5*(Mx[0,0:ngrid]*NX[dhss,0,0:ngrid]
                            + My[0,0:ngrid]*NX[dhss,1,0:ngrid]
                            + Mz[0,0:ngrid]*NX[dhss,2,0:ngrid])
                    rho_ahss = rhop + s
                    rho_bhss = rhop - s
                    exc, vxc = ni.eval_xc(xc_code, (rho_ahss, rho_bhss), spin=1,
                                        relativity=relativity, deriv=1,
                                        verbose=verbose)[:2]
                    vrho = vxc[0]
                    den = rho_ahss[0] * weight
                    excsum[idm] += numpy.dot(den, exc)
                    den = rho_bhss[0] * weight
                    excsum[idm] += numpy.dot(den, exc)

                    vmm = (vrho[:,0]-vrho[:,1])*0.5
                    vpp = (vrho[:,0]+vrho[:,1])*0.5
                    vaa_factor+= vpp + vmm*NX[dhss,2,:]
                    vbb_factor+= vpp - vmm*NX[dhss,2,:]
                    vab_r_factor+= vmm*NX[dhss,0,:]
                    vab_i_factor+= vmm*NX[dhss,1,:]
                    Bxc = vmm

                    Bxc_tot[0] += Bxc*NX[dhss,0]
                    Bxc_tot[1] += Bxc*NX[dhss,1]
                    Bxc_tot[2] += Bxc*NX[dhss,2]
                   
                    rho_ahss = rho_bhss = exc = vxc = vrho = None
                # numpy.save('Bxc_tot_part'+str(ipart),Bxc_tot)
                # ! Double counting part!
                rho_ahss = rhop
                rho_bhss = rhop
                exc, vxc = ni.eval_xc(xc_code, (rho_ahss, rho_bhss), spin=1,
                                        relativity=relativity, deriv=1,
                                        verbose=verbose)[:2]
                vrho = vxc[0]
                exc = exc*2
                den = rho_ahss[0] * weight
                excsum[idm] -= numpy.dot(den, exc)
                den = rho_bhss[0] * weight
                excsum[idm] -= numpy.dot(den, exc)

                vpp = vrho[:,0]+vrho[:,1]
                vaa_factor-= vpp
                vbb_factor-= vpp

                # * aow[ngrid,nao]
                aow = numint._scale_ao(ao[0], .5*weight*vaa_factor, out=aow)
                # print(aow.shape)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,1], out=aow)
                aow = numint._scale_ao(ao[0], .5*weight*vab_r_factor, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                aow = numint._scale_ao(ao[0], .5*weight*vab_i_factor, out=aow)
                vmat[1,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                aow = numint._scale_ao(ao[0], .5*weight*vbb_factor, out=aow)
                vmat[3,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                
                rho_ahss = rho_bhss = exc = vxc = vrho = None
                # Nelectron
                den = rho_aa[0]*weight
                nelec[0,idm] += den.sum()
                den = rho_bb[0]*weight
                nelec[1,idm] += den.sum()
    elif xctype == 'GGA':
        ao_deriv = 1
        ipart = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            ipart += 1
            # ~ real time
            if rt._istep == 0:
                grids2atom = tools_hss.partition2atom.partition2atom_rt(mol.atom_coords(), coords)
                for i in range(rt.rtmf.mol.natm):
                    rt._idx_part[i][ipart] = numpy.where(grids2atom == i)[0]
            # ~ end
            for idm in range(nset):
                # calculate densities and M vector
                rho_aa = make_rhoa(idm, ao, mask, 'GGA')
                rho_bb = make_rhob(idm, ao, mask, 'GGA')
                Mx = make_rhoMx(idm, ao, mask, 'GGA')
                My = make_rhoMy(idm, ao, mask, 'GGA')
                # Mx (4,ngrid) Mx, nablax Mx, nablay Mx, nablaz Mx
                Mz = numpy.real(rho_aa - rho_bb)
                ngrid = rho_aa.shape[1]
                ni.M = numpy.zeros((3,ngrid))
                ni.M[0] = Mx[0]
                ni.M[1] = My[0]
                ni.M[2] = Mz[0]
                # ~ real time
                for iatm in range(mol.natm):
                    M_atm[iatm,0]+= (Mx[0,rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                    M_atm[iatm,1]+= (My[0,rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                    M_atm[iatm,2]+= (Mz[0,rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                # ~ end
                # numpy.save('Mx_part'+str(ipart)+'.txt',Mx[:4,:])
                # numpy.save('My_part'+str(ipart)+'.txt',My[:4,:])
                # numpy.save('Mz_part'+str(ipart)+'.txt',Mz[:4,:])
                # numpy.savetxt('grids.txt',rho_aa + rho_bb)
                rhop = 0.5*(rho_aa + rho_bb)
                # get Three principle directions
                NX = numpy.zeros((3, 3, ngrid))
                # start_Nx = time.clock()
                NX = ni.eval_NX_SA(Mx, My, Mz, weight)

                s = numpy.zeros((4, ngrid))
                wvaahss_factor = numpy.zeros((4,ngrid))
                wvbbhss_factor = numpy.zeros((4,ngrid))
                wvabhss_i_factor = numpy.zeros((4,ngrid))
                wvabhss_r_factor = numpy.zeros((4,ngrid))
                # numpy.savetxt('NXI'+debug+'.txt',NX[0,:,:])
                # numpy.savetxt('NXII'+debug+'.txt',NX[1,:,:])
                # numpy.savetxt('NXIII'+debug+'.txt',NX[2,:,:])
                # calculate functional derivatives
                for dhss in range(0,3):
                    s[:,0:ngrid] = 0.5*(Mx[:,0:ngrid]*NX[dhss,0,0:ngrid]
                            + My[:,0:ngrid]*NX[dhss,1,0:ngrid]
                            + Mz[:,0:ngrid]*NX[dhss,2,0:ngrid])
                    rho_ahss = rhop + s
                    rho_bhss = rhop - s
                    exc, vxc = ni.eval_xc(xc_code, (rho_ahss, rho_bhss), spin=1,
                                        relativity=relativity, deriv=1,
                                        verbose=verbose)[:2]
                    den = rho_ahss[0]*weight
                    excsum[idm] += numpy.dot(den, exc)
                    den = rho_bhss[0]*weight
                    excsum[idm] += numpy.dot(den, exc)

                    wva, wvb = numint._uks_gga_wv0((rho_ahss,rho_bhss), vxc, weight)
                    wvm = (wva - wvb)*0.5
                    wvp = (wva + wvb)*0.5

                    wvaahss_factor+= wvp[:,:] + wvm[0:4,:]*NX[dhss,2,:]
                    wvbbhss_factor+= wvp[:,:] - wvm[0:4,:]*NX[dhss,2,:]
                    wvabhss_i_factor+= wvm*NX[dhss,1,:]
                    wvabhss_r_factor+= wvm*NX[dhss,0,:]

                    rho_ahss = rho_bhss = exc = vxc = wva = wvb = None
                # Double counting part!
                rho_ahss = rhop
                rho_bhss = rhop
                exc, vxc = ni.eval_xc(xc_code, (rho_ahss, rho_bhss), spin=1,
                                    relativity=relativity, deriv=1,
                                    verbose=verbose)[:2]
                exc = exc*2.0
                den = rho_ahss[0]*weight
                excsum[idm] -= numpy.dot(den, exc)
                den = rho_bhss[0]*weight
                excsum[idm] -= numpy.dot(den, exc)
                # Nelectron
                den = rho_aa[0]*weight
                nelec[0,idm] += den.sum()
                den = rho_bb[0]*weight
                nelec[1,idm] += den.sum()
                # M_tot

                wva, wvb = numint._uks_gga_wv0((rho_ahss,rho_bhss), vxc, weight)
                wvp = wva + wvb
                wvaahss_factor-= wvp
                wvbbhss_factor-= wvp
                #:aow = numpy.einsum('npi,np->pi', ao, wva, out=aow)
                aow = numint._scale_ao(ao, wvaahss_factor, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # real part
                aow = numint._scale_ao(ao, wvabhss_r_factor, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # imag part
                aow = numint._scale_ao(ao, wvabhss_i_factor, out=aow)
                vmat[1,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
                aow = numint._scale_ao(ao, wvbbhss_factor, out=aow)
                vmat[3,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho_ahss = rho_bhss = exc = vxc = wva = wvb = None
                # aow = numint._scale_ao(ao, wvp, out=aow)
                # vmat[0,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # vmat[3,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho_ahss = rho_bhss = exc = vxc = wva = wvb = aow = None
                rho_aa = rho_bb = Mx = My = Mz = NX = s = rhop = None
    elif xctype == 'MGGA':
        raise NotImplementedError("Only non-relative no ibp LDA and ibp GGA have been implemented")
    for i in range(nset):
        # vmat[2,i] = vmat[2,i].T
        # Note that all the gradients are calculated by conjugate, thus terms without
        # gradients are producted by 0.5!
        vmat[0,i] = vmat[0,i] + vmat[0,i].conj().T
        # vtemp = vmat[1,i].copy()
        # vmat[1,i] = vmat[1,i] + vmat[2,i].conj().T
        # vmat[2,i] = vmat[2,i] + vtemp.conj().T
        vmat[1,i] = vmat[1,i] + vmat[2,i].conj().T
        vmat[2,i] = vmat[1,i].conj().T
        vmat[3,i] = vmat[3,i] + vmat[3,i].conj().T

    # ~ real time
    rt._M = M_atm
    # ~ end
    vmat = vmat[:,0]
    nelec = nelec.reshape(2)
    excsum = excsum[0]

    return nelec, excsum, vmat

def nr_uksmc_intbyparts_rt(ni, rt, mol, grids, xc_code, dms, relativity=0, hermi=0,
           max_memory=2000, verbose=None):
    '''Calculate UKS XC functional with Multi-directions framework
    and potential matrix on given meshgrids for a set of density matrices.
    This subroutine uses Tri-directions for non-collinear spin systems.

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
    raise NotImplementedError("The SD subroutine is not implemented yet for multi-collinear approach!")
    # import pdb
    # pdb.set_trace()
    # * Real time initial
    M_atm = numpy.zeros((mol.natm,3))
    # * end
    xctype = ni._xc_type(xc_code)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmaa, dmab, dmba, dmbb = dms
    #print(dmaa, dmab, dmba, dmbb)
    nao = dmaa.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator_ibp(mol, dmaa, hermi,
     True)[:2]
    make_rhob       = ni._gen_rho_evaluator_ibp(mol, dmbb, hermi, True)[0]
    make_rhoMx      = ni._gen_rho_evaluator_ibp(mol, dmba+dmab, hermi, True)[0]
    make_rhoMy      = ni._gen_rho_evaluator_ibp(mol, -dmba*1.0j+dmab*1.0j, hermi, True)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    # vmat will save in the order(alpha_alpha,alpha_beta,beta_alpha,beta_beta).
    vmat = numpy.zeros((4,nset,nao,nao), dtype=numpy.complex128)
    M_tot_out = 0.0
    NX = None
    aow = None
    ao_deriv = 2
    
    ao_deriv=2
    ipart = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
        ipart += 1
        # ~ real time
        if rt._istep == 0:
            grids2atom = tools_hss.partition2atom.partition2atom_rt(mol.atom_coords(), coords)
            for i in range(rt.rtmf.mol.natm):
                rt._idx_part[i][ipart] = numpy.where(grids2atom == i)[0]
        # ~ end
        
        for idm in range(nset):
            # calculate densities and M vector
            rho_aa = make_rhoa(idm, ao, mask, 'GGA')
            rho_bb = make_rhob(idm, ao, mask, 'GGA')
            Mx = make_rhoMx(idm, ao, mask, 'GGA')
            My = make_rhoMy(idm, ao, mask, 'GGA')
            # Mx (4,ngrid) Mx, nablax Mx, nablay Mx, nablaz Mx
            Mz = numpy.real(rho_aa - rho_bb)
            ngrid = rho_aa.shape[1]
            ni.M = numpy.zeros((3,ngrid))
            ni.M[0] = Mx[0]
            ni.M[1] = My[0]
            ni.M[2] = Mz[0]
            # ~ real time
            for iatm in range(mol.natm):
                M_atm[iatm,0]+= (Mx[0,rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                M_atm[iatm,1]+= (My[0,rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                M_atm[iatm,2]+= (Mz[0,rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
            # ~ end
            # numpy.save('Mx_part'+str(ipart),Mx[:4,:])
            # numpy.save('My_part'+str(ipart),My[:4,:])
            # numpy.save('Mz_part'+str(ipart),Mz[:4,:])

            rhop = 0.5*(rho_aa + rho_bb)
            # get Three principle directions
            NX = numpy.zeros((3, 3, ngrid))
            NX = ni.eval_NX_SA_opt(Mx[:4,:], My[:4,:], Mz[:4,:], weight)

            s = numpy.zeros((10, ngrid))
            vaa_factor = numpy.zeros((ngrid))
            vbb_factor = numpy.zeros((ngrid))
            vabi_factor = numpy.zeros((ngrid))
            vabr_factor = numpy.zeros((ngrid))
            M_tot = numpy.zeros((ngrid))
            # numpy.save('NXI'  +str(ipart),NX[0,:,:])
            # numpy.save('NXII' +str(ipart),NX[1,:,:])
            # numpy.save('NXIII'+str(ipart),NX[2,:,:])
            Bxc_tot = numpy.zeros((3,ngrid))
            for dhss in range(0,3):
                s[:,0:ngrid] = 0.5*(Mx[:,0:ngrid]*NX[dhss,0,0:ngrid]
                        + My[:,0:ngrid]*NX[dhss,1,0:ngrid]
                        + Mz[:,0:ngrid]*NX[dhss,2,0:ngrid])
                rho_ahss = rhop + s
                rho_bhss = rhop - s
                exc, vxc, fxc = ni.eval_xc(xc_code, (rho_ahss[:4], rho_bhss[:4]), spin=1,
                                    relativity=relativity, deriv=2,
                                    verbose=verbose)[:3]

                den = rho_ahss[0]*weight
                excsum[idm] += numpy.dot(den, exc)
                den = rho_bhss[0]*weight
                excsum[idm] += numpy.dot(den, exc)

                wva, wvb, wvrho_nrho, wvnrho_nrho \
                    = ni.uks_gga_wv0_intbypart_noweight((rho_ahss,rho_bhss), vxc, fxc)

                Bxc = cal_Bxc((rhop*2, Mx, My, Mz), NX[dhss],
                    (wva, wvb, wvrho_nrho, wvnrho_nrho))
                Wxc = cal_Wxc((rhop*2, Mx, My, Mz), NX[dhss],
                    (wva, wvb, wvrho_nrho, wvnrho_nrho))
                Bxc_tot[0] += Bxc*NX[dhss,0]
                Bxc_tot[1] += Bxc*NX[dhss,1]
                Bxc_tot[2] += Bxc*NX[dhss,2]
                wvp = 0.5*(wva[0]+wvb[0])
                # for i in range(Mx.shape[-1]):
                #     sum_Bxc += Bxc[i]/numpy.sqrt(3)*NX[dhss,0,i] \
                #         + Bxc[i]/numpy.sqrt(3)*NX[dhss,1,i] + Bxc[i]/numpy.sqrt(3)*NX[dhss,2,i]
                # numpy.save('Bxc_d'+str(dhss)+'_part'+str(ipart),Bxc)
                
                vaa_factor += wvp + Bxc*NX[dhss,2] + Wxc 
                vbb_factor += wvp - Bxc*NX[dhss,2] + Wxc 
                vabr_factor += Bxc*NX[dhss,0]
                vabi_factor += Bxc*NX[dhss,1]

                wva = wvb = wvrho_nrho = wvnrho_nrho = None
                Bxc = None
                Wxc = None
            # Double counting part!
            # numpy.save('Bxc_tot_part'+str(ipart),Bxc_tot)
            rho_ahss = rhop.copy()
            rho_bhss = rhop.copy()
            exc, vxc, fxc = ni.eval_xc(xc_code, (rho_ahss[:4], rho_bhss[:4]), spin=1,
                                    relativity=relativity, deriv=2,
                                    verbose=verbose)[:3]
            exc = exc*2.0
            den = rho_ahss[0]*weight
            excsum[idm] -= numpy.dot(den, exc)
            den = rho_bhss[0]*weight
            excsum[idm] -= numpy.dot(den, exc)
            # Nelectron
            den = rho_aa[0]*weight
            nelec[0,idm] += den.sum()
            den = rho_bb[0]*weight
            nelec[1,idm] += den.sum()
            # M_tot
            M_tot = numpy.sqrt(Mx[0,:]*Mx[0,:] + My[0,:]*My[0,:] + Mz[0,:]*Mz[0,:])*weight
            M_tot_out+= M_tot.sum()

            wva, wvb, wvrho_nrho, wvnrho_nrho = ni.uks_gga_wv0_intbypart_noweight((rho_ahss,rho_bhss), vxc, fxc)
            NX0 = numpy.zeros((3,ngrid))

            Wxc = cal_Wxc((rho_ahss+rho_bhss, Mx, My, Mz), NX0,
                (wva, wvb, wvrho_nrho, wvnrho_nrho))

            vtmp= wva[0] + wvb[0] + 2*Wxc
            vaa_factor -= vtmp
            vbb_factor -= vtmp
 
            aow = numint._scale_ao(ao[0], vaa_factor *weight, out=aow)
            vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            # real part
            aow = numint._scale_ao(ao[0], vabr_factor *weight, out=aow)
            vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            # imag part
            aow = numint._scale_ao(ao[0], vabi_factor *weight, out=aow)
            vmat[1,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
            vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
            #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
            aow = numint._scale_ao(ao[0], vbb_factor *weight, out=aow)
            vmat[3,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

            rho_ahss = rho_bhss = exc = vxc = wva = wvb = None

            rho_ahss = rho_bhss = exc = vxc = wva = wvb = aow = None
            rho_aa = rho_bb = Mx = My = Mz = NX = s = rhop = None
            NX0 = wvp = vaa = None

    vmat = vmat[:,0]
    nelec = nelec.reshape(2)
    excsum = excsum[0]

    # ~ real time
    rt._M = M_atm
    # ~ end

    return nelec, excsum, vmat

def nr_ghf_rho_rt(ni, rt, mol, grids, xc_code, dms, relativity=0, hermi=0,
           max_memory=2000, verbose=None):
    '''Calculate UKS XC functional with Multi-directions framework
    and potential matrix on given meshgrids for a set of density matrices.
    This subroutine uses Tri-directions for non-collinear spin systems.

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
    raise NotImplementedError("The SD subroutine is not implemented yet for multi-collinear approach!")
    # * Real time initial
    M_atm = numpy.zeros((mol.natm,3))
    # * end
    
    dmaa, dmab, dmba, dmbb = dms

    nao = dmaa.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dmaa.real, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmbb.real, hermi)[0]
    make_rhoMx      = ni._gen_rho_evaluator(mol, (dmba+dmab).real, hermi)[0]
    make_rhoMy      = ni._gen_rho_evaluator(mol, (-dmba*1.0j+dmab*1.0j).real, hermi)[0]

    ao_deriv = 0
    aow = None
    ipart = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
        ipart += 1
        # ~ real time
        if rt._istep == 0:
            grids2atom = tools_hss.partition2atom.partition2atom_rt(mol.atom_coords(), coords)
            for i in range(rt.rtmf.mol.natm):
                rt._idx_part[i][ipart] = numpy.where(grids2atom == i)[0]
        # ~ end
        for idm in range(nset):
            # calculate densities and M vector
            # Cause we need \nabla M informations, so we use 
            rho_aa = make_rhoa(idm, ao, mask, 'LDA')
            rho_bb = make_rhob(idm, ao, mask, 'LDA')
            Mx = make_rhoMx(idm, ao, mask, 'LDA')
            My = make_rhoMy(idm, ao, mask, 'LDA')
            # Mx (4,ngrid) Mx, nablax Mx, nablay Mx, nablaz Mx
            Mz = numpy.real(rho_aa - rho_bb)
            
            # ~ real time
            for iatm in range(mol.natm):
                M_atm[iatm,0]+= (Mx[rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                M_atm[iatm,1]+= (My[rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
                M_atm[iatm,2]+= (Mz[rt._idx_part[iatm][ipart]]*weight[rt._idx_part[iatm][ipart]]).sum()
            # ~ end
                
    # ~ real time
    rt._M = M_atm
    # ~ end
    
def MC_LDA_parallel_kernel(Mx, My, Mz, NX, rhop, xc_code, ni, index, ngrid, weight, factor,
                            relativity=0, verbose=None, LIBXCT_factor=None):
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
        # idx_small = numpy.abs(numpy.abs(rhop) - numpy.abs(s)) <=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        rho_ahss = rhop + s
        rho_bhss = rhop - s
        exc, exc_cor, vxcn, vxcs = ni.eval_xc_new_MC(xc_code, (rho_ahss, rho_bhss), spin=1,
                            relativity=relativity, deriv=1,
                            verbose=verbose, LIBXCT_factor = LIBXCT_factor)
        # den = rho_ahss * weight
        # import pdb
        # pdb.set_trace()
        # ! Note that for energy, factor must product here!
        exctot += exc*factor[idrct]*rhop*2*weight
        # den = rho_bhss * weight
        # exctot += numpy.dot(den, exc)*factor[idrct]
        exctot += weight*exc_cor*factor[idrct]
        
        vaa+= (vxcn + vxcs*NX[idrct,2])*factor[idrct]
        vbb+= (vxcn - vxcs*NX[idrct,2])*factor[idrct]
        vab_r+= vxcs*NX[idrct,0]*factor[idrct]
        vab_i+= vxcs*NX[idrct,1]*factor[idrct]
        
        Bxc[0] += vxcs*NX[idrct,0]*factor[idrct]
        Bxc[1] += vxcs*NX[idrct,1]*factor[idrct]
        Bxc[2] += vxcs*NX[idrct,2]*factor[idrct]

    return exctot, vaa, vbb, vab_r, vab_i, Bxc


def MC_GGA_parallel_kernel(Mx, My, Mz, NX, rhop, xc_code, ni, index, ngrid, weight, factor,
                           relativity=0, verbose=None, LIBXCT_factor=None):
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
        exc, exc_cor, vxcn, vxcs = ni.eval_xc_new_MC(xc_code, (rho_ahss, rho_bhss), spin=1,
                            relativity=relativity, deriv=1,
                            verbose=verbose, LIBXCT_factor = LIBXCT_factor)
        # den = rho_ahss * weight
        # import pdb
        # pdb.set_trace()
        # ! Note that for energy, factor must product here!
        exctot += exc*factor[idrct]*rhop[0]*2*weight
        # den = rho_bhss * weight
        # exctot += numpy.dot(den, exc)*factor[idrct]
        exctot += weight*exc_cor*factor[idrct]
        
        # wva, wvb = numint._uks_gga_wv0((rho_ahss,rho_bhss), vxc, weight)
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


def nr_mc_parallel(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
           max_memory=2000, Ndirect=1454, LIBXCT_factor=1.0E-10, 
           MSL_factor = None, verbose=None, ncpu = None):
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
    start = time.process_time()
    xctype = ni._xc_type(xc_code)
    
    import math
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    nsbatch = math.ceil(Ndirect/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, Ndirect-nsbatch, nsbatch)]
    if NX_list[-1][-1] < Ndirect:
        NX_list.append((NX_list[-1][-1], Ndirect))
    # import pdb
    # pdb.set_trace()
    NX,factor = ni.Spoints.make_sph_samples(Ndirect)
    # ~ init some parameters in parallel
    pool = multiprocessing.Pool()
    # ~ parallel run spherical average
        
    if xctype == 'NLC':
        NotImplementedError

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmaa, dmab, dmba, dmbb = dms
    nao = dmaa.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dmaa.real, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmbb.real, hermi)[0]
    make_rhoMx      = ni._gen_rho_evaluator(mol, (dmba+dmab).real, hermi)[0]
    make_rhoMy      = ni._gen_rho_evaluator(mol, (-dmba*1.0j+dmab*1.0j).real, hermi)[0]

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
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # ao = ao + 0.0j
            aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            ipart += 1
            for idm in range(nset):
                # calculate densities and M vector
                # Cause we need \nabla M informations, so we use 
                rho_aa = make_rhoa(idm, ao, mask, xctype)
                rho_bb = make_rhob(idm, ao, mask, xctype)
                Mx = make_rhoMx(idm, ao, mask, xctype)
                My = make_rhoMy(idm, ao, mask, xctype)
                Mz = numpy.real(rho_aa - rho_bb)

                ngrid = rho_aa.shape[0]
                ni.M = numpy.zeros((3,ngrid))
                ni.M[0] = Mx
                ni.M[1] = My
                ni.M[2] = Mz

                numpy.save('Mx_part'+str(ipart),Mx)
                numpy.save('My_part'+str(ipart),My)
                numpy.save('Mz_part'+str(ipart),Mz)
                numpy.save('ao'+str(ipart),ao)
                # ! Debug use

                rhop = 0.5*(rho_aa + rho_bb)
                
                numpy.save('rho'+str(ipart),rhop)
                vaa_factor = numpy.zeros((ngrid))
                vbb_factor = numpy.zeros((ngrid))
                vab_r_factor = numpy.zeros((ngrid))
                vab_i_factor = numpy.zeros((ngrid))
                Bxc_tot = numpy.zeros((3,ngrid))
                exc_tot = 0.0
                # ~ init some parameters in parallel
                para_results = [] 
                # ~ parallel run spherical average
                for para in NX_list:
                    para_results.append(pool.apply_async(MC_LDA_parallel_kernel,
                                                         (Mx, My, Mz, NX, rhop, xc_code, ni, para,ngrid, weight, 
                                                          factor, relativity, verbose, LIBXCT_factor)))
                    
                # ~ finisht the parallel part.
                pool.close()
                pool.join()
                # ~ get the final result
                for result_para in para_results:
                    result = result_para.get()
                    exc_tot += result[0]
                    Bxc_tot += result[5]
                    vaa_factor+= result[1]
                    vbb_factor+= result[2]
                    vab_r_factor+= result[3]
                    vab_i_factor+= result[4]
                Bxc_tot = Bxc_tot
                # excsum[idm]+= exc_tot.sum()
                # ^ Numerical instability
                if MSL_factor:
                    s_norm = numpy.sqrt(Mx*Mx + My*My + Mz*Mz)
                    s_norm_half = 0.5*s_norm
                    idx_instable = s_norm_half>=MSL_factor*rhop
                    
                    idx_u_polar = rhop[idx_instable] - s_norm_half[idx_instable] <= LIBXCT_factor
                    idx_d_polar = rhop[idx_instable] + s_norm_half[idx_instable] <= LIBXCT_factor
                    N_instable = idx_instable.sum()
                    NX_instable = numpy.zeros((N_instable,3))
                    NX_instable[:,0] = Mx[idx_instable]/s_norm[idx_instable]
                    NX_instable[:,1] = My[idx_instable]/s_norm[idx_instable]
                    NX_instable[:,2] = Mz[idx_instable]/s_norm[idx_instable]
                    exc_lc, vxc_lc = ni.eval_xc(xc_code, (rhop[idx_instable] + s_norm_half[idx_instable]
                                                        , rhop[idx_instable] - s_norm_half[idx_instable])
                                                , spin=1,
                                            relativity=relativity, deriv=1,
                                            verbose=verbose)[:2]
                    vrho = vxc_lc[0]
                    vrho[idx_d_polar,0] = 0.0
                    vrho[idx_u_polar,1] = 0.0
                    
                    den = 2*rhop[idx_instable]*weight[idx_instable]
                    exc_tot[idx_instable] = exc_lc*den
                    
                    vmm = (vrho[:,0]-vrho[:,1])*0.5
                    vpp = (vrho[:,0]+vrho[:,1])*0.5
                    vaa_factor[idx_instable] = vpp + vmm*NX_instable[:,2]
                    vbb_factor[idx_instable] = vpp - vmm*NX_instable[:,2]
                    vab_r_factor[idx_instable] = vmm*NX_instable[:,0]
                    vab_i_factor[idx_instable] = vmm*NX_instable[:,1]
                    
                    Bxc_tot[0,idx_instable] = vmm*NX_instable[:,0]
                    Bxc_tot[1,idx_instable] = vmm*NX_instable[:,1]
                    Bxc_tot[2,idx_instable] = vmm*NX_instable[:,2]
                excsum[idm]+= exc_tot.sum()
                # ^ End of numerical instability
                numpy.save('Bxc_tot_part'+str(ipart),Bxc_tot)
                # ! Note that for Vmat, factor can be product here!
                # !     different from energy.
                # contraction
                aow = numint._scale_ao(ao, .5*weight*vaa_factor, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,1], out=aow)
                aow = numint._scale_ao(ao, .5*weight*vab_r_factor, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                vmat[2,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                aow = numint._scale_ao(ao, .5*weight*vab_i_factor, out=aow)
                vmat[1,idm] -= numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)*1.0j
                vmat[2,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)*1.0j
                aow = numint._scale_ao(ao, .5*weight*vbb_factor, out=aow)
                vmat[3,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                
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
        for ao, mask, weight, coords \
                  in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            ipart += 1
            for idm in range(nset):
                exc_tot = 0.0
                # calculate densities and M vector
                rho_aa = make_rhoa(idm, ao, mask, 'GGA')
                rho_bb = make_rhob(idm, ao, mask, 'GGA')
                Mx = make_rhoMx(idm, ao, mask, 'GGA')
                My = make_rhoMy(idm, ao, mask, 'GGA')
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
                s = numpy.zeros((4, ngrid))
                wvaahss_factor = numpy.zeros((4,ngrid))
                wvbbhss_factor = numpy.zeros((4,ngrid))
                wvabhss_i_factor = numpy.zeros((4,ngrid))
                wvabhss_r_factor = numpy.zeros((4,ngrid))
                # ~ init some parameters in parallel
                para_results = [] 
                # ~ parallel run spherical average
                for para in NX_list:
                    para_results.append(pool.apply_async(MC_GGA_parallel_kernel,
                                                         (Mx, My, Mz, NX, rhop, xc_code, ni, para, ngrid, weight,
                                                          factor, relativity, verbose, LIBXCT_factor)))
                # ~ finisht the parallel part.
                pool.close()
                pool.join()
                # ~ get the final result
                for result_para in para_results:
                    result = result_para.get()
                    exc_tot += result[0]
                    wvaahss_factor+= result[1]
                    wvbbhss_factor+= result[2]
                    wvabhss_r_factor+= result[3]
                    wvabhss_i_factor+= result[4]
                    
                # import pdb
                # pdb.set_trace()
                # excsum[idm]+= exc_tot.sum()
                # ! Note that for Vmat, factor can be product here!
                # !     different from energy.
                if MSL_factor:
                    # ^ Numerical instability
                    s_norm = numpy.sqrt(Mx[0]*Mx[0] + My[0]*My[0] + Mz[0]*Mz[0])
                    idx_instable = s_norm*0.5>=MSL_factor*rhop[0]
                    N_instable = idx_instable.sum()
                    NX_instable = numpy.zeros((N_instable,3))
                    s_stability = numpy.zeros((4,N_instable))
                    NX_instable[:,0] = Mx[0,idx_instable]/s_norm[idx_instable]
                    NX_instable[:,1] = My[0,idx_instable]/s_norm[idx_instable]
                    NX_instable[:,2] = Mz[0,idx_instable]/s_norm[idx_instable]
                    s_stability[0] = s_norm[idx_instable]
                    s_stability[1:] = Mx[1:,idx_instable]*NX_instable[:,0] \
                        + My[1:,idx_instable]*NX_instable[:,1] \
                        + Mz[1:,idx_instable]*NX_instable[:,2]
                    s_stability *=0.5
                    
                    rho_ahss = rhop[:,idx_instable] + s_stability
                    rho_bhss = rhop[:,idx_instable] - s_stability
                    
                    idx_u_polar = rho_bhss[0] <= LIBXCT_factor
                    idx_d_polar = rho_ahss[0] <= LIBXCT_factor
                    
                    exc_lc, vxc_lc = ni.eval_xc(xc_code, (rho_ahss, rho_bhss), spin=1,
                                            relativity=relativity, deriv=1,
                                            verbose=verbose)[:2]
                    
                    den = 2*rhop[0,idx_instable]*weight[idx_instable]
                    exc_tot[idx_instable] = exc_lc*den
                    
                    wva, wvb = numint._uks_gga_wv0((rho_ahss,rho_bhss), vxc_lc, weight[idx_instable])
                    wva[:,idx_d_polar] = 0.0
                    wvb[:,idx_u_polar] = 0.0
                    wvm = (wva - wvb)*0.5
                    wvp = (wva + wvb)*0.5
                    
                    wvaahss_factor[:,idx_instable] = wvp + wvm*NX_instable[:,2]
                    wvbbhss_factor[:,idx_instable] = wvp - wvm*NX_instable[:,2]
                    wvabhss_r_factor[:,idx_instable] = wvm*NX_instable[:,0]
                    wvabhss_i_factor[:,idx_instable] = wvm*NX_instable[:,1]
                    # ^ End of numerical instability
                wvaahss_factor = wvaahss_factor*weight
                wvbbhss_factor = wvbbhss_factor*weight
                wvabhss_r_factor = wvabhss_r_factor*weight
                wvabhss_i_factor = wvabhss_i_factor*weight
                excsum[idm]+= exc_tot.sum()
                # Nelectron
                den = rho_aa[0]*weight
                nelec[0,idm] += den.sum()
                den = rho_bb[0]*weight
                nelec[1,idm] += den.sum()
                # contraction
                aow = numint._scale_ao(ao, wvaahss_factor, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # real part
                aow = numint._scale_ao(ao, wvabhss_r_factor, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # imag part
                aow = numint._scale_ao(ao, wvabhss_i_factor, out=aow)
                vmat[1,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
                aow = numint._scale_ao(ao, wvbbhss_factor, out=aow)
                vmat[3,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

                rho_ahss = rho_bhss = exc = vxc = wva = wvb = None
                rho_ahss = rho_bhss = exc = vxc = wva = wvb = aow = None
                rho_aa = rho_bb = Mx = My = Mz = s = rhop = None
    elif xctype == 'MGGA':
        raise NotImplementedError("There is no meta-GGA fuctionals for AGEC")
    for i in range(nset):
        # Note that all the gradients are calculated by conjugate, thus terms without
        # gradients are producted by 0.5!
        vmat[0,i] = vmat[0,i] + vmat[0,i].conj().T
        vmat[1,i] = vmat[1,i] + vmat[2,i].conj().T
        vmat[2,i] = vmat[1,i].conj().T
        vmat[3,i] = vmat[3,i] + vmat[3,i].conj().T

    vmat = vmat[:,0]
    nelec = nelec.reshape(2)
    excsum = excsum[0]

    end = time.process_time()
    print('Running time for uksm: %s Seconds'%(end-start))
    return nelec, excsum, vmat


def MC_GGA_intbypart_parallel_kernel(Mx, My, Mz, NX, rhop, xc_code, ni, index,  ngrid, factor, 
                                    weight, relativity=0, verbose=None, LIBXCT_factor=None):
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
        exc, exc_cor, vxcn, vxcs = ni.eval_xc_new_MC(xc_code, (rho_ahss, rho_bhss), spin=1,
                        relativity=relativity, deriv=2,
                        verbose=verbose, ibp = True, LIBXCT_factor = LIBXCT_factor)
        # den = rho_ahss * weight
        # import pdb
        # pdb.set_trace()
        # ! Note that for energy, factor must product here!
        exctot += exc*factor[idrct]*rhop[0]*2*weight
        # den = rho_bhss * weight
        # exctot += numpy.dot(den, exc)*factor[idrct]
        exctot += weight*exc_cor*factor[idrct]

        # numpy.savetxt('Bxc_d'+str(idrct)+'_part'+str(ipart)+'.txt',Bxc)
        Bxc[0] += vxcs*NX[idrct,0]*factor[idrct]
        Bxc[1] += vxcs*NX[idrct,1]*factor[idrct]
        Bxc[2] += vxcs*NX[idrct,2]*factor[idrct]
        
        vaa += (vxcn + vxcs*NX[idrct,2])*factor[idrct]
        vbb += (vxcn - vxcs*NX[idrct,2])*factor[idrct]
        vab_r += vxcs*NX[idrct,0]*factor[idrct]
        vab_i += vxcs*NX[idrct,1]*factor[idrct]
        
    return exctot, vaa, vbb, vab_r, vab_i, Bxc


def nr_mc_parallel_ibp(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
           max_memory=2000, Ndirect=1454, LIBXCT_factor=1.0E-10, 
           MSL_factor = 0.999, verbose=None, ncpu = None):
    '''Calculate UKS XC functional with Multi-directions framework
    and potential matrix on given meshgrids for a set of density matrices.
    This subroutine uses Tri-directions for non-collinear spin systems.

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
    start = time.process_time()
    xctype = ni._xc_type(xc_code)
    import math
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    nsbatch = math.ceil(Ndirect/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, Ndirect-nsbatch, nsbatch)]
    if NX_list[-1][-1] < Ndirect:
        NX_list.append((NX_list[-1][-1], Ndirect))
    NX,factor = ni.Spoints.make_sph_samples(Ndirect)
    # ~ init some parameters in parallel
    pool = multiprocessing.Pool()
    # ~ parallel run spherical average

    if xctype == 'NLC':
        NotImplementedError

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmaa, dmab, dmba, dmbb = dms
    #print(dmaa, dmab, dmba, dmbb)
    nao = dmaa.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator_ibp(mol, dmaa.real, hermi, True)[:2]
    make_rhob       = ni._gen_rho_evaluator_ibp(mol, dmbb.real, hermi, True)[0]
    make_rhoMx      = ni._gen_rho_evaluator_ibp(mol, (dmba+dmab).real, hermi, True)[0]
    make_rhoMy      = ni._gen_rho_evaluator_ibp(mol, (-dmba*1.0j+dmab*1.0j).real, hermi, True)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    # vmat will save in the order(alpha_alpha,alpha_beta,beta_alpha,beta_beta).
    vmat = numpy.zeros((4,nset,nao,nao), dtype=numpy.complex128)
    aow = None

    if xctype == 'LDA':
        raise ValueError("Please check the input file. Should not reach here.")
    elif xctype == 'GGA':
        ao_deriv = 2
        ipart = 0
        numpy.save('coords',grids.coords)
        numpy.save('weights',grids.weights)
        N_instable_tot = 0
        for ao, mask, weight, coords \
                  in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            ipart += 1
            for idm in range(nset):
                # calculate densities and M vector
                rho_aa = make_rhoa(idm, ao, mask, 'GGA')
                rho_bb = make_rhob(idm, ao, mask, 'GGA')
                Mx = make_rhoMx(idm, ao, mask, 'GGA')
                My = make_rhoMy(idm, ao, mask, 'GGA')
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
                # calculate functional derivatives
                Bxc_tot = numpy.zeros((3,ngrid))
                exc_tot = 0.0
                # ~ init some parameters in parallel
                para_results = [] 
                # ~ parallel run spherical average
                for para in NX_list:
                    para_results.append(pool.apply_async(MC_GGA_intbypart_parallel_kernel,
                                                         (Mx, My, Mz, NX, rhop, xc_code, ni, para, ngrid, factor,
                                                         weight, relativity, verbose, LIBXCT_factor)))
                                                         
                # ~ finisht the parallel part.
                pool.close()
                pool.join()
                # ~ get the final result
                for result_para in para_results:
                    result = result_para.get()
                    exc_tot += result[0]
                    Bxc_tot += result[5]
                    vaa_factor+= result[1]
                    vbb_factor+= result[2]
                    vabr_factor+= result[3]
                    vabi_factor+= result[4]
                if MSL_factor:
                    # ^ Numerical instability
                    s_norm = numpy.sqrt(Mx[0]*Mx[0] + My[0]*My[0] + Mz[0]*Mz[0])
                    idx_instable = s_norm*0.5>=MSL_factor*rhop[0]
                    N_instable = idx_instable.sum()
                    NX_instable = numpy.zeros((N_instable,3))
                    N_instable_tot += N_instable
                    s_stability = numpy.zeros((10,N_instable))
                    NX_instable[:,0] = Mx[0,idx_instable]/s_norm[idx_instable]
                    NX_instable[:,1] = My[0,idx_instable]/s_norm[idx_instable]
                    NX_instable[:,2] = Mz[0,idx_instable]/s_norm[idx_instable]
                    s_stability[0] = s_norm[idx_instable]
                    s_stability[1:] = Mx[1:,idx_instable]*NX_instable[:,0] \
                        + My[1:,idx_instable]*NX_instable[:,1] \
                        + Mz[1:,idx_instable]*NX_instable[:,2]
                    s_stability *=0.5
                    
                    rho_ahss = rhop[:,idx_instable] + s_stability
                    rho_bhss = rhop[:,idx_instable] - s_stability
                    
                    idx_u_polar = rho_bhss[0] <= LIBXCT_factor
                    idx_d_polar = rho_ahss[0] <= LIBXCT_factor
                    
                    exc_lc, vxc, fxc = ni.eval_xc(xc_code, (rho_ahss[:4], rho_bhss[:4]), spin=1,
                                            relativity=relativity, deriv=2,
                                            verbose=verbose)[:3]
                    
                    den = 2*rhop[0,idx_instable]*weight[idx_instable]
                    exc_tot[idx_instable] = exc_lc*den
                    
                    wva, wvb, wvrho_nrho, wvnrho_nrho \
                        = ni.uks_gga_wv0_intbypart_noweight((rho_ahss,rho_bhss), vxc, fxc)
                    wva[:,idx_d_polar] = 0.0
                    wvb[:,idx_u_polar] = 0.0
                    wvrho_nrho[:3, idx_d_polar] = 0.0
                    wvrho_nrho[3:9, idx_d_polar] = 0.0
                    wvrho_nrho[3:9, idx_u_polar] = 0.0
                    wvrho_nrho[9:, idx_u_polar] = 0.0
                    wvnrho_nrho[:6, idx_d_polar] = 0.0
                    wvnrho_nrho[6:15, idx_d_polar] = 0.0
                    wvnrho_nrho[6:15, idx_u_polar] = 0.0
                    wvnrho_nrho[15:, idx_u_polar] = 0.0
                    
                    wvp = 0.5*(wva[0]+wvb[0])
                    Bxc = cal_Bxc((rhop[:,idx_instable]*2, Mx[:,idx_instable], My[:,idx_instable], Mz[:,idx_instable]), 
                                NX_instable.transpose(1,0),
                        (wva, wvb, wvrho_nrho, wvnrho_nrho))
                    Wxc = cal_Wxc((rhop[:,idx_instable]*2, Mx[:,idx_instable], My[:,idx_instable], Mz[:,idx_instable]), 
                                NX_instable.transpose(1,0),
                        (wva, wvb, wvrho_nrho, wvnrho_nrho))
                    Bxc_tot[:,idx_instable] = Bxc
                    vaa_factor[idx_instable] = wvp + Bxc*NX_instable[:,2] + Wxc 
                    vbb_factor[idx_instable] = wvp - Bxc*NX_instable[:,2] + Wxc 
                    vabr_factor[idx_instable] = Bxc*NX_instable[:,0]
                    vabi_factor[idx_instable] = Bxc*NX_instable[:,1]
                    # ^ Numerical instability 
                excsum[idm]+= exc_tot.sum()
                # numpy.save('Bxc_d_binary_part'+str(ipart),Bxcp)
                numpy.save('Bxc_tot_part'+str(ipart),Bxc_tot)
                # ! Note that for Vmat, factor can be product here!
                # !     different from energy.
                
                # Nelectron
                den = rho_aa[0]*weight
                nelec[0,idm] += den.sum()
                den = rho_bb[0]*weight
                nelec[1,idm] += den.sum()
                # contraction
                aow = numint._scale_ao(ao[0], vaa_factor *weight, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # real part
                aow = numint._scale_ao(ao[0], vabr_factor *weight, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # imag part
                aow = numint._scale_ao(ao[0], vabi_factor *weight, out=aow)
                vmat[1,idm] -= numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                vmat[2,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)*1.0j
                #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
                aow = numint._scale_ao(ao[0], vbb_factor *weight, out=aow)
                vmat[3,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

                rho_ahss = rho_bhss = exc = vxc = wva = wvb = None

                rho_ahss = rho_bhss = exc = vxc = wva = wvb = aow = None
                rho_aa = rho_bb = Mx = My = Mz = s = rhop = None
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA method in Multi-directions framework')

    vmat = vmat[:,0]
    nelec = nelec.reshape(2)
    excsum = excsum[0]

    end = time.process_time()
    print('Running time for uksm: %s Seconds'%(end-start))
    return nelec, excsum, vmat

class numint_gksmc(numint.NumInt):
    '''numint_gksmc'''
    def __init__(self):
        numint.NumInt.__init__(self)
        self._M = None
        # * self._rt_vxc_list will be used in real-time part.
        # self._rt_vxc_list = []
        self.Spoints = Spoints.Spoints()
        
    @property
    def M(self):
        return self._M
    @M.setter
    def M(self, M_inp):
        self._M = M_inp
    nr_uks_intbyparts = nr_uks_intbyparts
    nr_uks_lc = nr_uks_lc
    nr_mc_parallel = nr_mc_parallel
    nr_mc_parallel_ibp = nr_mc_parallel_ibp
    nr_uksmc_rt = nr_uksmc_rt
    nr_uksmc_intbyparts_rt = nr_uksmc_intbyparts_rt
    nr_ghf_rho_rt = nr_ghf_rho_rt

    def eval_rho_intbypart(self, mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
        return eval_rho_intbypart(mol, ao, dm, non0tab, xctype, hermi, verbose)
    
    def _gen_rho_evaluator_ibp(self, mol, dms, hermi=0, intbypart=False):
        if getattr(dms, 'mo_coeff', None) is not None:
            #TODO: test whether dm.mo_coeff matching dm
            raise NotImplementedError("")
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
            if intbypart :
                def make_rho(idm, ao, non0tab, xctype):
                    return self.eval_rho_intbypart(mol, ao, dms[idm], non0tab, xctype, hermi=1)
            else:
                def make_rho(idm, ao, non0tab, xctype):
                    return self.eval_rho(mol, ao, dms[idm], non0tab, xctype, hermi=1)
        return make_rho, ndms, nao

    def eval_xc_new_MC(self, xc_code, rho, spin=1, relativity=0, deriv=1, omega=None,
                verbose=None, ibp = False, LIBXCT_factor = None):
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
        if LIBXCT_factor is None:
            LIBXCT_factor = -1
        if xctype == 'LDA':
            idx_u_polar = rho[1] <= LIBXCT_factor
            idx_d_polar = rho[0] <= LIBXCT_factor

        elif xctype == 'GGA':
        
            idx_u_polar = rho[1][0] <= LIBXCT_factor
            idx_d_polar = rho[0][0] <= LIBXCT_factor
            
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
                n_Nn_Ns, s_Nn_Nn, Nn_Ns_Ns, Ns_Ns_Ns, Nn_Nn_Ns = \
                    get_kxc_in_s_n_kernel(rho, v2rhosigma, v2sigma2, 
                            kxc, LIBXCT_factor)[1:] 

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
            # ^ NUMERICASL STABLE
            u_u[idx_d_polar] = 0.0
            u_d[idx_d_polar] = 0.0
            u_d[idx_u_polar] = 0.0
            d_d[idx_u_polar] = 0.0
            wva[:,idx_d_polar] = 0.0
            wvb[:,idx_u_polar] = 0.0
            wvrho_nrho[:3, idx_d_polar] = 0.0
            wvrho_nrho[3:9, idx_d_polar] = 0.0
            wvrho_nrho[3:9, idx_u_polar] = 0.0
            wvrho_nrho[9:, idx_u_polar] = 0.0
            wvnrho_nrho[:6, idx_d_polar] = 0.0
            wvnrho_nrho[6:15, idx_d_polar] = 0.0
            wvnrho_nrho[6:15, idx_u_polar] = 0.0
            wvnrho_nrho[15:, idx_u_polar] = 0.0
            # ^ NUMERICASL STABLE
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
                # ^ NUMERICASL STABLE
                u[idx_d_polar] = 0.0
                d[idx_u_polar] = 0.0
                u_u[idx_d_polar] = 0.0
                u_d[idx_d_polar] = 0.0
                u_d[idx_u_polar] = 0.0
                d_d[idx_u_polar] = 0.0
                # ^ NUMERICASL STABLE
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
                # ^ NUMERICASL STABLE
                wva[:,idx_d_polar] = 0.0
                wvb[:,idx_u_polar] = 0.0
                u_u[idx_d_polar] = 0.0
                u_d[idx_d_polar] = 0.0
                u_d[idx_u_polar] = 0.0
                d_d[idx_u_polar] = 0.0
                wvrho_nrho[:3, idx_d_polar] = 0.0
                wvrho_nrho[3:9, idx_d_polar] = 0.0
                wvrho_nrho[3:9, idx_u_polar] = 0.0
                wvrho_nrho[9:, idx_u_polar] = 0.0
                wvnrho_nrho[:6, idx_d_polar] = 0.0
                wvnrho_nrho[6:15, idx_d_polar] = 0.0
                wvnrho_nrho[6:15, idx_u_polar] = 0.0
                wvnrho_nrho[15:, idx_u_polar] = 0.0
                # ^ NUMERICASL STABLE
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
 