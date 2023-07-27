'''
Author: Li Hao
Date: 2021-10-04 15:31:50
LastEditTime: 2023-03-20 05:44:43
LastEditors: Li Hao
Description: 

FilePath: /pyMC/tools/tools_pbc/puwant.py
Motto: A + B = C!
'''
import numpy as np

def prt_mo_energy(mo_energy, kpts):
    print('# ! Mo_energy analysis')
    print('kpts \n')
    for k in range(kpts.shape[-2]):
        print('{:<6d}'.format(k) + '(' + '{:<10.8f}, {:<10.8f}, {:<10.8f}'.format(kpts[k][0],kpts[k][1] ,kpts[k][2]) + ')')
    print('\n')

    for k in range(mo_energy.shape[-2]):
        print('{:<14d}'.format(k), end = '')
    print('\n')

    if mo_energy.ndim == 2:
        for j in range(mo_energy.shape[-1]):
            for i in range((mo_energy.shape[-2] + 1)):
                if i == (mo_energy.shape[-2]):
                    print('\n')
                if i < (mo_energy.shape[-2]):
                    print('{:<14.8f}'.format(mo_energy[i][j]),end = '')
    
    elif mo_energy.ndim == 3:
        print('alpha_mo_energy \n')
        for j in range(mo_energy.shape[-1]):
            for i in range((mo_energy.shape[-2] + 1)):
                if i == (mo_energy.shape[-2]):
                    print('\n')
                if i < (mo_energy.shape[-2]):
                    print('{:<14.8f}'.format(mo_energy[0][i][j]),end = '')
        print('\n')

        print('belta_mo_energy \n')
        for j in range(mo_energy.shape[-1]):
            for i in range((mo_energy.shape[-2] + 1)):
                if i == (mo_energy.shape[-2]):
                    print('\n')
                if i < (mo_energy.shape[-2]):
                    print('{:<14.8f}'.format(mo_energy[1][i][j]),end = '')