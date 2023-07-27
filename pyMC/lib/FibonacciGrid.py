#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-04-08 10:20:10
LastEditTime: 2023-02-25 08:17:14
LastEditors: Li Hao
Description: Fibonacci sample points and weights on the sphere surface.

FilePath: /pyMC/lib/FibonacciGrid.py
Motto: A + B = C!
'''
import numpy as np

# Reference: https://blog.csdn.net/qq_41035283/article/details/124689331
def MakeFibonacciGrid(N,r=1.0):
    N=int(N)
    assert(N > 0)
    # points.shape: 4 -> x,y,z,weights of grids.
    points = np.zeros((4,N))
    phi = (np.sqrt(5)- 1)/ 2
    n = np.arange(0, N)
    z = (2*n + 1)/N - 1
    
    points[2] += z
    points[0] = np.sqrt(1-z**2)* np.cos(2*np.pi * (n+1)*phi)
    points[1] = np.sqrt(1-z**2)* np.sin(2*np.pi * (n+1)*phi)
    points[3] += 1/N
    return points.transpose(1,0)

    
    


    