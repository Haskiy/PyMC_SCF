#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-06-03 10:56:51
LastEditTime: 2022-03-25 11:52:32
LastEditors: Pu Zhichen
Description: 
    Principle directions for Multi-directions.
FilePath: \pyMC\lib\priciple_direction.py

 May the force be with you!
'''

from pyMC.lib import Pri_dirt
import numpy

# NX = {
#     50 : Pri_dirt.more_data.N50,
#     100 : Pri_dirt.more_data.N100,
#     500 : Pri_dirt.more_data.N500,
#     1000 : Pri_dirt.more_data.N1000,
#     1800 : Pri_dirt.more_data.N1800,
#     4999 : Pri_dirt.more_data.N4999,
#     5000 : Pri_dirt.more_data.N5000,
#     10000 : Pri_dirt.more_data.N10000,
#     12800 : Pri_dirt.more_data.N12800,
#     50000 : Pri_dirt.more_data.N50000,
#     100000 : Pri_dirt.more_data.N100000,
#     500000 : Pri_dirt.more_data.N500000,
#     1000000 : Pri_dirt.more_data.N1000000,
# }

# WX = {
#     1800 : Pri_dirt.more_data.W1800,
#     4999 : Pri_dirt.more_data.W4999,
#     12800 : Pri_dirt.more_data.W12800
# }

NX = {
    # ~ Lebedev
    14    : Pri_dirt.more_data.N14   ,
    38    : Pri_dirt.more_data.N38   ,
    110   : Pri_dirt.more_data.N110  ,
    266   : Pri_dirt.more_data.N266  ,
    434   : Pri_dirt.more_data.N434  ,
    770   : Pri_dirt.more_data.N770  ,
    1454  : Pri_dirt.more_data.N1454 ,
    3074  : Pri_dirt.more_data.N3074 ,
    5810  : Pri_dirt.more_data.N5810 ,
    
    # ~ Legendre
    20    : Pri_dirt.more_data.N20   ,
    50    : Pri_dirt.more_data.N50   ,
    100   : Pri_dirt.more_data.N100  ,
    200   : Pri_dirt.more_data.N200  ,
    400   : Pri_dirt.more_data.N400  ,
    800   : Pri_dirt.more_data.N800  ,
    1800  : Pri_dirt.more_data.N1800 ,
    3200  : Pri_dirt.more_data.N3200 ,
    5000  : Pri_dirt.more_data.N5000 ,
    12800 : Pri_dirt.more_data.N12800,
    20000 : Pri_dirt.more_data.N20000,
    45000 : Pri_dirt.more_data.N45000,
    
    # ~ Fibonacci
    10    : Pri_dirt.more_data.N10    ,
    40    : Pri_dirt.more_data.N40    ,
    150   : Pri_dirt.more_data.N150   ,
    300   : Pri_dirt.more_data.N300   ,
    600   : Pri_dirt.more_data.N600   ,
    1200  : Pri_dirt.more_data.N1200  ,
    2400  : Pri_dirt.more_data.N2400  ,
    4800  : Pri_dirt.more_data.N4800  ,
    10000 : Pri_dirt.more_data.N10000 ,
    50000 : Pri_dirt.more_data.N50000 ,
    100000 : Pri_dirt.more_data.N100000,
    
    # ~ Spherical design
    48    : Pri_dirt.more_data.N48   ,
    94    : Pri_dirt.more_data.N94   ,
    278   : Pri_dirt.more_data.N278  ,
    782   : Pri_dirt.more_data.N782  ,
    1542  : Pri_dirt.more_data.N1542 ,
    2280  : Pri_dirt.more_data.N2280 ,
    3162  : Pri_dirt.more_data.N3162 ,
    5998  : Pri_dirt.more_data.N5998 ,
    12092 : Pri_dirt.more_data.N12092,
    25428 : Pri_dirt.more_data.N25428,
    37952 : Pri_dirt.more_data.N37952,
    52978 : Pri_dirt.more_data.N52978
}

WX = {
    # ~ Lebedev
    14    : Pri_dirt.more_data.W14   ,
    38    : Pri_dirt.more_data.W38   ,
    110   : Pri_dirt.more_data.W110  ,
    266   : Pri_dirt.more_data.W266  ,
    434   : Pri_dirt.more_data.W434  ,
    770   : Pri_dirt.more_data.W770  ,
    1454  : Pri_dirt.more_data.W1454 ,
    3074  : Pri_dirt.more_data.W3074 ,
    5810  : Pri_dirt.more_data.W5810 ,
    
    # ~ Legendre
    20    : Pri_dirt.more_data.W20   ,
    50    : Pri_dirt.more_data.W50   ,
    100   : Pri_dirt.more_data.W100  ,
    200   : Pri_dirt.more_data.W200  ,
    400   : Pri_dirt.more_data.W400  ,
    800   : Pri_dirt.more_data.W800  ,
    1800  : Pri_dirt.more_data.W1800 ,
    3200   : Pri_dirt.more_data.W3200  ,
    5000  : Pri_dirt.more_data.W5000 ,
    5810  : Pri_dirt.more_data.W5810 ,
    12800 : Pri_dirt.more_data.W12800,
    20000 : Pri_dirt.more_data.W20000,
    45000 : Pri_dirt.more_data.W45000,
    
    # ~ Fibonacci
    10     : Pri_dirt.more_data.W10    ,
    40     : Pri_dirt.more_data.W40    ,
    150    : Pri_dirt.more_data.W150   ,
    300    : Pri_dirt.more_data.W300   ,
    600    : Pri_dirt.more_data.W600   ,
    1200   : Pri_dirt.more_data.W1200  ,
    2400   : Pri_dirt.more_data.W2400  ,
    4800   : Pri_dirt.more_data.W4800  ,
    10000  : Pri_dirt.more_data.W10000 ,
    50000  : Pri_dirt.more_data.W50000 ,
    100000 : Pri_dirt.more_data.W100000,
    
    # ~ Spherical design
    48    : Pri_dirt.more_data.W48   ,
    94    : Pri_dirt.more_data.W94   ,
    278   : Pri_dirt.more_data.W278  ,
    782   : Pri_dirt.more_data.W782  ,
    1542  : Pri_dirt.more_data.W1542 ,
    2280  : Pri_dirt.more_data.W2280 ,
    3162  : Pri_dirt.more_data.W3162 ,
    5998  : Pri_dirt.more_data.W5998 ,
    12092 : Pri_dirt.more_data.W12092,
    25428 : Pri_dirt.more_data.W25428,
    37952 : Pri_dirt.more_data.W37952,
    52978 : Pri_dirt.more_data.W52978
}