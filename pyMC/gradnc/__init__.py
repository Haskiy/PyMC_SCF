#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-11-12 01:18:16
LastEditTime: 2022-11-12 01:21:49
LastEditors: Li Hao
Description: 

FilePath: /pyMC/gradnc/__init__.py
Motto: A + B = C!
'''

from pyscf.grad import rhf
from .ghf import Gradients as GHF # add by lihao

grad_nuc = rhf.grad_nuc

try:
    from .gksnc import Gradients as GKS # add by lihao
except (ImportError, OSError):
    pass