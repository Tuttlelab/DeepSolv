#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:03:01 2024

@author: bwb16179
"""

from xtb.ase.calcualtor import XTB
from ase.io import read, write
from ase.optimize import BFGS
import glob, sys

xyzs = glob.glob("SOURCE_XYZS/*.xyz")

for xyz in xyzs:
    asemol = read(xyz)
    asemol.calc = XTB(method="GFN2-xTB", accuracy=2.0, cache_api=False)
    sys.exit()
    opt = BFGS(asemol)
    opt.run(fmax=0.05)
    write(xyz.split("/")[-1].replace('.xyz', '_xtb.xyz'), images=asemol)
    