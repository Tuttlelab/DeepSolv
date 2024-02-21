#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:57:14 2024

@author: bwb16179
"""

import os
import glob
import shutil


xyzs = glob.glob("*.xyz")
cwd = os.path.abspath(".")

for xyz in xyzs:
    folder = xyz.replace(".xyz", "")
    if not os.path.exists(folder):
        os.mkdir(folder)
    shutil.move(xyz, folder)
    os.chdir(folder)
    if "+" in xyz:
        os.system(f"crest {xyz} --gfn2 --chrg 1 -T 4")
    else:
        os.system(f"crest {xyz} --gfn2 -T 4")
    os.chdir(cwd)
    