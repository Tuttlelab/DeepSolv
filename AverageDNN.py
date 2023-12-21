# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:06:15 2023

@author: Alex
"""

import torch
import glob
from collections import OrderedDict


models = "TrainDNN/model/Aq_Z=0"
files = glob.glob(f"{models}*/best_L1.pt")

print(files)

WandB = OrderedDict()

for i, file in enumerate(files):
    pt = torch.load(file)
    for key in list(pt.keys()):
        if i == 0:
            WandB[key] = pt[key]
        else:
            WandB[key] += pt[key]

for key in list(pt.keys()):
    pt[key] /= len(files)
print(WandB)

torch.save(WandB, f"{models}_L1.pt")