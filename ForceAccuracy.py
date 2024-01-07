# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 22:46:13 2024

@author: Alex
"""
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from DeepSolv import *
import orca_parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

if __name__ == "__main__":
    x = pKa()
    #x.load_models("TrainDNN/model/", "best.pt"); work_folder = "Calculations/MSE"
    x.load_models("TrainDNN/model/", "best_L1.pt"); work_folder = "Calculations/L1"
    os.makedirs(work_folder, exist_ok=True)
    print(x.Gmodels)
    assert "prot_aq" in x.Gmodels
    x.load_yates()
    x.use_yates_structures()
    
    idx = 1
    state = "deprot_aq"
    
    F = np.ones((len(x.input_structures[1]["deprot_aq"]), 3))
    
    #F = x.get_forces(idx, state, drs=[0.1, 0.05, -0.1])
    
    drs = np.linspace(0, 0.05, 100)
    mol = x.input_structures[1]["deprot_aq"].copy()
    Y = []
    for dr in drs:
        moldr = mol.copy()
        moldr.calc = x.Gmodels[state].SUPERCALC
        moldr.positions[0,0] += dr
        Y.append(moldr.get_potential_energy())
    Y = np.array(Y)
    plt.plot(drs, Y-Y[0])
    plt.plot(drs, np.gradient(Y))
    plt.xlabel("dr")
    plt.xlabel("dE")
# =============================================================================
#     for drs in [[0.1], [-0.1], [0.1, -0.1]]:
#         F = x.get_forces(idx, state, drs=drs)
#         print((F/F.max()).round(2))
#         ax = sns.heatmap(F/F.max(), linewidth=0.0005)
#         plt.title(str(drs))
#         plt.show()
# =============================================================================
