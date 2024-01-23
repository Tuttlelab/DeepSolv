# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 22:46:13 2024

@author: Alex
"""
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances, mean_squared_error

from DeepSolv import *
import orca_parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

protons = {1: "H",
           6: "C",
           7: "N",
           8: "O",
           9: "F",
           15:"P",
           17:"Cl",
           77:"Ir"
           }
inv_protons = {v: k for k, v in protons.items()}

if __name__ == "__main__":
    x = pKa()
    x.load_models("TrainDNN/model/", "best.pt")
    #x.load_models("TrainDNN/model/", "best_L1.pt")
    print(x.Gmodels)
    assert "prot_aq" in x.Gmodels
    x.load_yates()
    x.use_yates_structures()
    
    idx = 1
    state = "deprot_aq"
    #mol = x.input_structures[1]["deprot_aq"].copy()
    
    natoms, Eh, atomic_numbers, coords, Forces = orca_parser.parse_engrad("engrads/CO2.engrad")
    
    _, Eh2, _, _, Forces2 = orca_parser.parse_engrad("engrads/CO2_1.engrad")
    dE = (Eh2 - Eh)*627.5
    dft_calc_F = dE / 0.1
    
    mol = Atoms([protons[x] for x in atomic_numbers], coords)
    x.input_structures[idx]["deprot_aq"] = mol.copy()
    x.input_structures[idx]["deprot_aq"].calc = x.Gmodels[state].SUPERCALC
    
    
    #F = x.get_forces(idx, state, drs = [0.01, 0.001, 0.0001] + [-0.015, -0.0015, -0.00015])
    F = x.get_forces(idx, state, drs = [-0.2])
    print("DNN")
    print((F/F.max()).round(2))
    print("DFT")
    print((Forces/Forces.max()).round(2))
    
    err = (F/F.max()) - (Forces/Forces.max())
    rmse = mean_squared_error(F/F.max(), Forces/Forces.max(), squared=False)
    
    print("RMSE:", rmse)
    Fx = Forces/0.52917724900001
    Fx *= 627.5
    print("RMSE:", mean_squared_error(Fx, F, squared=False), "kcal/mol / A")
    print("RMSE:", mean_squared_error(Fx, -F, squared=False), "kcal/mol / A")
          
    #x.work_folder = "engrads"
    #x.Min(idx, state)

    sys.exit()

    #F = x.get_forces(idx, state, drs=[0.1, 0.05, -0.1])
    
    drs = np.linspace(0, 0.05, 10)
    mol = x.input_structures[1]["deprot_aq"].copy()
    Y = []
    for dr in drs:
        moldr = mol.copy()
        #moldr.calc = x.Gmodels[state].SUPERCALC
        moldr.positions[0,0] += dr
        #Y.append(moldr.get_potential_energy())
        F = x.get_forces(idx, state, drs=drs)
        Y.append(F[0,0])
    Y = np.array(Y)
    print(Y)
    plt.plot(drs, Y-Y[0])
    plt.plot(drs, np.gradient(Y))

    drs = np.linspace(0, -0.05, 10)
    mol = x.input_structures[1]["deprot_aq"].copy()
    Y = []
    for dr in drs:
        moldr = mol.copy()
        #moldr.calc = x.Gmodels[state].SUPERCALC
        moldr.positions[0,1] += dr
        #Y.append(moldr.get_potential_energy())
        F = x.get_forces(idx, state, drs=drs)
        Y.append(F[0,0])
        
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
# 
# =============================================================================
