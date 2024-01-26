# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:35:47 2024

@author: Alex
"""
import h5py
import torch, tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances, mean_squared_error
from ase import Atoms
import pandas, pickle

from DeepSolv import *
import orca_parser


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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

x = pKa()
x.load_models("TrainDNN/models/", "best.pt")
#x.load_models("TrainDNN/model/", "best_L1.pt")
print(x.Gmodels)
assert "prot_aq" in x.Gmodels
x.load_yates()
x.use_yates_structures()

# =============================================================================
# calculator = x.Gmodels["deprot_aq"]
# h5 = h5py.File("BuildDataset/AqZ=0_rmsd=2.h5", 'r')
# DS = h5py.File("BuildDataset/AqZ=0_rmsd=2_cleaned.h5", 'w')
# 
# =============================================================================

calculator = x.Gmodels["prot_aq"]
h5 = h5py.File("BuildDataset/AqZ=1_rmsd=2.h5", 'r')
DS = h5py.File("BuildDataset/AqZ=1_rmsd=2_cleaned.h5", 'w')


ds_mol, ds_E, ds_C, ds_S = [],[],[],[]
        
if os.path.exists("PoisonIvy.pkl"):
    print("Reloading:", "PoisonIvy.pkl")
    data = pickle.load(open("PoisonIvy.pkl", 'rb'))
else:
    data = {}

if os.path.exists("Poison.xyz"):
    os.remove("Poison.xyz")

total = 0
removed = 0
cutoff = 20 # kcal/mol
for i in tqdm.tqdm(h5):
    E = h5[i]["energies"][()]
    C = h5[i]["coordinates"][()]
    S = h5[i]["species"][()]
    
    if i not in data:
        mols = []
        for n in range(E.shape[0]):
            mols.append(Atoms(S.astype("<U2"), C[n]))
        
        #print(i, C.shape, E.shape, len(mols))
        
        # Cant load too many samples on the GPU
        if E.shape[0] > 9000:
            print("Too many for gpu")
            continue
        
        calculator.mol = mols
        calculator.MakeTensors()
        pred = calculator.ProcessTensors(units="Ha", return_all=True)
        
        Mean = pred.mean(axis=0)
        Range = pred.max(axis=0)-pred.min(axis=0)
        data[i] = {"Mean": Mean, "Range": Range}
        
        
        with open("PoisonIvy.pkl", 'wb') as f:
            pickle.dump(data, f)
    else:
        Range = data[i]["Range"]
        Mean = data[i]["Mean"]
        
    err = np.sqrt((Mean-E)**2)*627.5
    indices = []
    if ((Range*627.5) > cutoff).any() or (err > cutoff).any():
        mols = []
        for n in range(E.shape[0]):
            mols.append(Atoms(S.astype("<U2"), C[n]))
        #print()
        #print(f"Range/err > {cutoff} kcal/mol")
        indices = np.where((Range*627.5) > cutoff)[0]
        indices = np.hstack((indices, np.where(err > cutoff)[0]))
        indices = np.unique(indices)
        for index in indices:
            mols[index].write("Poison.xyz", append=True)
            removed += 1
    
    # Remove and add to new DS
    E = np.delete(E, indices, axis=0)
    C = np.delete(C, indices, axis=0)
    ds_mol.append(DS.create_group(i))
    ds_E.append(ds_mol[-1].create_dataset("energies", (E.shape[0],), dtype='float64'))
    ds_E[-1][()] = E        
    ds_C.append(ds_mol[-1].create_dataset("coordinates", C.shape, dtype='float64'))
    ds_C[-1][()] = C    
    ds_S.append(ds_mol[-1].create_dataset("species", data=S))



    total += E.shape[0]

print("Removed:", removed, "/", total, removed/ total)

h5.close()
DS.close()
