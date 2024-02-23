#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:59:30 2024

@author: bwb16179
"""
import os, pandas, json, orca_parser
from _DNN import *
import torch
from ase import Atoms
import ase
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def load_models(folder, checkpoint="best.pt"):
    Gmodels = {}
    for fullpath in find_all(checkpoint, folder):
        chk = fullpath.replace(folder, "")
        subdir = os.path.dirname(chk)
        if 'aq' in subdir or 'AQ' in subdir:
            continue
        Dir = os.path.join(folder, subdir)
        
        if "Z=0" in subdir.upper():
            key = "deprot_"
        elif "Z=1" in subdir.upper():
            key = "prot_"
        if "AQ" in subdir.upper():
            key += "aq"
        elif "GAS" in subdir.upper():
            key += "gas"
        
        if key not in Gmodels:
            self_energy = os.path.join(Dir, "Self_Energies.csv")
            training_config = os.path.join(Dir, "training_config.json")
            # Load traning_config to keep ANI parameters consistent
            with open(training_config, 'r') as jin:
                training_config = json.load(jin)
            SelfE = pandas.read_csv(self_energy, index_col=0)
            species_order = SelfE.index
            Gmodels[key] = IrDNN(SelfE, verbose=False, device=device,
                                        training_config=training_config, next_gen=False)
            Gmodels[key].GenModel(species_order)
        Gmodels[key].load_checkpoint(fullpath, typ="Energy")
        #print(f"{key} Checkpoint:  {fullpath} loaded successfully")
    return Gmodels
        
def load_yates(mol_indices):
    yates_mols = {}
    for mol_index in mol_indices:
        dft_folder_aq_prot = "Complete_DFT_Outs\\Aq_Z=1"
        dft_folder_aq_deprot = "Complete_DFT_Outs\\Aq_Z=0"
        dft_folder_gas_prot = "Complete_DFT_Outs\\GasZ=1"
        dft_folder_gas_deprot = "Complete_DFT_Outs\\GasZ=0"
        
        yates_mols[mol_index] = {}
        yates_mols[mol_index]["deprot_aq"]  = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder_aq_deprot, f"{mol_index}_deprot.out"))}
        yates_mols[mol_index]["prot_aq"]    = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder_aq_prot, f"{mol_index}_prot.out"))}
        yates_mols[mol_index]["deprot_gas"] = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder_gas_deprot, f"{mol_index}_deprot.out"))}
        yates_mols[mol_index]["prot_gas"]   = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder_gas_prot, f"{mol_index}_prot.out"))}
        
        for state in yates_mols[mol_index]:
            yates_mols[mol_index][state]["orca_parse"].parse_coords()
            yates_mols[mol_index][state]["orca_parse"].parse_free_energy()
            yates_mols[mol_index][state]["DFT G"] = yates_mols[mol_index][state]["orca_parse"].Gibbs * 627.5095
            yates_mols[mol_index][state]["ase"] = Atoms(yates_mols[mol_index][state]["orca_parse"].atoms, 
                                                       yates_mols[mol_index][state]["orca_parse"].coords[-1])
            
    return yates_mols

models_folder = "TrainDNN/models/uncleaned/"
models = load_models(models_folder)

mol_indices = [1,2,3,4,5,6,7,8,9,10,11]
yates_mols = load_yates(mol_indices)


carbenes = glob.glob("DFT/*.out")
carbene_mols = {}
for carbene in carbenes:
    if 'gasSP' in carbene:
        continue
    carbene_name = carbene.split("/")[-1].replace(".out", "")
    carbene_mols[carbene_name] = {}
    if '+' in carbene_name:
        state = 'prot_gas'
        mol_index = carbene_name.split('+')[0]
    else:
        state = 'deprot_gas'
        mol_index = carbene_name
        
    asemol = yates_mols[int(mol_index)][state]["ase"]
    
    yates_mols[int(mol_index)][state]["ase"].calc = models[state].SUPERCALC
    
    Gibbs = asemol.get_potential_energy() *23.06035
    carbene_mols[carbene_name]['G_pred'] = Gibbs
    carbene_mols[carbene_name]['DFT'] = yates_mols[int(mol_index)][state]['DFT G']
    carbene_mols[carbene_name]['delta_DFT_value'] = Gibbs - yates_mols[int(mol_index)][state]['DFT G']
    
for state in ['prot_gas', 'deprot_gas']:
    G_preds = []
    G_DFTs = []
    for carbene in carbene_mols:
        if state == 'prot_gas':
            if "+" not in carbene:
                continue
        G_pred = carbene_mols[carbene]['G_pred']
        G_preds.append(G_pred)
        G_DFT = carbene_mols[carbene]['DFT']
        G_DFTs.append(G_DFT)
    rmse = round(root_mean_squared_error(G_DFTs, G_preds), 2)

    print(state, "RMSE :", rmse)
    
print("7 DFT difference:   ", round(carbene_mols['7']['delta_DFT_value'],4))
print("7+ DFT difference:  ", round(carbene_mols['7+']['delta_DFT_value'],4))
print("8 DFT difference:   ", round(carbene_mols['8']['delta_DFT_value'],4))
print("8+ DFT difference:  ", round(carbene_mols['8+']['delta_DFT_value'],4))
