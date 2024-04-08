#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:24:27 2024

@author: bwb16179
"""

import orca_parser
import glob
import pandas
import numpy
import ase
import os
import json
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from ase import Atoms
import torch
from _DNN import *
from ase.io import read

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def calculate_pka(G_deprot, G_prot):
    ln_log_conversion = 2.303 # ln(10) / log(10) = N //// ln(10) = log(10)*N //// 2.303 = 1*N //// N = 2.303
    R = 0.0019872036 # Gas constant in kcal / K / mol
    T = 298.15 # Temperature in K
    guess_pka = ((G_deprot - (G_prot - -4.39 - -264.61)) / (ln_log_conversion * R * T))
    #guess_pka = (((G_deprot + (dG_solv_H + G_H)) - G_prot) / ln_log_conversion * R * T)
    return guess_pka

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yates_mols = {}
yates_unopt_pKa = pandas.DataFrame()
dft_folder = 'Avogadro_outputs/'
checkpoint = 'best_L1.pt'
pKas = pandas.read_csv(os.path.join(os.path.dirname(__file__), "DFT_Data_pKa.csv"), index_col=0)

Gmodels = {}
for fullpath in find_all(checkpoint, "TrainDNN/models/uncleaned/"):
    chk = fullpath.replace("TrainDNN/models/uncleaned/", "")
    subdir = os.path.dirname(chk)
    Dir = os.path.join("TrainDNN/models/uncleaned/", subdir)
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
    print(f"{key} Checkpoint:  {fullpath} loaded successfully")

for mol_index in [1,2,3,4,5,6,7,8,9,10,11]:
    yates_mols[mol_index] = {}
    yates_mols[mol_index]["deprot_aq"]  = {"ase": read(os.path.join(dft_folder, f"{mol_index}_deprot.xyz"))}
    yates_mols[mol_index]["prot_aq"]    = {"ase": read(os.path.join(dft_folder, f"{mol_index}_prot.xyz"))}
    
    for state in yates_mols[mol_index]:
        yates_mols[mol_index][state]["ase"].calc = Gmodels[state].SUPERCALC

    deprot_aq_G = yates_mols[mol_index]['deprot_aq']['ase'].get_potential_energy()*23.06035
    prot_aq_G = yates_mols[mol_index]['prot_aq']['ase'].get_potential_energy()*23.06035
    guess_pKa = calculate_pka(deprot_aq_G, prot_aq_G)
    yates_unopt_pKa.at[mol_index, "DFT_unopt_pKa_pred"] = round(guess_pKa,2)
    yates_unopt_pKa.at[mol_index, "Yates_pKa_lit"] = pKas.at[mol_index, "Yates"]


print(yates_unopt_pKa)
print("RMSE:  ", round(root_mean_squared_error(yates_unopt_pKa['Yates_pKa_lit'].dropna(), yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna()),2))
print("MAE:  ", round(mean_absolute_error(yates_unopt_pKa['Yates_pKa_lit'].dropna(), yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna()),2))
