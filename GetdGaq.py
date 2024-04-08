#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:10:47 2024

@author: bwb16179
"""

import os, glob, orca_parser

prot_aq_outs = glob.glob('Complete_DFT_Outs/Aq_Z=1/*.out')
deprot_aq_outs = glob.glob('Complete_DFT_Outs/Aq_Z=0/*.out')
prot_gas_outs = glob.glob('Complete_DFT_Outs/GasZ=1/*.out')
deprot_gas_outs = glob.glob('Complete_DFT_Outs/GasZ=0/*.out')

Energies = {}

# Loop through each molecule
for i in range(1, 12):  # 1 through 11
    Energies[i] = {}
    for form in ('prot', 'deprot'):
        if form == 'prot':
            charge = 1
        else:
            charge = 0
        for phase in ('Gas', 'Aq'):
            # Constructing filename
            filename = f"Complete_DFT_Outs/{phase}_Z={charge}/{i}_{form}.out"
            # Assuming files are in the current directory, adjust path as needed
            if os.path.exists(filename):
                op = orca_parser.ORCAParse(filename)
                op.parse_free_energy()
                # Store the Gibbs energy
                Energies[i][f"{form}_{phase}"] = op.Gibbs
            else:
                print(f"File {filename} not found.")

    # Calculate solvation energy: aq - gas for both prot and deprot
    for form in ('prot', 'deprot'):
        try:
            solvation_energy = Energies[i][f"{form}_Aq"] - Energies[i][f"{form}_Gas"]
            # Store the solvation energy
            Energies[i][f"{form}_solvation_energy"] = solvation_energy
        except KeyError as e:
            print(f"Could not calculate solvation energy for {i} {form}: {e}")
