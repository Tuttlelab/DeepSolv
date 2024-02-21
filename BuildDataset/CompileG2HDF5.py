# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:41:38 2023

@author: rkb19187
"""

import os, sys, pandas, pickle, h5py, tqdm
from pony import orm
import scotch_db
import orca_parser
import json, glob
from json import JSONEncoder
from ase import Atoms
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, euclidean_distances
import matplotlib.pyplot as plt
from ase.io import read



if not os.path.exists("CompileG2HDF5_cache.pkl"):
    database = scotch_db.DFT_db()
    try:
        database.start("r")
    except:
        pass
    with orm.db_session:
        CompileG2HDF5 = orm.select((p.G, p.conf_id.coordinates, p.conf_id.species, p.dft_output.Charge, p.dft_output.Solvation) 
                                 for p in scotch_db.DFT.Energy if p.G < 0 and 
                                 p.conf_id.nO < 4  and p.conf_id.nC > 0 and p.conf_id.nIr < 1 and p.dft_output.Functional == "WB97X" and
                                 p.dft_output.Dispersion == "D4" and p.dft_output.def2J == True and p.dft_output.BasisSet == "def2-SVP" and
                                 (p.dft_output.Charge == 0 or p.dft_output.Charge == 1))
        CompileG2HDF5 = list(CompileG2HDF5)
            
        with open("CompileG2HDF5_cache.pkl", 'wb') as pklo:
            pickle.dump(CompileG2HDF5, pklo)
    database.disconnect()
else:
    print("Reloading from: CompileG2HDF5_cache.pkl")
    with open("CompileG2HDF5_cache.pkl", 'rb') as pklo:
        CompileG2HDF5 = pickle.load(pklo)


# Load yates structures to make sure we dont have any of them in the training data
yates = {}
print("Loading Yates structures")
for xyzfile in tqdm.tqdm(glob.glob("../../DeepSolvation/BuildDataset/ValidationDS/DFT_water_outs/*.xyz")):
    name = xyzfile.split("/")[1]
    mol = read(xyzfile)
    yates[name] = dict()
    #yates[name]["species"] = dict(zip(*np.unique(mol.get_chemical_symbols(), return_counts=True)))
    yates[name]["species"] = np.unique(mol.get_chemical_symbols(), return_counts=True)
    yates[name]["get_chemical_symbols"] = mol.get_chemical_symbols()
    yates[name]["coords"] = mol.positions

AllSpecies = np.ndarray(len(CompileG2HDF5), dtype="<U1000")
Solvated   = np.ndarray(len(CompileG2HDF5), dtype=np.bool_)
Charge = np.ndarray(len(CompileG2HDF5), dtype=np.int64)
G = np.ndarray(len(CompileG2HDF5), dtype=np.float64)
Coordinates = []
print("Sorting data into typed arrays")
for i in tqdm.tqdm(range(AllSpecies.shape[0])):
    G[i] = CompileG2HDF5[i][0]
    Coordinates.append(np.frombuffer(CompileG2HDF5[i][1]).reshape(-1,3))
    AllSpecies[i] = CompileG2HDF5[i][2]
    Charge[i] = CompileG2HDF5[i][3]
    if CompileG2HDF5[i][4] == "Gas":
        Solvated[i] = False
    else:
        Solvated[i] = True
Coordinates = np.array(Coordinates, dtype=object)


# Check for overlapping atoms, we dont need these few hard examples in this project.
print("Searching for conformers with unreasonable structures or are part of the Yates set")
if os.path.exists("remove_indices.tmp.npy"):
    remove_indices = np.load("remove_indices.tmp.npy")
else:
    remove_indices = []
    for i in tqdm.tqdm(range(Coordinates.shape[0])):
        d = euclidean_distances(Coordinates[i], Coordinates[i])
        d[np.diag_indices(d.shape[0])] = 100 # The diagonal will be [0 .. 0] since its an atoms distance from itself
        species = np.array(AllSpecies[i].split())
        # check if there are unbonded hydrogen atoms
        if "H" in species:
            Hd = euclidean_distances(Coordinates[i][species == "H"], Coordinates[i][~(species == "H")]).min(axis=1)
        else:
            Hd = np.array([1.0])
        if d.min() < 0.9 or (Hd>1.4).any():
            remove_indices.append(i)
            continue
        
        # Check if the molecule is part of the validation / yates set
        reduced_species = np.unique(species, return_counts=True)
        for carbene in yates.keys():
            # Check if they have the same number of atoms
            if reduced_species[1].sum() != yates[carbene]["species"][1].sum():
                continue
            # Check if they have the same number of types of atoms
            if reduced_species[0].shape[0] != yates[carbene]["species"][0].shape[0]:
                continue
            #Check the elements are the same
            if not (reduced_species[0] == yates[carbene]["species"][0]).all():
                continue
            #Check the number of each element is the same
            if not (reduced_species[1] == yates[carbene]["species"][1]).all():
                continue
            
            # same number of each species (the order is not a reliable comparitor)
            rmsd = orca_parser.calc_rmsd(yates[carbene]["coords"], Coordinates[i]) # minimizes rmsd and returns final measure
            mol = Atoms(species, Coordinates[i])
    # =============================================================================
    #         if rmsd > 1 and rmsd < 2:
    #             mol.write(f"rmsd_{rmsd}.xyz", append=False)
    # =============================================================================
            if rmsd > 2:
                continue
            else:
                remove_indices.append(i)
    remove_indices = np.array(remove_indices)
    np.save("remove_indices.tmp.npy", remove_indices)
        
G = np.delete(G, remove_indices)
Coordinates = np.delete(Coordinates, remove_indices)
AllSpecies = np.delete(AllSpecies, remove_indices)
Charge = np.delete(Charge, remove_indices)
Solvated = np.delete(Solvated, remove_indices)
print(f"Removed {len(remove_indices)} unreasonable structures from the dataset")
unique_species = np.unique(AllSpecies)


for CPCM in [True, False]:
    for Z in [0, 1]:
        rmsd = '_rmsd=2'
        print(f"\nBuilding HDF5 datasets CPCM={CPCM} Z={Z}")
        mol, E, C, S, F = [],[],[],[],[]
        atom_types = np.ndarray((0,), dtype="<U2")
        if CPCM:
            DS = h5py.File(f"AqZ={Z}{rmsd}.h5", 'w')
        else:
            DS = h5py.File(f"GasZ={Z}{rmsd}.h5", 'w')
        
        for species in tqdm.tqdm(unique_species):
            #species = AllSpecies[i]
            #Z = Charge[i]
            groupname = f"{i}-{'_'.join(species.split())}"
        
            same_species = np.where((AllSpecies == species) & (Charge == Z) & (Solvated == CPCM))[0]
            if same_species.shape[0] == 0:
                continue
            
            assert Z in [0,1], f"Charge is not 0/+1 for {species}, this isn't necessarily wrong"
            
            
            species = np.array(species.split(), dtype="<U2")
            species = np.array(species, dtype = h5py.special_dtype(vlen=str) )
            
            if "O" in np.unique(species) and "H" in np.unique(species) and np.unique(species).shape[0] == 2:
                print(np.unique(species), "only water")
                continue
                
            if "Cl" in np.unique(species) and "H" in np.unique(species) and np.unique(species).shape[0] == 2:
                print(np.unique(species), "only ClH")
                continue
                
            energies = G[same_species]
            
            Conformers = np.ndarray((same_species.shape[0], *Coordinates[same_species[0]].shape))
            
            # Need to pull the coordiates one-at-a-time because they are stored in a python list and not a numpy array
            for i,j in enumerate(same_species):
                Conformers[i] = Coordinates[j]
            
    
            mol.append(DS.create_group(groupname))
            E.append(mol[-1].create_dataset("energies", (energies.shape[0],), dtype='float64'))
            E[-1][()] = energies        
            C.append(mol[-1].create_dataset("coordinates", Conformers.shape, dtype='float64'))
            C[-1][()] = Conformers    
            S.append(mol[-1].create_dataset("species", data=species))

        DS.close()

        print("Atom types in dataset(s):", atom_types)
        
