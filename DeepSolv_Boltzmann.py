# -*- coding: utf-8 -*-

from ase import Atoms
from ase.calculators.mixing import SumCalculator, MixedCalculator
from ase.optimize import LBFGS
from ase.build import minimize_rotation_and_translation
from ase.io import read
from sklearn.metrics import mean_squared_error, euclidean_distances, mean_absolute_error, r2_score
import torch
import json
import pandas
import os, itertools, tqdm, pickle, warnings, sys
import numpy as np
from colour import Color
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
import orca_parser 
from _DNN import *

from scipy.special import logsumexp


class MyWarning(DeprecationWarning):
    pass

warnings.simplefilter("once", MyWarning)
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 1000)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

bond_cutoffs = pandas.DataFrame()
bond_cutoffs.at["H", "H"] = 1.1
bond_cutoffs.at["H", "C"] = 1.4
bond_cutoffs.at["H", "N"] = 1.4
bond_cutoffs.at["H", "O"] = 1.5
bond_cutoffs.at["H", "Cl"] = 1.4
bond_cutoffs.at["C", "C"] = 1.7
bond_cutoffs.at["C", "N"] = 1.7
bond_cutoffs.at["C", "O"] = 1.7
bond_cutoffs.at["C", "Cl"] = 1.9
bond_cutoffs.at["N", "N"] = 1.7
bond_cutoffs.at["N", "O"] = 1.7
bond_cutoffs.at["N", "Cl"] = 1.9
bond_cutoffs.at["O", "O"] = 1.7
bond_cutoffs.at["O", "Cl"] = 1.9
bond_cutoffs.at["Cl", "Cl"] = 2.0
for i in bond_cutoffs.index:
    for j in bond_cutoffs.columns:
        bond_cutoffs.at[j,i] = bond_cutoffs.at[i,j]
        
def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

class pKa:
    def load_models(self, folder, checkpoint="best.pt"):
        self.Gmodels = {}
        for fullpath in find_all(checkpoint, folder):
            chk = fullpath.replace(folder, "")
            subdir = os.path.dirname(chk)
            Dir = os.path.join(folder, subdir)
            if "Z=0" in subdir.upper():
                key = "deprot_"
            elif "Z=1" in subdir.upper():
                key = "prot_"
            if "AQ" in subdir.upper():
                key += "aq"
            elif "GAS" in subdir.upper():
                key += "gas"
            
            if key not in self.Gmodels:
                self_energy = os.path.join(Dir, "Self_Energies.csv")
                training_config = os.path.join(Dir, "training_config.json")
                # Load traning_config to keep ANI parameters consistent
                with open(training_config, 'r') as jin:
                    training_config = json.load(jin)
                SelfE = pandas.read_csv(self_energy, index_col=0)
                species_order = SelfE.index
                self.Gmodels[key] = IrDNN(SelfE, verbose=False, device=device,
                                            training_config=training_config, next_gen=False)
                self.Gmodels[key].GenModel(species_order)
            self.Gmodels[key].load_checkpoint(fullpath, typ="Energy")
            #print(f"{key} Checkpoint:  {fullpath} loaded successfully")


    def load_yates(self):
        self.G_H = -4.39
        self.dG_solv_H = -264.61
        self.yates_mols = {}
        self.yates_unopt_pKa = pandas.DataFrame()
        for mol_index in self.mol_indices:
            dft_folder = os.path.join(os.path.dirname(__file__), "DFT")
            
            self.yates_mols[mol_index] = {}
            self.yates_mols[mol_index]["deprot_aq"]  = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder, f"{mol_index}.out"))}
            self.yates_mols[mol_index]["prot_aq"]    = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder, f"{mol_index}+.out"))}
            self.yates_mols[mol_index]["deprot_gas"] = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder, f"{mol_index}_gasSP.out"))}
            self.yates_mols[mol_index]["prot_gas"]   = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder, f"{mol_index}+_gasSP.out"))}
            
            for state in self.yates_mols[mol_index]:
                self.yates_mols[mol_index][state]["orca_parse"].parse_coords()
                self.yates_mols[mol_index][state]["orca_parse"].parse_free_energy()
                self.yates_mols[mol_index][state]["DFT G"] = self.yates_mols[mol_index][state]["orca_parse"].Gibbs * 627.5
                self.yates_mols[mol_index][state]["ase"] = Atoms(self.yates_mols[mol_index][state]["orca_parse"].atoms, 
                                                           self.yates_mols[mol_index][state]["orca_parse"].coords[-1])
                try:
                    self.yates_mols[mol_index][state]["ase"].calc = self.Gmodels[state].SUPERCALC
                except:
                    warnings.warn("Couldnt load model for "+state, MyWarning)
                self.yates_mols[mol_index][state]["Yates pKa"] = self.pKas.at[mol_index, "Yates"]
                
            deprot_aq_G = self.yates_mols[mol_index]['deprot_aq']['ase'].get_potential_energy()*23.06035
            prot_aq_G = self.yates_mols[mol_index]['prot_aq']['ase'].get_potential_energy()*23.06035
            guess_pKa = ((deprot_aq_G) - ((prot_aq_G) - self.G_H - self.dG_solv_H))/(2.303*0.0019872036*298.15)
            self.yates_unopt_pKa.at[mol_index, "DFT_unopt_pKa_pred"] = guess_pKa
            self.yates_unopt_pKa.at[mol_index, "Yates_pKa_lit"] = self.pKas.at[mol_index, "Yates"]
        self.yates_unopt_pKa.at['MSE', "MSE"] = mean_squared_error(self.yates_unopt_pKa['Yates_pKa_lit'].dropna(), self.yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna(), squared=True)
        self.yates_unopt_pKa.at['RMSE', "RMSE"] = mean_squared_error(self.yates_unopt_pKa['Yates_pKa_lit'].dropna(), self.yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna(), squared=False)
        self.yates_unopt_pKa.at['MAE', "MAE"] = mean_absolute_error(self.yates_unopt_pKa['Yates_pKa_lit'].dropna(), self.yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna())
            
            

    def use_yates_structures(self):
        self.input_structures = {}
        for mol_index in self.yates_mols:
            self.input_structures[mol_index] = {}
            for state in self.yates_mols[mol_index]:
                self.input_structures[mol_index][state] = self.yates_mols[mol_index][state]["ase"]
                try:
                    self.input_structures[mol_index][state].calc = self.Gmodels[state].SUPERCALC
                except KeyError:
                    warnings.warn("Failed to set Gmodels["+state+"]", MyWarning)
        
    def get_rmsd(self, idx: int, state: str):
        assert self.yates_mols[idx][state]["ase"].get_chemical_symbols() == self.input_structures[idx][state].get_chemical_symbols(), "Cant measure RMSD if the atom order isnt the same"
        return orca_parser.calc_rmsd(self.yates_mols[idx][state]["ase"].positions,
                                     self.input_structures[idx][state].positions)

    def calc_pKa_full_cycle(self, idx):
        # Get the predictions, also does EnsembleEnergy
        Deprot_aq = self.input_structures[idx]["deprot_aq"].get_potential_energy()*23.06035
        Deprot_gas = self.input_structures[idx]["deprot_gas"].get_potential_energy()*23.06035
        Prot_aq = self.input_structures[idx]["prot_aq"].get_potential_energy()*23.06035
        Prot_gas = self.input_structures[idx]["prot_gas"].get_potential_energy()*23.06035
        # 2 methods of calculating pKa
        # Method 1 (traditional, dft-like thermodynamic cycle)
        dGgas = (Deprot_gas + G_H) - Prot_gas
        dG_solv_A = Deprot_aq - Deprot_gas
        dG_solv_HA = Prot_aq - Prot_gas
        dG_aq = dGgas + dG_solv_A + dG_solv_H  - dG_solv_HA
        pKa_1 = dG_aq/(2.303*0.0019872036*298.15)        
        self.pKas.at[idx, "pKa_pred_1"] = pKa_1
        return pKa_1
    
    def calc_pKa_direct(self, idx):
        # Get the predictions, also does EnsembleEnergy
        Deprot_aq = self.input_structures[idx]["deprot_aq"].get_potential_energy()*23.06035
        Deprot_gas = self.input_structures[idx]["deprot_gas"].get_potential_energy()*23.06035
        Prot_aq = self.input_structures[idx]["prot_aq"].get_potential_energy()*23.06035
        Prot_gas = self.input_structures[idx]["prot_gas"].get_potential_energy()*23.06035
        # 2 methods of calculating pKa
        # Method 1 (traditional, dft-like thermodynamic cycle)
        dGgas = (Deprot_gas + G_H) - Prot_gas
        dG_solv_A = Deprot_aq - Deprot_gas
        dG_solv_HA = Prot_aq - Prot_gas
        dG_aq = dGgas + dG_solv_A + dG_solv_H  - dG_solv_HA
        pKa_1 = dG_aq/(2.303*0.0019872036*298.15)        
        self.pKas.at[idx, "pKa_pred_1"] = pKa_1
        return pKa_1

    def xyzrmsd(self, idx):
        Transparency = 0.8
        xyzout = open(f"rmsd_{idx}.xyz", 'w')
        for state in self.input_structures[idx]:
            minimize_rotation_and_translation(self.yates_mols[idx][state]["ase"],
                                              self.input_structures[idx][state])
            N =  self.input_structures[idx][state].positions.shape[0]*2
            red = Color("red")
            blue = Color("blue")
            xyzout.write(f"{N}\n\n")
            
            for mol, c in zip([self.yates_mols[idx][state]["ase"], self.input_structures[idx][state]], [red, blue]):
                for bead, coord in zip(mol.get_chemical_symbols(), mol.positions):
                    xyzout.write(bead +"\t"+str(coord[0])+"\t"+str(coord[1])+"\t"+str(coord[2])+"\t")
                    xyzout.write(str(c.get_rgb()[0])+"\t"+str(c.get_rgb()[1])+"\t"+str(c.get_rgb()[2]))
                    xyzout.write("\t"+str(Transparency))
                    xyzout.write("\n")

    def find_connections(self, idx, state):
        # Determine the things we are going to vary
        mol = self.input_structures[idx][state].copy()
        if os.path.exists(f"{self.work_folder}/{idx}_{state}_Connections.json"):
            with open(f"{self.work_folder}/{idx}_{state}_Connections.json") as jin:
                Connections = json.load(jin)
        else:
            Connections = {}
            for i in range(len(mol)):
                d = mol.get_distances(i, indices=np.arange(0, len(mol)))
                for j in np.where((d > 0.01) & (d < 1.8))[0].tolist():
                    cutoff = bond_cutoffs.at[mol[i].symbol, mol[j].symbol]
                    if d[j] > cutoff:
                        continue
                    key = f"{i}-{j}"
                    max_dist = (self.radii.at[mol[i].symbol, "vdw_radius"]+self.radii.at[mol[j].symbol, "vdw_radius"]) / 2
                    min_dist = max_dist/1.75
                    Connections[key] = {"Type":"Bond", "a0": i, "a1": j,
                                  "s0": mol[i].symbol, "s1": mol[j].symbol,
                                  "val": float(d[j]), "Min": min_dist, "Max": max_dist,
                                  "fineness": 50}
            for triplet in itertools.permutations(np.arange(len(mol)).tolist(), 3):
                i,j,k = triplet
                if mol.get_distance(i, j) > 1.8 or mol.get_distance(j, k) > 1.8:# or mol.get_distance(j, k) > 3.0:
                    continue
                key=f"{i}-{j}-{k}"
                angle = float(mol.get_angle(*triplet))
                Connections[key] = {"Type":"Angle", "a0": i, "a1": j, "a2": k,
                               "s0": mol[i].symbol, "s1": mol[j].symbol, "s2": mol[k].symbol,
                               "val": angle, "Min": angle-20, "Max": angle+20,
                               "fineness": 100}
            for quadruplet in itertools.permutations(np.arange(len(mol)).tolist(), 4):
                i,j,k,l = quadruplet
                if mol.get_distance(i, j) > 1.8 or mol.get_distance(j, k) > 1.8 or mol.get_distance(k,l) > 1.8:
                    continue
                key=f"{i}-{j}-{k}-{l}"
                dihedral = float(mol.get_dihedral(*quadruplet))
                Connections[key] = {"Type":"Dihedral", "a0": i, "a1": j, "a2": k, "a3": l,
                               "s0": mol[i].symbol, "s1": mol[j].symbol, "s2": mol[k].symbol, "s3": mol[l].symbol,
                               "val": dihedral, "Min": dihedral-40, "Max": dihedral+40,
                               "fineness": 100}
            with open(f"{self.work_folder}/{idx}_{state}_Connections.json", 'w') as jout:
                json.dump(Connections, jout, indent=4)
        return Connections
    
    def get_forces(self, idx: int, state: str, drs: list = None):
        natoms = self.input_structures[idx][state].positions.shape[0]
        E0 = np.zeros((natoms, 3))
        E0[:] = self.input_structures[idx][state].get_potential_energy()* 23.06035
        if drs is None:
            drs = [0.01, 0.001, 0.0001] + [-0.015, -0.0015, -0.00015]
            #drs = [0.01, 0.001]
            #drs = [0.01]
        mol = self.input_structures[idx][state].copy()
        dims = 3
        natoms = mol.positions.shape[0]
        coords = torch.tensor(mol.positions, dtype=torch.float)
        nconfs = len(drs) * dims * natoms
        T = coords.repeat((nconfs, 1))
        T = T.reshape(nconfs, natoms, 3)
        conformer = 0
        for batch, dr in enumerate(drs):
            for atom in range(natoms):
                for dim in range(dims):
                    T[conformer, atom, dim] += dr
                    conformer += 1
                    
        # Number of neural network core-jobs required is (natoms**2) * dimensions * drs
        species_tensors = self.Gmodels[state].species_to_tensor(mol.get_chemical_symbols())
        species_tensors = species_tensors.repeat(nconfs).reshape(nconfs, natoms)
        
        self.Gmodels[state].Multi_Coords = T.to(device)
        self.Gmodels[state].Multi_Species = species_tensors.to(device)
        
        MultiChemSymbols = np.tile(mol.get_chemical_symbols(), nconfs).reshape(nconfs, -1)
        self.Gmodels[state].MultiChemSymbols = MultiChemSymbols

        batch_dE = self.Gmodels[state].ProcessTensors(units="kcal/mol", return_all=False) # return_all is about all models, not all conformers            
        batch_dE = batch_dE.reshape((len(drs), natoms, 3))
        batch_dE = batch_dE - E0
        
        batch_Forces = np.ndarray((len(drs), natoms, 3))
        for i in range(batch_Forces.shape[0]):
            batch_Forces[i] = batch_dE[i] / drs[i]
            batch_Forces[i] -= batch_Forces[i].min(axis=0)
        return batch_Forces.mean(axis=0)
    
        
    def Min(self, idx: int, state: str, fix_atoms: list = [], reload_fmax=True, Plot=True, traj_ext=""):
        maxstep = 0.01
        trajfile = f"{self.work_folder}/Min_{idx}_{state}{traj_ext}.xyz"
        self.input_structures[idx][state].positions -= self.input_structures[idx][state].positions.min(axis=0)
        self.input_structures[idx][state].write(trajfile, append=False)

        Y = [self.input_structures[idx][state].get_potential_energy()* 23.06035]
        Fmax = []
        for minstep in tqdm.tqdm(range(150)):
            reset_pos = self.input_structures[idx][state].positions.copy()
            Forces = self.get_forces(idx, state)
            for atom_indice in fix_atoms:
                #Forces[atom_indice] = [0,0,0]
                pass
            Fmax.append(np.abs(Forces).max())
            Y.append(self.input_structures[idx][state].get_potential_energy()* 23.06035)
            if len(fix_atoms) > 0:
                Forces[fix_atoms] = 0
            step = (Forces / Forces.max()) * maxstep

            self.input_structures[idx][state].positions -= step
            self.input_structures[idx][state].positions -= self.input_structures[idx][state].positions.min(axis=0)
            self.input_structures[idx][state].write(trajfile, append=True)
            
            if len(Y) > 5:
                Grad = np.gradient(Y-Y[0])[-1]
                # Gromacs algorithm
                if Grad > 0:
                    maxstep *= 0.2
                    self.input_structures[idx][state].positions = reset_pos.copy()
                    Y[-1] = self.input_structures[idx][state].get_potential_energy()* 23.06035
                else:
                    maxstep *= 1.1
                if maxstep < 0.01:
                    break

        if reload_fmax:
            print("Reloading at Fmax=", np.min(Fmax), np.argmin(Fmax))
            self.input_structures[idx][state] = read(trajfile, index=np.argmin(Fmax))
            self.input_structures[idx][state].calc = self.Gmodels[state].SUPERCALC
        
        Y = np.array(Y)
        Fmax = np.array(Fmax)
        if Plot:
            dY = Y-Y[0]
            dFmax = Fmax-Fmax[0]
            plt.plot(dY)
            plt.scatter([np.argmin(Fmax)], [dY[np.argmin(Fmax)]], marker="1", color="red", s=100)
            plt.ylabel("$\\Delta$G")
            plt.plot(np.gradient(dY))
            plt.plot(dFmax)
            plt.title(f"{idx}_{state}")
            plt.tight_layout()
            plt.savefig(f"{self.work_folder}/Min_{idx}_{state}.png")
            plt.show()
        return Y, Fmax
    

    def generate_confs(self, idx, state):
        Connections = self.find_connections(idx, state)
        asemol = self.yates_mols[idx][state]["ase"].copy()
        atom_symbols = asemol.get_chemical_symbols()
        premol = Chem.RWMol()
        # Add atoms to the molecule
        for symbol, [xpos, ypos, zpos] in zip(atom_symbols, asemol.positions):
            atom = Chem.Atom(symbol)
            atom.SetDoubleProp("x", xpos)
            atom.SetDoubleProp("y", ypos)
            atom.SetDoubleProp("z", zpos)
            premol.AddAtom(atom)
        
        for connection in Connections.values():
            if connection["Type"] != "Bond":
                continue
            bond = Chem.BondType.SINGLE
            if connection["a0"] > connection["a1"]: # cant add the same one twice backwards
                continue
            if connection["s0"] == connection["s1"] == "C": # C=C
                # Check if it is aromatic
                possible_angles = [angle for angle in Connections.values() if angle["Type"] == "Angle" and angle["a0"] == connection["a0"] and angle["a1"] == connection["a1"]]
                CCC_angles = [angle for angle in possible_angles if angle["s0"] == angle["s1"] == angle["s2"] =="C"]
                if len(CCC_angles) > 0:
                    angles = np.array([angle["val"] for angle in CCC_angles])
                    if (abs((angles-120)) < 5).any():
                        bond = Chem.BondType.AROMATIC
                if bond != Chem.BondType.AROMATIC:
                    if connection["val"] < 1.44:
                        bond = Chem.BondType.DOUBLE
                    else:
                        bond = Chem.BondType.SINGLE    
            else:
                bond = Chem.BondType.SINGLE
            premol.AddBond(connection["a0"], connection["a1"], bond)
            print("Bonding:", connection["a0"], connection["a1"], connection["s0"], connection["s1"], bond)
        mol =  premol.GetMol()
        rdDepictor.Compute2DCoords(mol)
        
        print("Generating initial guesses")
        clearConfs = True
        NumConfs = 0
        pruneRmsThresh = 1.0
        with tqdm.tqdm(total = 1000) as pbar:
            while NumConfs < 1000:
                cids = AllChem.EmbedMultipleConfs(mol, # This doesnt include hydrogens in the RMS calculation!!
                                                  clearConfs=clearConfs,
                                                  numConfs=20, 
                                                  useBasicKnowledge = clearConfs,
                                                  maxAttempts=10000,
                                                  forceTol=0.01,
                                                  pruneRmsThresh = pruneRmsThresh)
                clearConfs = False
                pbar.update(mol.GetNumConformers() - NumConfs)
                NumConfs = mol.GetNumConformers()
                pruneRmsThresh *= 0.95
        asemol_guesses = []
        for i in range(mol.GetNumConformers()):
            asemol_guesses.append(Atoms(atom_symbols, mol.GetConformer(i).GetPositions()))
            if i > 0:
                minimize_rotation_and_translation(asemol_guesses[0], asemol_guesses[i])
                asemol_guesses[i].write(f"{self.work_folder}/{idx}_{state}_inputs_all.xyz", append=True)
            else:
                asemol_guesses[0].write(f"{self.work_folder}/{idx}_{state}_inputs_all.xyz", append=False)
        return asemol_guesses
    
    def boltzmann_dist(self, energies):
# =============================================================================
#         self.boltzmann_const_eV = 8.617333262145e-5  # Boltzmann constant in eV/K
#         self.Temperature = 298.15 # Temperature in K
#         self.boltzmann_exponents = []
#         for E in energies:
#             exp_term = np.exp(-E/(self.boltzmann_const_eV*self.Temperature))
#             self.boltzmann_exponents.append(exp_term)
#         self.boltzmann_distributions = []
#         for Exp in self.boltzmann_exponents:
#             distribution = (Exp / np.sum(self.boltzmann_exponents))
#             self.boltzmann_distributions.append(distribution)
#         return self.boltzmann_distributions
# =============================================================================
    
        beta = 1 / (298.15 * 8.617333262145e-5)  # Boltzmann constant in eV/K
        # Calculate the Boltzmann factors
        factors = -beta * energies
        # Use logsumexp to handle the exponentiation more stably
        log_partition_function = logsumexp(factors)
        # Calculate the Boltzmann probabilities
        probabilities = np.exp(factors - log_partition_function)
        return probabilities
        
        
    def filter_confs(self, idx, state, asemol_guesses, keep_n_confs = 10):
        print("Evaluating initial guesses")
        x.Gmodels[state].mol = asemol_guesses # Just copy the mols straight in, no need to write and reload from an xyz
        x.Gmodels[state].MakeTensors()
        G = x.Gmodels[state].ProcessTensors(units="kcal/mol", return_all=True) # return_all is about all models, not all conformers            
        # We want conformer guess that not only low energy but also that the ensemble of models are confident in
        rng = G.max(axis=0)-G.min(axis=0)
        mean = G.mean(axis=0)
        # Filter the 1000+ conformer starting points to low energy and high confidence
        indices = np.argsort(rng)[:keep_n_confs]
        indices = np.hstack((indices, np.argsort(mean)[:keep_n_confs]))
        indices = np.unique(indices)
        for i in range(len(indices)):
            asemol_guesses[i].write(f"{self.work_folder}/{idx}_{state}_inputs_filtered.xyz", append = (i!=indices[0]))
        return indices
        
    def __init__(self):
        self.mol_indices = [1,2,3,4,5,6,7,8,9,10,11]
        self.pKas = pandas.read_csv(os.path.join(os.path.dirname(__file__), "DFT_Data_pKa.csv"), index_col=0)
        self.radii = pandas.read_csv(os.path.join(os.path.dirname(__file__), "Alvarez2013_vdwradii.csv"), index_col=0)


if __name__ == "__main__":
    G_H = -4.39 # kcal/mol
    dG_solv_H = -264.61 # kcal/mol (Liptak et al., J M Chem Soc 2021)    
    x = pKa()
    #x.load_models("TrainDNN/model/", "best.pt"); x.work_folder = "Calculations/MSE"
    x.load_models("TrainDNN/models/", "best_L1.pt"); x.work_folder = "Calculations/Boltzmann_1_only"
    os.makedirs(x.work_folder, exist_ok=True)
    print(x.Gmodels)
    assert "prot_aq" in x.Gmodels
    x.load_yates()
    x.use_yates_structures()
    #sys.exit()
    

    predictions = pandas.DataFrame()
    for idx in [1,2,3,4,5,6,7,9,10,11]:
    #for idx in [1]:
        pkl_opt = f"{x.work_folder}/{idx}_optimization.pkl"
        if os.path.exists(pkl_opt):
            print("Reloading:", pkl_opt)
            optimization = pickle.load(open(pkl_opt, 'rb'))
        else:
            optimization = {}
            for state in ["prot_aq", "deprot_aq"]:
                optimization[state] = {}
                asemol_guesses = x.generate_confs(idx, state)
                confs_eV = []
                for j in asemol_guesses:
                    j.calc = x.Gmodels[state].SUPERCALC
                    confs_eV.append(j.get_potential_energy())
                #boltzmann_dist = x.boltzmann_dist(confs_eV)
                probabilities = x.boltzmann_dist(np.array(confs_eV))
                sorted_probs = sorted(probabilities, reverse=True)
                index_percent = int(len(sorted_probs) * 0.025)
                indices = sorted_probs[:index_percent]
                
                #indices = x.filter_confs(idx, state, asemol_guesses, keep_n_confs = 15)
                
                for i in range(len(indices)):
                    x.input_structures[idx][state] = asemol_guesses[i].copy()
                    x.input_structures[idx][state].calc = x.Gmodels[state].SUPERCALC
                    Y, Fmax = x.Min(idx, state, Plot=False, traj_ext=f"_{i}")
                    optimization[state][i] = {"G": Y,
                                              "Fmax": Fmax,
                                              "Final": x.input_structures[idx][state].get_potential_energy()* 23.06035,
                                               "ase": x.input_structures[idx][state].copy()}

                
            with open(pkl_opt, 'wb') as f:
                pickle.dump(optimization, f)
        
       
        for state in ["prot_aq", "deprot_aq"]:
            optimised_eV = []
            for atoms in optimization[state]:
                mol = optimization[state][atoms]['ase']
                mol.calc = x.Gmodels[state].SUPERCALC
                optimised_eV.append(mol.get_potential_energy())
            optimised_probabilities = x.boltzmann_dist(np.array(optimised_eV))
            indices = [np.argmax(optimised_probabilities)]
            #print(optimization)
            
    
            if state == 'prot_aq':
                prot_Gs = []
                for i in optimization[state]:
                    G_prot = optimization[state][i]["Final"]
                    #G_prot = optimization["prot_aq"][i]["G"].min()
                    prot_Gs.append(G_prot)
            if state == 'deprot_aq':
                deprot_Gs = []
                for i in optimization[state]:
                    G_deprot = optimization[state][i]["Final"]
                    deprot_Gs.append(G_deprot)
    
        for state in ["deprot_aq", "prot_aq"]:
            if state == "deprot_aq":    
                conformer = list(optimization[state].keys())[np.argmin(deprot_Gs)]
            else:
                conformer = list(optimization[state].keys())[np.argmin(prot_Gs)]
            final_mol = read(f"{x.work_folder}/Min_{idx}_{state}_{conformer}.xyz", index=np.argmin(optimization[state][conformer]["Fmax"]))
            final_mol.write(f"{x.work_folder}/FINAL_{idx}_{state}.xyz")
        
        
        
        G_deprot = min(deprot_Gs)
        G_prot = min(prot_Gs)
        guess_pka = ((G_deprot) - ((G_prot) - G_H - dG_solv_H))/(2.303*0.0019872036*298.15)
        print("Final Min: guess_pka:", guess_pka, "vs", x.pKas.at[idx, "pKa"])
        predictions.at[idx, "Pred"] = guess_pka
        predictions.at[idx, "Target"] = x.pKas.at[idx, "pKa"]
        predictions.at[idx, "Yates"] = x.pKas.at[idx, "Yates"]
     
        
        for label, G in zip(["deprot_aq", "prot_aq"], [G_deprot, G_prot]):
            yatesG = (x.yates_mols[idx][label]["ase"].get_potential_energy()* 23.06035)
            print(idx, label, G - yatesG, end=" ")
            if G > yatesG:
                print("(yates' lower, underoptimized)")
            else:
                print("(yates' higher, overoptimized)")
        
        
        X, Y = [], []
        for i in optimization["deprot_aq"]:
            Final = optimization["deprot_aq"][i]["Final"]
            Initial = optimization["deprot_aq"][i]["G"][0]
            X.append(Initial)
            Y.append(Final)
            dY = optimization["deprot_aq"][i]["G"]
            dY -= min(deprot_Gs)
            plt.plot(dY)
            Fmax = optimization["deprot_aq"][i]["Fmax"]
            plt.scatter([np.argmin(Fmax)], [dY[np.argmin(Fmax)]], marker="1", color="red", s=100)
        plt.show()


    for index in predictions.index:
        plt.text(predictions.at[index, "Pred"], predictions.at[index,"Target"], str(index))
    plt.scatter(predictions["Pred"], predictions["Target"], label="vs DFT")
    plt.scatter(predictions["Pred"], predictions["Yates"], label="vs Yates")
    plt.xlabel("Predicted $pK_a$")
    plt.legend()
    plt.plot([21, 35], [21, 35], lw=1, color="black")
    print("DFT RMSE:", mean_squared_error(predictions["Pred"], predictions["Target"], squared=False))
    print("Yates RMSE:", mean_squared_error(predictions["Pred"], predictions["Yates"], squared=False))
    print("DFT MAE:", mean_absolute_error(predictions["Pred"], predictions["Target"]))
    print("Yates RMSE:", mean_absolute_error(predictions["Pred"], predictions["Yates"]))
    print("DFT r2:", r2_score(predictions["Pred"], predictions["Target"]))
    print("Yates r2:", r2_score(predictions["Pred"], predictions["Yates"]))
    


