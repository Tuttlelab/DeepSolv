# -*- coding: utf-8 -*-

from ase import Atoms
from ase.calculators.mixing import SumCalculator, MixedCalculator
from ase.optimize import LBFGS, BFGS
from ase.build import minimize_rotation_and_translation
from ase.io import read
from sklearn.metrics import mean_squared_error, root_mean_squared_error, euclidean_distances, mean_absolute_error, r2_score
import torch
import json
import pandas
import os, itertools, tqdm, pickle, warnings, sys, glob, time
import numpy as np
from colour import Color
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
import orca_parser 
from _DNN import *

from scipy.special import logsumexp
from itertools import combinations

from ase.optimize import GPMin, MDMin, FIRE


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
            print(f"{key} Checkpoint:  {fullpath} loaded successfully")

    def load_yates(self):
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
            guess_pKa = self.calculate_pka(deprot_aq_G, prot_aq_G)
            self.yates_unopt_pKa.at[mol_index, "DFT_unopt_pKa_pred"] = guess_pKa
            self.yates_unopt_pKa.at[mol_index, "Yates_pKa_lit"] = self.pKas.at[mol_index, "Yates"]
        self.yates_unopt_pKa.at['MSE', "MSE"] = mean_squared_error(self.yates_unopt_pKa['Yates_pKa_lit'].dropna(), self.yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna())
        self.yates_unopt_pKa.at['RMSE', "RMSE"] = root_mean_squared_error(self.yates_unopt_pKa['Yates_pKa_lit'].dropna(), self.yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna())
        self.yates_unopt_pKa.at['MAE', "MAE"] = mean_absolute_error(self.yates_unopt_pKa['Yates_pKa_lit'].dropna(), self.yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna())
            
    def load_PM7(self):
        self.PM7_pKa = pandas.DataFrame()
        for mol_index in self.mol_indices:
            prot_G = self.PM7['prot'][mol_index] * 627.5095
            deprot_G = self.PM7['deprot'][mol_index] * 627.5095
            
            guess_pKa = self.calculate_pka(deprot_G, prot_G)
            self.PM7_pKa.at[mol_index, 'PM7_pKa'] = guess_pKa
            self.PM7_pKa.at[mol_index, 'Yates_pKa_lit'] = self.pKas.at[mol_index, "Yates"]
        self.PM7_pKa.at['MSE', "MSE"] = mean_squared_error(self.PM7_pKa['Yates_pKa_lit'].dropna(), self.PM7_pKa['PM7_pKa'].dropna())
        self.PM7_pKa.at['RMSE', "RMSE"] = root_mean_squared_error(self.PM7_pKa['Yates_pKa_lit'].dropna(), self.PM7_pKa['PM7_pKa'].dropna())
        self.PM7_pKa.at['MAE', "MAE"] = mean_absolute_error(self.PM7_pKa['Yates_pKa_lit'].dropna(), self.PM7_pKa['PM7_pKa'].dropna())    
        
    def load_alternate(self, path_to_files):
        files = glob.glob(path_to_files)
        
        self.additional_mols = {}       
        for file in files:
            dft_folder = os.path.join(os.path.dirname(__file__), "DFT")
            filename = file.split("/")[-1].split("_")[0]
            self.additional_mols[filename] = {}
            self.additional_mols[filename]["deprot_aq"]  = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder, f"{filename}_deprot.out"))}
            self.additional_mols[filename]["prot_aq"]    = {"orca_parse": orca_parser.ORCAParse(os.path.join(dft_folder, f"{filename}_prot.out"))}
            
            for state in ['prot_aq', 'deprot_aq']:
                self.additional_mols[filename][state]["orca_parse"].parse_coords()
                self.additional_mols[filename][state]["orca_parse"].parse_free_energy()
                self.additional_mols[filename][state]["DFT G"] = self.additional_mols[filename][state]["orca_parse"].Gibbs * 627.5
                self.additional_mols[filename][state]["ase"] = Atoms(self.additional_mols[filename][state]["orca_parse"].atoms, 
                                                           self.additional_mols[filename][state]["orca_parse"].coords[-1])
                try:
                    self.additional_mols[filename][state]["ase"].calc = self.Gmodels[state].SUPERCALC
                except:
                    warnings.warn("Couldnt load model for "+state, MyWarning)
                self.additional_mols[filename][state]["Yates pKa"] = self.additional_pKas.at[1, "Yates"]
                
# =============================================================================
#                 op = orca_parser.ORCAParse(file)
#                 op.parse_coords()
#                 op.parse_free_energy()
#                 DFT_G = op.Gibbs * 627.5095
#                 mol = Atoms(symbols=op.atoms, positions=op.coords[-1])
# 
# 
#             self.additional_mols[filename][state] = {}
#             self.additional_mols[filename][state]['orca_parse'] = op
#             self.additional_mols[filename][state]['ase'] = mol
#             self.additional_mols[filename][state]['DFT G'] = DFT_G
#             self.additional_mols[filename][state]["ase"].calc = self.Gmodels[state].SUPERCALC
# =============================================================================

        for mol in self.additional_mols:
            deprot_aq_G = self.additional_mols[mol]['deprot_aq']['ase'].get_potential_energy()*23.06035
            prot_aq_G = self.additional_mols[mol]['prot_aq']['ase'].get_potential_energy()*23.06035
            guess_pKa = self.calculate_pka(deprot_aq_G, prot_aq_G)
            self.yates_unopt_pKa.at[12, "DFT_unopt_pKa_pred"] = guess_pKa
            self.yates_unopt_pKa.at[12, "Yates_pKa_lit"] = self.additional_pKas.at[1, "Yates"]
        
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

    def calc_pKa_full_cycle(self, idx, Deprot_aq, Deprot_gas, Prot_aq, Prot_gas):

        dGgas = (Deprot_gas + self.G_H) - Prot_gas
        dG_solv_A = Deprot_aq - Deprot_gas
        dG_solv_HA = Prot_aq - Prot_gas
        dG_aq = dGgas + dG_solv_A + self.dG_solv_H  - dG_solv_HA
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
    
    def Min(self, idx: int, state: str, fix_atoms: list = [], 
            reload_fmax=True, reload_gmin=False,
            Plot=True, traj_ext="", numsteps=100):
        print("Optimizing:", idx, state)
        maxstep = 0.055
        trajfile = f"{self.work_folder}/Min_{idx}_{state}{traj_ext}.xyz"
        self.input_structures[idx][state].positions -= self.input_structures[idx][state].positions.min(axis=0)
        
        opt = FIRE(self.input_structures[idx][state])
        opt.run()
        
        Y = [self.input_structures[idx][state].get_potential_energy() * 23.06035]
        Fmax = [self.get_forces(idx, state)]
        
        return np.array(Y), np.array(Fmax)
    
    def boltzmann_dist(self, energies): # energies in kcal/mol
    
        energies_normalized = energies - np.min(energies) # Normalize relative to the lowest energy
        beta = 1 / (298.15 * 1.987204259e-3)  # Boltzmann constant in kcal/mol K
        factors = -beta * energies_normalized # Calculate the Boltzmann factors
        log_partition_function = logsumexp(factors) # Use logsumexp to handle the exponentiation more stably
        probabilities = np.exp(factors - log_partition_function) # Calculate the Boltzmann probabilities
        
        return probabilities
                    
    def load_filter_confs(self, mol_indices, conformer_path, states):
        self.optimization = {}
        for idx in mol_indices:
            self.optimization[idx] = {}
            self.pkl_opt = f"{self.work_folder}/{idx}_optimization.pkl"
            self.Boltzmann_data[idx] = {}

            if os.path.exists(self.pkl_opt):
                print("Reloading:", self.pkl_opt)
                with open(f"{self.work_folder}/{idx}_optimization.pkl", 'rb') as file:
                    self.optimization[idx] = pickle.load(file)
            else:
                for state in states:
                    self.optimization[idx][state] = self.process_state(idx, state, conformer_path)
                with open(f"{self.work_folder}/{idx}_optimization.pkl", 'wb') as file:
                    pickle.dump(self.optimization, file)

    def process_state(self, idx, state, conformer_path):
        state_optimization = {}
        crest_conformers_path = f"{conformer_path}/{idx}_{state}/crest_conformers.xyz"
        if os.path.exists(crest_conformers_path):
            asemol_guesses = read(crest_conformers_path, index=':')
            confs_kcal, probabilities, highest_prob = self.calculate_probabilities(asemol_guesses, state)
            indices = self.filter_conformations(confs_kcal, highest_prob)

            for i, asemol_guess in enumerate(asemol_guesses):
                if i in indices:
                    state_optimization[i] = self.optimize_conformation(idx, state, i, asemol_guess)
        return state_optimization

    def calculate_probabilities(self, asemol_guesses, state):
        confs_kcal = [self.calculate_energy(molecule, state) for molecule in asemol_guesses]
        probabilities = self.boltzmann_dist(np.array(confs_kcal))
        highest_prob = np.argmax(probabilities)
        return confs_kcal, probabilities, highest_prob

    def calculate_energy(self, molecule, state):
        molecule.calc = self.Gmodels[state].SUPERCALC
        return molecule.get_potential_energy() * 23.06035

    def filter_conformations(self, confs_kcal, highest_prob):
        highest_prob_E = confs_kcal[highest_prob]
        closest = np.where(np.array(confs_kcal) <= highest_prob_E + 5.0)[0]
        indices = list({highest_prob, *closest})
        return indices

    def optimize_conformation(self, idx, state, i, asemol_guess):
        self.input_structures[idx][state] = asemol_guess.copy()
        self.input_structures[idx][state].calc = self.Gmodels[state].SUPERCALC
        Y, Fmax = self.Min(idx, state, Plot=True, traj_ext=f"_{i}", 
                           reload_fmax=False, reload_gmin=True, numsteps=100)
        return {"G": Y, "Fmax": Fmax, "ase": self.input_structures[idx][state].copy(), "conf": i}
    
    def plot_optimizations(self, idx):
        yates = {}
        for state in ["deprot_aq", "prot_aq"]:
            x.yates_mols[idx][state]["ase"].calc = x.Gmodels[state].SUPERCALC
            yates[state] = (x.yates_mols[idx][state]["ase"].get_potential_energy()* 23.06035)
        #print(optimization)
        prot_Gs = []
        deprot_Gs = []
        for i in self.optimization[idx]["prot_aq"]:
            G_prot = self.optimization[idx]["prot_aq"][i]["G"].min()
            plt.plot(self.optimization[idx]["prot_aq"][i]["G"], label=f"prot_{i}")
            prot_Gs.append(G_prot)
        plt.plot(np.arange(self.optimization[idx]["prot_aq"][i]["G"].shape[0]), [yates["prot_aq"]]*self.optimization[idx]["prot_aq"][i]["G"].shape[0], label="yates DNN G val")
        plt.plot(np.arange(self.optimization[idx]["prot_aq"][i]["G"].shape[0]), [x.yates_mols[idx]["prot_aq"]["DFT G"]]*self.optimization[idx]["prot_aq"][i]["G"].shape[0], label="yates DFT G val")
        plt.legend()
        plt.title(f"{idx} prot_aq")
        plt.show()
        for i in self.optimization[idx]["deprot_aq"]:
            G_deprot = self.optimization[idx]["deprot_aq"][i]["G"].min()
            plt.plot(self.optimization[idx]["deprot_aq"][i]["G"], label=f"deprot_{i}")
            deprot_Gs.append(G_deprot)
        plt.plot(np.arange(self.optimization[idx]["deprot_aq"][i]["G"].shape[0]), [yates["deprot_aq"]]*self.optimization[idx]["deprot_aq"][i]["G"].shape[0], label="yates DNN G val")
        plt.plot(np.arange(self.optimization[idx]["deprot_aq"][i]["G"].shape[0]), [x.yates_mols[idx]["deprot_aq"]["DFT G"]]*self.optimization[idx]["deprot_aq"][i]["G"].shape[0], label="yates DFT G val")
        plt.legend()
        plt.title(f"{idx} deprot_aq")
        plt.show()
    
    def post_opt_filtering(self, idx, state, opt=True):
        
        energies_kcal = []
        self.Boltzmann_data[idx] = {}
    
        if opt == True:
            for i in self.optimization[idx][idx][state]:
                energies_kcal.append(self.optimization[idx][idx][state][i]['G'].min())
        elif opt == False:
            for i in self.crest_conformers[str(idx)][state]:
                energies_kcal.append(self.crest_conformers[str(idx)][state][i]['DFT G'])

        optimised_probabilities = self.boltzmann_dist(np.array(energies_kcal))
        most_probable = np.argmax(optimised_probabilities)
        energy_most_probable = energies_kcal[most_probable]
        closest = np.where(np.array(energies_kcal) <= energy_most_probable + 3.0)[0]
        
        if len(closest) != 1:
            indices = list(set([most_probable] + closest.tolist()))
        else:
            indices = [most_probable]    
    
        probable_conformers = {}
        if opt == True:
            for struct in indices:
                probable_conformers[struct] = {
                    "G_kcal": energies_kcal[struct],
                    "ase": self.optimization[idx][idx][state][struct]['ase'],
                    "probability": optimised_probabilities[struct],
                    "conf": self.optimization[idx][idx][state][struct]['conf']}
            self.Boltzmann_data[idx][state] = {
                'Opt_probs': optimised_probabilities,
                'Opt_confs_kcal': energies_kcal,
                'Opt_Rel_E': energies_kcal - np.min(energies_kcal),
                'Opt_Cut': [x for x in range(len(optimised_probabilities)) if x not in indices]}
        elif opt == False:
            for struct in indices:

                probable_conformers[struct] = {
                    "G_kcal": energies_kcal[struct],
                    "ase": self.crest_conformers[str(idx)][state][struct]['ase'],
                    "probability": optimised_probabilities[struct],
                    "conf": self.crest_conformers[str(idx)][state][struct]['conf']}

        return probable_conformers
    
    def RMSD_probable_conformers(self):
        keep = {'prot_aq': [], 'deprot_aq': []}
    
        for state in ["deprot_aq", "prot_aq"]:
            if len(probable_conformers[state]) == 1:
                # Handle the case with only one conformer
                conformer = probable_conformers[state][0]['conf']
                final_mol = probable_conformers[state][0]["ase"]
                final_mol.write(f"{self.work_folder}/FINAL_{idx}_{state}.xyz")
                continue
            test_mol = []
            for i in probable_conformers[state]:
                # Assuming confs is defined earlier in your code
                confs = deprot_indices if state == "deprot_aq" else prot_indices
                if probable_conformers[state][i]['conf'] not in confs:
                    continue
                mol = probable_conformers[state][i]['ase']
                mol.calc = self.Gmodels[state].SUPERCALC
                mol.write(f"{self.work_folder}/FINAL_{idx}_{state}_{confs[i]}.xyz")
                test_mol.append((mol, probable_conformers[state][i]['G_kcal']))
            for (mol_1, kcal_1), (mol_2, kcal_2) in combinations(test_mol, 2):
                rmsd = orca_parser.calc_rmsd(mol_1.positions, mol_2.positions)
                # Determine which molecule to keep based on RMSD and energy
                better_mol_energy = kcal_1 if kcal_1 < kcal_2 else kcal_2
                if rmsd < 0.1:
                    # Remove the higher energy molecule if it's already in the list
                    if better_mol_energy not in keep[state]:
                        keep[state].append(better_mol_energy)
                else:
                    # Add both molecules if they are sufficiently different
                    if kcal_1 not in keep[state]:
                        keep[state].append(kcal_1)
                    if kcal_2 not in keep[state]:
                        keep[state].append(kcal_2)
        return keep
    
    def write_final_conformers(self, state, probable_conformers):
        for conf in probable_conformers[state]:
            mol = probable_conformers[state][conf]
            mol.write(f"{self.work_folder}/FINAL_{idx}_{state}.xyz")
    
    def calculate_pka(self, G_deprot, G_prot):
        ln_log_conversion = 2.303 # ln(10) / log(10) = N //// ln(10) = log(10)*N //// 2.303 = 1*N //// N = 2.303
        R = 0.0019872036 # Gas constant in kcal / K / mol
        T = 298.15 # Temperature in K
        guess_pka = ((G_deprot - (G_prot - self.G_H - self.dG_solv_H)) / (ln_log_conversion * R * T))
        #guess_pka = (((G_deprot + (self.dG_solv_H + self.G_H)) - G_prot) / ln_log_conversion * R * T)
        return guess_pka
    
    def calculate_weighted_pka(self, probable_conformers, idx, DFT_crest=False):
        if len(probable_conformers.keys()) == 2:
            if len(probable_conformers['prot_aq']) == 1 and len(probable_conformers['deprot_aq']) == 1:
                G_deprot = probable_conformers['deprot_aq'][0]["G_kcal"]
                G_prot = probable_conformers['prot_aq'][0]["G_kcal"]
                guess_pka = self.calculate_pka(G_deprot, G_prot, idx)
                if not DFT_crest:
                    print(f"Molecule {idx}: guess_pka: {round(guess_pka, 2)} vs {self.pKas.at[idx, 'pKa']}")
                self.update_predictions(idx, guess_pka, DFT_crest)
            else:
                energy_lists = {}
                boltzmann_weights = {}
                for state in ["deprot_aq", "prot_aq"]:
                    energy_lists[state] = [conformer['G_kcal'] for conformer in probable_conformers[state].values()]
                    boltzmann_weights[state] = self.boltzmann_dist(energy_lists[state])
                pKas_weighting = []
                for i, G_prot in enumerate(energy_lists['prot_aq']):
                    for j, G_deprot in enumerate(energy_lists['deprot_aq']):
                        guess_pka = self.calculate_pka(G_deprot, G_prot)
                        weight = boltzmann_weights['prot_aq'][i] + boltzmann_weights['deprot_aq'][j]
                        pKas_weighting.append((guess_pka, weight))
        
                total_weight = sum(weight for _, weight in pKas_weighting)
                weighted_guess = sum(guess_pka * weight for guess_pka, weight in pKas_weighting) / total_weight
                if not DFT_crest:
                    print(f"Molecule {idx}: guess_pka: {round(weighted_guess, 2)} vs {self.pKas.at[idx, 'pKa']}")
                self.update_predictions(idx, weighted_guess, DFT_crest)
        elif len(probable_conformers.keys()) == 4:
            if len(probable_conformers['prot_aq']) == 1 and len(probable_conformers['deprot_aq']) == 1 and len(probable_conformers['prot_gas']) == 1 and len(probable_conformers['deprot_gas']) == 1:
                G_deprot_aq = probable_conformers['deprot_aq'][0]["G_kcal"]
                G_prot_aq = probable_conformers['prot_aq'][0]["G_kcal"]
                G_deprot_gas = probable_conformers['deprot_gas'][0]["G_kcal"]
                G_prot_gas = probable_conformers['prot_gas'][0]["G_kcal"]
                guess_pka = self.calc_pKa_full_cycle(idx, G_deprot_aq, G_deprot_gas, G_prot_aq, G_prot_gas)
                if not DFT_crest:
                    print(f"Molecule {idx}: guess_pka: {round(guess_pka, 2)} vs {self.pKas.at[idx, 'pKa']}")
                self.update_predictions(idx, guess_pka, DFT_crest)
            else:
                energy_lists = {}
                boltzmann_weights = {}
                for state in ["deprot_aq", "prot_aq", "prot_gas", "deprot_gas"]:
                    energy_lists[state] = [conformer['G_kcal'] for conformer in probable_conformers[state].values()]
                    boltzmann_weights[state] = self.boltzmann_dist(energy_lists[state])
                pKas_weighting = []
                for i, G_prot_aq in enumerate(energy_lists['prot_aq']):
                    for j, G_deprot_aq in enumerate(energy_lists['deprot_aq']):
                        for k, G_prot_gas in enumerate(energy_lists['prot_gas']):
                            for l, G_deprot_gas in enumerate(energy_lists['deprot_gas']):
                                guess_pka = self.calc_pKa_full_cycle(idx, G_deprot_aq, G_deprot_gas, G_prot_aq, G_prot_gas)
                                weight = boltzmann_weights['prot_aq'][i] + boltzmann_weights['deprot_aq'][j] + boltzmann_weights['prot_gas'][k] + boltzmann_weights['deprot_gas'][l]
                                pKas_weighting.append((guess_pka, weight))
        
                total_weight = sum(weight for _, weight in pKas_weighting)
                weighted_guess = sum(guess_pka * weight for guess_pka, weight in pKas_weighting) / total_weight
                if not DFT_crest:
                    print(f"Molecule {idx}: guess_pka: {round(weighted_guess, 2)} vs {self.pKas.at[idx, 'pKa']}")
                self.update_predictions(idx, weighted_guess, DFT_crest)
            
    def update_predictions(self, idx, guess_pka, DFT_crest=False):
        self.predictions.at[idx, "Target"] = self.pKas.at[idx, "pKa"]
        self.predictions.at[idx, "Yates"] = self.pKas.at[idx, "Yates"]
        if DFT_crest:
            self.predictions.at[idx, "DFT_Boltzmann_Pred"] = round(guess_pka, 2)
        if not DFT_crest:
            self.predictions.at[idx, "Pred"] = round(guess_pka, 2)
    
    def print_results(self):
        for index in self.predictions.index:
            plt.text(self.predictions.at[index, "Pred"], self.predictions.at[index,"Target"], str(index))
        plt.scatter(self.predictions["Pred"], self.predictions["Yates"], label="vs Yates GM")
        plt.scatter(self.predictions["Pred"], self.predictions["Target"], label="vs DFT GM")
        plt.scatter(self.predictions["Pred"], self.predictions["DFT_Boltzmann_Pred"], label="vs DFT Ensemble")
        plt.xlabel("Predicted $pK_a$")
        plt.ylabel("Reference $pK_a$")
        plt.legend()
        plt.plot([21, 35], [21, 35], lw=1, color="black")
        print("DFT RMSE:", round(root_mean_squared_error(self.predictions["Pred"], self.predictions["Target"]),2))
        print("DFT Boltzmann RMSE:", round(root_mean_squared_error(self.predictions["Pred"], self.predictions["DFT_Boltzmann_Pred"]),2))
        print("Yates RMSE:", round(root_mean_squared_error(self.predictions["Pred"], self.predictions["Yates"]),2))
        print("DFT MAE:", round(mean_absolute_error(self.predictions["Pred"], self.predictions["Target"]),2))
        print("DFT Boltzmann MAE:", round(mean_absolute_error(self.predictions["Pred"], self.predictions["DFT_Boltzmann_Pred"]),2))
        print("Yates MAE:", round(mean_absolute_error(self.predictions["Pred"], self.predictions["Yates"]),2))
        print("DFT r2:", round(r2_score(self.predictions["Pred"], self.predictions["Target"]),2))
        print("DFT Boltzmann r2:", round(r2_score(self.predictions["Pred"], self.predictions["DFT_Boltzmann_Pred"]),2))
        print("Yates r2:", round(r2_score(self.predictions["Pred"], self.predictions["Yates"]),2))
    
    def DFT_crest_calc(self, path_to_files: str):
        files = glob.glob(path_to_files)
        self.crest_conformers = {}
        for file in files:
            filename = file.split("/")[-1]
            parts = filename.split('_')
            molecule = parts[0]
            state = parts[1] + '_' + parts[2]
            conf = int(parts[3].split(".")[0]) - 1
            if molecule not in self.crest_conformers:
                self.crest_conformers[molecule] = {}
            if state not in self.crest_conformers[molecule]:
               self.crest_conformers[molecule][state] = {}
            self.crest_conformers[molecule][state][conf] = {'orca_parse': orca_parser.ORCAParse(file)}
            self.crest_conformers[molecule][state][conf]['orca_parse'].parse_free_energy()
            self.crest_conformers[molecule][state][conf]['DFT G'] = self.crest_conformers[molecule][state][conf]['orca_parse'].Gibbs * 627.5095
            self.crest_conformers[molecule][state][conf]['ase'] = self.crest_conformers[molecule][state][conf]['orca_parse'].asemol
            self.crest_conformers[molecule][state][conf]['conf'] = conf

        for idx in [1,2,3,4,5,6,7,8,9,10,11]:
            self.DFT_probable_conformers = {'prot_aq': {}, 'deprot_aq': {}}
            for state in ['prot_aq', 'deprot_aq']:
                self.DFT_probable_conformers[state] = x.post_opt_filtering(idx, state, opt=False)
                
            self.calculate_weighted_pka(self.DFT_probable_conformers, idx, DFT_crest=True)
        
    def __init__(self):
        self.G_H = -4.39 # kcal/mol
        self.dG_solv_H = -264.6 # kcal/mol (Liptak et al., J M Chem Soc 2021)    
        self.mol_indices = [1,2,3,4,5,6,7,8,9,10,11]
        self.pKas = pandas.read_csv(os.path.join(os.path.dirname(__file__), "DFT_Data_pKa.csv"), index_col=0)
        self.additional_pKas = pandas.read_csv(os.path.join(os.path.dirname(__file__), "DFT_Data_additional_pKa.csv"), index_col=0)
        self.radii = pandas.read_csv(os.path.join(os.path.dirname(__file__), "Alvarez2013_vdwradii.csv"), index_col=0)
        self.PM7 = pandas.read_csv(os.path.join(os.path.dirname(__file__), "PM7_Data_pKa.csv"), index_col=0)
        self.Boltzmann_data = {}
        self.predictions = pandas.DataFrame()

if __name__ == "__main__":
    
    start = time.time()
    
    x = pKa()
    #x.load_models("TrainDNN/model/", "best.pt"); x.work_folder = "Calculations/MSE"
    x.load_models("TrainDNN/models/Final_model_set/", "best_L1.pt"); x.work_folder = "Calculations/Test_2_models_time_trial"
    os.makedirs(x.work_folder, exist_ok=True)
    print(x.Gmodels)
    assert "prot_aq" in x.Gmodels
    x.load_yates()
    x.use_yates_structures()
    # x.load_alternate('Complete_DFT_Outs/**/lys*.out')
    x.load_PM7()
# =============================================================================
#     direct_full_cycle = []
#     for idx in [1,2,3,4,5,6,7,8,9,10,11]:
#         pKa = x.calc_pKa_full_cycle(idx)
#         direct_full_cycle.append(pKa)
#         x.predictions.at[idx, 'Gas Full Cycle'] = pKa
# =============================================================================
    # states=['prot_aq', 'deprot_aq', 'prot_gas', 'deprot_gas']
    states=['prot_aq', 'deprot_aq']
    # sys.exit()    
    # x.load_filter_confs([1,2,3,4,5,6,7,8,9,10,11], "Crest_Avogadro_Confs/")
    x.load_filter_confs([1,2,3,4,5,6,7,8,9,10,11], "Crest_Avogadro_Confs/", states)
    # sys.exit()
    for idx in [1,2,3,4,5,6,7,8,9,10,11]:
        if len(states) == 4:
            probable_conformers = {'prot_aq': {}, 'deprot_aq': {}, 'prot_gas': {}, 'deprot_gas': {}}
        if len(states) == 2:
            probable_conformers = {'prot_aq': {}, 'deprot_aq': {}}
        for state in states:
            probable_conformers[state] = x.post_opt_filtering(idx, state)
            
        x.calculate_weighted_pka(probable_conformers, idx)
    x.DFT_crest_calc('/users/bwb16179/postgrad/2024_03/DFT_Crest_Conformers/finished_DFT_crest_mols_Opt/*.out')
    x.print_results()
    end = time.time()
    print(end - start)