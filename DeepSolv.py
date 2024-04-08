# -*- coding: utf-8 -*-

from ase import Atoms
from ase.calculators.mixing import SumCalculator, MixedCalculator
from ase.optimize import BFGS
import pathlib
from ase.build import minimize_rotation_and_translation
from ase.io import read
from sklearn.metrics import root_mean_squared_error, mean_squared_error, euclidean_distances, mean_absolute_error, r2_score
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

from scipy.optimize import minimize


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
        assert os.path.exists(folder)
        assert len(glob.glob(f"{folder}/*/{checkpoint}")) > 0, "No .pt files of that name found"
        
        for fullpath in find_all(checkpoint, folder):
            p = pathlib.Path(fullpath).parent
            print(p)
            chk = fullpath.replace(folder, "")#.replace("\\", "/")

            if "Z=0" in p.name.upper():
                key = "deprot_"
            elif "Z=1" in p.name.upper():
                key = "prot_"
            if "AQ" in p.name.upper():
                key += "aq"
            elif "GAS" in p.name.upper():
                key += "gas"
            else:
                print("Cannot decode:", p.parent.name)
                sys.exit()
            
            if key not in self.Gmodels:
                self_energy = pathlib.Path(p, "Self_Energies.csv")
                training_config = pathlib.Path(p, "training_config.json")
                print("p:", p)
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
        self.yates_unopt_pKa.at['MSE', "MSE"] = mean_squared_error(self.yates_unopt_pKa['Yates_pKa_lit'].dropna(), self.yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna())
        self.yates_unopt_pKa.at['RMSE', "RMSE"] = root_mean_squared_error(self.yates_unopt_pKa['Yates_pKa_lit'].dropna(), self.yates_unopt_pKa['DFT_unopt_pKa_pred'].dropna())
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
                if mol[i].symbol == mol[j].symbol == mol[k].symbol == "H":
                    continue
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
            drs = [0.001] #+ [-0.01, -0.001]
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
# =============================================================================
#         # DEBUG
#         # Check the solution is making the right changes for the force calculations
#         self.T = T
#         force_confs = T.numpy()
#         for i in range(force_confs.shape[0]):
#             conf = Atoms(mol.get_chemical_symbols(), force_confs[i])
#             conf.write(f"{x.work_folder}/Forces_{idx}_{state}.xyz", append = (i!=0))
#         sys.exit()
# =============================================================================
        
        # Number of neural network core-jobs required is (natoms**2) * dimensions * drs
        species_tensors = self.Gmodels[state].species_to_tensor(mol.get_chemical_symbols())
        species_tensors = species_tensors.repeat(nconfs).reshape(nconfs, natoms)
        
        self.Gmodels[state].Multi_Coords = T.to(device)
        self.Gmodels[state].Multi_Species = species_tensors.to(device)
        
        MultiChemSymbols = np.tile(mol.get_chemical_symbols(), nconfs).reshape(nconfs, -1)
        self.Gmodels[state].MultiChemSymbols = MultiChemSymbols

        batch_dE = self.Gmodels[state].ProcessTensors(units="eV", return_all=False) # return_all is about all models, not all conformers            
        batch_dE = batch_dE.reshape((len(drs), natoms, 3))
        batch_dE = batch_dE - E0
        

        
        batch_Forces = np.ndarray((len(drs), natoms, 3))
        for i in range(batch_Forces.shape[0]):
            batch_Forces[i] = batch_dE[i] / drs[i]
            batch_Forces[i] -= batch_Forces[i].min(axis=0)
            
        return batch_Forces.mean(axis=0) 
    
    def Min_conjugateGD(self, idx: int, state: str, fix_atoms: list = [], reload_fmax=True, Plot=True, traj_ext=""):
        
        def energy_function(positions):
            # Set positions and calculate energy
            self.input_structures[idx][state].positions = positions.reshape((-1, 3))
            energy = self.input_structures[idx][state].get_potential_energy() * 23.06035
            return energy

        def gradient(positions):
            # Calculate forces and flatten the gradient
            forces = self.get_forces(idx, state)
            for atom_index in fix_atoms:
                forces[atom_index] = [0, 0, 0]
            return forces.flatten()

        maxstep = 0.01
        trajfile = f"{self.work_folder}/Min_{idx}_{state}{traj_ext}.xyz"
        self.input_structures[idx][state].positions -= self.input_structures[idx][state].positions.min(axis=0)
        self.input_structures[idx][state].write(trajfile, append=False)
    
        Y = [self.input_structures[idx][state].get_potential_energy()* 23.06035]
        Fmax = []
        
        x0 = self.input_structures[idx][state].positions.flatten()
        x = x0.copy()
        g = gradient(x)
        d = -g
        
        for minstep in tqdm.tqdm(range(150)):
# =============================================================================
#             reset_pos = self.input_structures[idx][state].positions.copy()
#             Forces = self.get_forces(idx, state)
#             for atom_indice in fix_atoms:
#                 #Forces[atom_indice] = [0,0,0]
#                 pass
# =============================================================================
            result = minimize(energy_function, x0, method='CG', jac=gradient, options={'disp': False})

            Fmax.append(np.abs(gradient(result.x)).max())
            Y.append(energy_function(result.x))

            if len(fix_atoms) > 0:
                self.input_structures[idx][state].positions[fix_atoms] = 0

            step = (gradient(result.x) / Fmax[-1]) * maxstep
            SGD = np.random.random(step.shape).round()
            step *= SGD

            self.input_structures[idx][state].positions -= step.reshape((-1, 3))
            self.input_structures[idx][state].positions -= self.input_structures[idx][state].positions.min(axis=0)
            self.input_structures[idx][state].write(trajfile, append=True)
            
            
# =============================================================================
#             Fmax.append(np.abs(g).max())
#             Y.append(self.input_structures[idx][state].get_potential_energy()* 23.06035)
#             
#             if len(fix_atoms) > 0:
#                 Forces[fix_atoms] = 0
#             
#             step = (g / Fmax[-1]) * maxstep
#             
#             x_new = x - step
#             g_new = self.get_forces(idx, state)
#             
#             g_new = g_new.flatten()
#             g = g.flatten()
# 
#             beta = np.dot(g_new, g_new - g) / np.dot(g, g)
#             d = -g_new + beta * d
# 
#             x = x_new
#             g = g_new
# 
#             self.input_structures[idx][state].positions = x.reshape((-1, 3))
#             self.input_structures[idx][state].positions -= self.input_structures[idx][state].positions.min(axis=0)
#             self.input_structures[idx][state].write(trajfile, append=True)
# =============================================================================
            
            if len(Y) > 5 and np.gradient(Y)[-1] > 0:
                maxstep *= 0.9
                x = x0.copy()
                #self.input_structures[idx][state].positions = reset_pos.copy()
                Y[-1] = energy_function(self.input_structures[idx][state].positions)
            elif maxstep < 0.1:
                maxstep *= 1.1
            if maxstep < 0.001:
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
            #plt.plot(dFmax)
            plt.title(f"{idx}_{state}")
            plt.tight_layout()
            plt.savefig(f"{self.work_folder}/Min_{idx}_{state}.png")
            plt.show()
        return Y, Fmax
        
    def Min(self, idx: int, state: str, fix_atoms: list = [], 
            reload_fmax=True, reload_gmin=False,
            Plot=True, traj_ext=""):
        print("Optimizing:", idx, state)
        #maxstep = 0.055
        #trajfile = f"{self.work_folder}/Min_{idx}_{state}{traj_ext}.xyz"
        self.input_structures[idx][state].positions -= self.input_structures[idx][state].positions.min(axis=0)
        #self.input_structures[idx][state].write(trajfile, append=False)
        
        
        opt = BFGS(self.input_structures[idx][state], maxstep=0.001)
        opt.initialize()
        
        Y = [self.input_structures[idx][state].get_potential_energy() * 23.06035]
        Fmax = []
        #self.input_structures[idx][state].write(f"{self.work_folder}/Debug.xyz", append=False)
        for i in tqdm.tqdm(range(100)):
            Forces = -self.get_forces(idx, state)
            Fmax.append(np.abs(Forces).max())
            opt.step(f=Forces)
            self.input_structures[idx][state].write(f"{self.work_folder}/Debug_{idx}_{state}.xyz", append=True)
            Y.append(self.input_structures[idx][state].get_potential_energy() * 23.06035)
        self.o = opt
        
        return np.array(Y), np.array(Fmax)
    
    
        sys.exit()
        Y = [self.input_structures[idx][state].get_potential_energy()* 23.06035]
        Fmax = []
        
        nsteps = 100
        
        for minstep in tqdm.tqdm(range(nsteps)):
            reset_pos = self.input_structures[idx][state].positions.copy()
            Forces = self.get_forces(idx, state)
            for atom_indice in fix_atoms:
                #Forces[atom_indice] = [0,0,0]
                pass
            Fmax.append(np.abs(Forces).max())
            Y.append(self.input_structures[idx][state].get_potential_energy()* 23.06035)
            if len(fix_atoms) > 0:
                Forces[fix_atoms] = 0
            step = (Forces / np.abs(Forces).max()) * maxstep

            self.input_structures[idx][state].positions -= step
            self.input_structures[idx][state].positions -= self.input_structures[idx][state].positions.min(axis=0)
            self.input_structures[idx][state].write(trajfile, append=True)
            
            if len(Y) > 5:
                # Gromacs algorithm
                if Y[-1] > Y[-2]: # The energy increased after the last step
                #if (Fmax[-1] - Fmax[-2]) > 0: # The energy increased after the last step
                    maxstep *= 0.2
                    #print("Maxstep:", maxstep)
                    self.input_structures[idx][state].positions = reset_pos.copy()
                    Y[-1] = self.input_structures[idx][state].get_potential_energy()* 23.06035
                elif maxstep < 0.1: # put a limit on how high the maxstep can climb
                    maxstep *= 1.2
                if maxstep < 0.001:
                    break

        if reload_fmax:
            print("Reloading at Fmax=", np.min(Fmax), np.argmin(Fmax))
            self.input_structures[idx][state] = read(trajfile, index=np.argmin(Fmax))
            self.input_structures[idx][state].calc = self.Gmodels[state].SUPERCALC
        elif reload_gmin:
            print("Reloading at Gmin=", np.min(Y), np.argmin(Y))
            self.input_structures[idx][state] = read(trajfile, index=np.argmin(Y))
            self.input_structures[idx][state].calc = self.Gmodels[state].SUPERCALC            
        
        Y = np.array(Y)
        Fmax = np.array(Fmax)
        if Plot:
            dY = Y-Y[0]
            dFmax = Fmax-Fmax[0]
            plt.plot(dY, label="dG")
            plt.scatter([np.argmin(Fmax)], [dY[np.argmin(Fmax)]], marker="1", color="red", s=300, label="Force Min")
            plt.scatter([np.argmin(Y)], [dY[np.argmin(Y)]], marker="1", color="blue", s=300, label="Free Energy Min")
            plt.ylabel("$\\Delta$G")
            plt.legend()
            #plt.plot(dFmax)
            plt.title(f"{idx}_{state}")
            plt.tight_layout()
            plt.savefig(f"{self.work_folder}/Min_{idx}_{state}.png")
            
            plt.show()
        return Y, Fmax

    def fname_guesses(self, idx, state):
        return f"{self.work_folder}/{idx}_{state}_initial_guesses.xyz"
    
    def generate_confs(self, idx, state, nconfs=1000):
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
        with tqdm.tqdm(total = nconfs) as pbar:
            while NumConfs < nconfs:
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
                asemol_guesses[i].write(self.fname_guesses(idx, state), append=True)
            else:
                asemol_guesses[0].write(self.fname_guesses(idx, state), append=False)
        return asemol_guesses
    
    def boltzmann_dist(self, energies):
        self.boltzmann_const_eV = 8.617333262145e-5  # Boltzmann constant in eV/K
        self.Temperature = 298.15 # Temperature in K
        self.boltzmann_exponents = []
        for E in energies:
            exp_term = np.exp(-E/(self.boltzmann_const_eV*self.Temperature))
            self.boltzmann_exponents.append(exp)
        self.boltzmann_distributions = []
        for Exp in self.boltzmann_exponents:
            distribution = (Exp / np.sum(self.boltzmann_exponents))
            self.boltzmann_distributions.append(distribution)
        return self.boltzmann_distributions
        
        
    def fname_guesses(self, idx, state):
        return f"{self.work_folder}/{idx}_{state}_initial_guesses.xyz"
    
    def fname_filtered(self, idx, state):
        return f"{self.work_folder}/{idx}_{state}_inputs_filtered.xyz"
    
    def guesses_energies(self, idx, state, asemol_guesses):
        x.Gmodels[state].mol = asemol_guesses # Just copy the mols straight in, no need to write and reload from an xyz
        x.Gmodels[state].MakeTensors()
        G = x.Gmodels[state].ProcessTensors(units="kcal/mol", return_all=False) # return_all is about all models, not all conformers            
        return G
        
    def filter_confs(self, idx, state, asemol_guesses, keep_n_confs = 10):
        print("Evaluating initial guesses")
        print("Filtering by RMSD")

        RMSD = np.zeros((len(asemol_guesses), len(asemol_guesses)))
        for i in range(0, len(asemol_guesses)):
            for j in range(0, len(asemol_guesses)):
                rmsd = np.sqrt(np.mean((asemol_guesses[i].positions - asemol_guesses[j].positions)**2))
                RMSD[i,j] = rmsd

        energies = self.guesses_energies(idx, state, asemol_guesses)
        keep = [np.argmin(energies)]
        if keep_n_confs > 1:
            keep.append(np.argmax(RMSD[keep[0]]))
            if keep_n_confs > 2:
                for i in range(keep_n_confs-2):
                    keep.append(np.argmax(RMSD[keep].mean(axis=0))) # ith
        assert len(keep) == keep_n_confs
        for i in range(len(keep)):
            asemol_guesses[i].write(self.fname_filtered(idx, state), append = (i!=keep[0]))
        return keep
        
    def __init__(self):
        self.mol_indices = [1,2,3,4,5,6,7,8,9,10,11]
        self.pKas = pandas.read_csv(os.path.join(os.path.dirname(__file__), "DFT_Data_pKa.csv"), index_col=0)
        #for idx in self.pKas.index:
            #self.pKas.at[idx, "pKa"] = ((self.pKas.at[idx, "deprot_aq"]*627.5095) - ((self.pKas.at[idx, "prot_aq"]*627.5095) - G_H - dG_solv_H))/(2.303*0.0019872036*298.15)
        self.radii = pandas.read_csv(os.path.join(os.path.dirname(__file__), "Alvarez2013_vdwradii.csv"), index_col=0)


if __name__ == "__main__":
    G_H = -4.39 # kcal/mol
    dG_solv_H = -264.61 # kcal/mol (Liptak et al., J M Chem Soc 2021)    
    x = pKa()
    #x.load_models("TrainDNN/model/", "best.pt"); x.work_folder = "Calculations/MSE"
    #x.load_models("TrainDNN/models/Alex_9010", "best.pt"); x.work_folder = "Calculations/Alex"
    #x.load_models("TrainDNN/models/Alex_9010", "best_L1.pt"); x.work_folder = "Calculations/Alex_noFmax"
    
    #x.load_models("TrainDNN/models/L1", "best.pt"); x.work_folder = "Calculations/Ross_ConjGD_test"
    x.load_models("TrainDNN/models/uncleaned", "best_L1.pt"); x.work_folder = "Calculations/crest_testing_Es_avogadro_0.001"
    os.makedirs(x.work_folder, exist_ok=True)
    print(x.Gmodels)
    assert "prot_aq" in x.Gmodels
    x.load_yates()
    x.use_yates_structures()
    
    
    predictions = pandas.DataFrame()
    conformer_numbers = {}
    conformer_numbers['prot_aq'] = {}
    conformer_numbers['deprot_aq'] = {}
    # for idx in [2]:
    for idx in [1,2,3,4,5,6,7,9,10,11]:
    # for idx in [1]:
        pkl_opt = f"{x.work_folder}/{idx}_optimization.pkl"
# =============================================================================
#         if os.path.exists(pkl_opt):
#             os.remove(pkl_opt)
# =============================================================================
        if os.path.exists(pkl_opt):
            print("Reloading:", pkl_opt)
            optimization = pickle.load(open(pkl_opt, 'rb'))
        else:
            optimization = {}
            for state in ["prot_aq"] + ["deprot_aq"]:
                optimization[state] = {}
# =============================================================================
#                 if os.path.exists(x.fname_guesses(idx, state)): # reload
#                     asemol_guesses = read(x.fname_guesses(idx, state), index=":")
#                 else:
#                     asemol_guesses = x.generate_confs(idx, state, nconfs=100)
#                 
#                 
#                 if os.path.exists(x.fname_filtered(idx, state).replace(".xyz", "_indices.txt")):
#                     indices = np.loadtxt(x.fname_filtered(idx, state).replace(".xyz", "_indices.txt")).astype(np.uint64)
#                     try:
#                         len(indices)
#                     except TypeError:
#                         indices = np.array([indices])
#                 else:
#                     indices = x.filter_confs(idx, state, asemol_guesses, keep_n_confs = 5)
#                     np.savetxt(x.fname_filtered(idx, state).replace(".xyz", "_indices.txt"), indices)
# =============================================================================
                if os.path.exists(f"Crest_Avogadro_Confs/{idx}_{state}/crest_conformers.xyz"):
                    asemol_guesses = read(f"Crest_Avogadro_Confs/{idx}_{state}/crest_conformers.xyz", index=':')
                    print('Crest Generated Conformers Loaded')
                    
                indices = []
                count = 0
                for i in range(len(asemol_guesses)):
                    indices.append(i)
                    count += 1
                conformer_numbers[state][idx] = count
# =============================================================================
#                 
#                 # Scan minimization
#                 print("Scanning Minimization")
#                 i = indices[0]
#                 if not os.path.exists(f"{x.work_folder}/Scan_{i}_{state}.xyz"):
#                     x.input_structures[idx][state] = asemol_guesses[i].copy()
#                     x.input_structures[idx][state].calc = x.Gmodels[state].SUPERCALC
#                     Connections = x.find_connections(idx, state)
#                     G = [x.input_structures[idx][state].get_potential_energy()*23.06055]
#                     
#                     mol = asemol_guesses[i].copy()
#                     for scan_iter in range(3):
#                         for conn in tqdm.tqdm([x for x in Connections if Connections[x]["Type"] == "Dihedral"]):
#                             a0 = Connections[conn]["a0"]
#                             a1 = Connections[conn]["a1"]
#                             a2 = Connections[conn]["a2"]
#                             a3 = Connections[conn]["a3"]
#                             
#                             mols = []
#                             calculator = x.Gmodels[state]
#                             for dihedral in np.arange(0, 360, 1):
#                                 mol.set_dihedral(a0,a1,a2,a3, dihedral)
#                                 mols.append(mol.copy())
#                             calculator.mol = mols
#                             calculator.MakeTensors()
#                             pred = calculator.ProcessTensors(units="Ha", return_all=True)
#                             pred *= 627.5
#                             pred = pred.mean(axis=0)
#                             G.append(pred.min())
#                             pred-=pred.min()
#                             #plt.plot(pred[0])
#                             #plt.plot(pred[1])
#                             
#                             mols[np.argmin(pred)].write(f"{x.work_folder}/Scan_{i}_{state}.xyz", append=True)
#                             mol = mols[np.argmin(pred)].copy()
#                             
#                         for conn in tqdm.tqdm([x for x in Connections if Connections[x]["Type"] == "Angle"]):
#                             a0 = Connections[conn]["a0"]
#                             a1 = Connections[conn]["a1"]
#                             a2 = Connections[conn]["a2"]
#                             
#                             mols = []
#                             calculator = x.Gmodels[state]
#                             for angle in np.arange(0, 360, 1):
#                                 mol.set_angle(a0,a1,a2, angle)
#                                 mols.append(mol.copy())
#                             calculator.mol = mols
#                             calculator.MakeTensors()
#                             pred = calculator.ProcessTensors(units="Ha", return_all=True)
#                             pred *= 627.5
#                             pred = pred.mean(axis=0)
#                             G.append(pred.min())
#                             pred-=pred.min()
#                             #plt.plot(pred[0])
#                             #plt.plot(pred[1])
#                             
#                             mols[np.argmin(pred)].write(f"{x.work_folder}/Scan_{i}_{state}.xyz", append=True)
#                             mol = mols[np.argmin(pred)].copy()
# 
#                     plt.plot(G)
#                     plt.show()
# =============================================================================
                for i in indices:
                    #x.input_structures[idx][state] = read(f"{x.work_folder}/Scan_{i}_{state}.xyz", index=-1)
                    x.input_structures[idx][state] = asemol_guesses[i].copy()
                    x.input_structures[idx][state].calc = x.Gmodels[state].SUPERCALC
                    #Y, Fmax = x.Min(idx, state, Plot=False, traj_ext=f"_{i}", reload_fmax=False)
                    
                    Y, Fmax = x.Min(idx, state, Plot=True, traj_ext=f"_{i}", 
                                    reload_fmax=False, reload_gmin=True
                                    )
                    #Y, Fmax = x.Min_conjugateGD(idx, state, Plot=False, traj_ext=f"_{i}", reload_fmax=True)
                    optimization[state][i] = {"G": Y,
                                              "Fmax": Fmax,
                                              "Final": x.input_structures[idx][state].get_potential_energy()* 23.06035 / 627.5095}

 
            with open(pkl_opt, 'wb') as f:
                pickle.dump(optimization, f)
        #sys.exit()
        yates = {}
        for state in ["deprot_aq", "prot_aq"]:
            x.yates_mols[idx][state]["ase"].calc = x.Gmodels[state].SUPERCALC
            yates[state] = (x.yates_mols[idx][state]["ase"].get_potential_energy()* 23.06035)
        
        #print(optimization)
        prot_Gs = []
        deprot_Gs = []
        for i in optimization["prot_aq"]:
            #G_prot = optimization["prot_aq"][i]["Final"]
            G_prot = optimization["prot_aq"][i]["G"].min()
            plt.plot(optimization["prot_aq"][i]["G"], label=f"prot_{i}")
            prot_Gs.append(G_prot)
        plt.plot(np.arange(optimization["prot_aq"][i]["G"].shape[0]), [yates["prot_aq"]]*optimization["prot_aq"][i]["G"].shape[0], label="yates DNN G val")
        plt.plot(np.arange(optimization["prot_aq"][i]["G"].shape[0]), [x.yates_mols[idx]["prot_aq"]["DFT G"]]*optimization["prot_aq"][i]["G"].shape[0], label="yates DFT G val")
        plt.legend()
        plt.title(f"{idx} prot_aq")
        plt.show()
        for i in optimization["deprot_aq"]:
            #G_deprot = optimization["deprot_aq"][i]["Final"]
            G_deprot = optimization["deprot_aq"][i]["G"].min()
            plt.plot(optimization["deprot_aq"][i]["G"], label=f"deprot_{i}")
            
            deprot_Gs.append(G_deprot)
        plt.plot(np.arange(optimization["deprot_aq"][i]["G"].shape[0]), [yates["deprot_aq"]]*optimization["deprot_aq"][i]["G"].shape[0], label="yates DNN G val")
        plt.plot(np.arange(optimization["deprot_aq"][i]["G"].shape[0]), [x.yates_mols[idx]["deprot_aq"]["DFT G"]]*optimization["deprot_aq"][i]["G"].shape[0], label="yates DFT G val")
        plt.legend()
        plt.title(f"{idx} deprot_aq")
        plt.show()
        

        
        G_deprot = min(deprot_Gs)
        G_prot = min(prot_Gs)
        guess_pka = ((G_deprot) - ((G_prot) - G_H - dG_solv_H))/(2.303*0.0019872036*298.15)
        print("Final Min: guess_pka:", guess_pka, "vs", x.pKas.at[idx, "pKa"])
        predictions.at[idx, "Pred"] = guess_pka
        predictions.at[idx, "Target"] = x.pKas.at[idx, "pKa"]
        predictions.at[idx, "Yates"] = x.pKas.at[idx, "Yates"]
     
        
        for state, G in zip(["deprot_aq", "prot_aq"], [G_deprot, G_prot]):
            yatesG = yates[state]
            print(idx, state, G - yatesG, end=" ")
            if G > yatesG:
                print("(yates' lower, underoptimized)")
            else:
                print("(yates' higher, overoptimized)")
        
        

    for index in predictions.index:
        plt.text(predictions.at[index, "Pred"], predictions.at[index,"Target"], str(index))
    plt.scatter(predictions["Pred"], predictions["Target"], label="vs DFT")
    plt.scatter(predictions["Pred"], predictions["Yates"], label="vs Yates")
    plt.xlabel("Predicted $pK_a$")
    plt.legend()
    plt.plot([21, 35], [21, 35], lw=1, color="black")
    plt.savefig(f"{x.work_folder}/Predictions_Fig.png")
    print("DFT RMSE:", root_mean_squared_error(predictions["Pred"], predictions["Target"]))
    print("Yates RMSE:", root_mean_squared_error(predictions["Pred"], predictions["Yates"]))
    print("DFT MAE:", mean_absolute_error(predictions["Pred"], predictions["Target"]))
    print("Yates MAE:", mean_absolute_error(predictions["Pred"], predictions["Yates"]))
    print("DFT r2:", r2_score(predictions["Pred"], predictions["Target"]))
    print("Yates r2:", r2_score(predictions["Pred"], predictions["Yates"]))
    


