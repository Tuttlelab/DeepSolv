# -*- coding: utf-8 -*-

import torch
import torchani
import torchani.nn as TNN
from torchani import utils
from torchani.units import hartree2kcalmol
import os, ase, sys, copy, glob, time, pandas
import numpy as np
from ase.io import read,  iread


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


def readin(fname):
    f = open(fname, 'r', errors="ignore")
    content = f.read()
    return content

def checkout(gv_out):
    return "Normal termination of Gaussian" in gv_out

def eV_to_X(eV, units):
    if units.lower() == "kcal" or units.lower() == "kcal/mol":
        return hartree2kcalmol(eV / ase.units.Hartree)
    elif units.lower() == "ha":
        return eV / ase.units.Hartree
    elif units.lower() == "ev":
        return eV
    else:
        print("Unknown unit type:", units)
        sys.exit()
    

class CorrectSelfE:
    #CorrectSelfE(SelfEnergies)
    def CalcSelfE(self, species):
        SelfE = 0
        for atom in species:
            SelfE += self.self_energies.at[atom, "SelfEnergy"]
        return SelfE
    def __init__(self, SelfEnergies):
        assert type(SelfEnergies) == pandas.core.frame.DataFrame
        self.self_energies = SelfEnergies

    
class Calculator(ase.calculators.calculator.Calculator):
    """TorchANI calculator for ASE

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
            sequence of all supported species, in order.
        model (:class:`torch.nn.Module`): neural network potential model
            that convert coordinates into energies.
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in :class:`ase.Atoms`
            object with the wrapped positions.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, self_energies, model, overwrite=False):
        super(Calculator, self).__init__()
        self.species_to_tensor = utils.ChemicalSymbolsToInts(list(self_energies.index))
        self.model = model
        self.overwrite = overwrite

        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype
        try:
            # We assume that the model has a "periodic_table_index" attribute
            # if it doesn't we set the calculator's attribute to false and we
            # assume that species will be correctly transformed by
            # species_to_tensor
            self.periodic_table_index = model.periodic_table_index
        except AttributeError:
            self.periodic_table_index = False
            
        self.CSE = CorrectSelfE(self_energies)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super(Calculator, self).calculate(atoms, properties, system_changes)
        cell = torch.tensor(self.atoms.get_cell(complete=True),
                            dtype=self.dtype, device=self.device)
        pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool,
                           device=self.device)
        pbc_enabled = pbc.any().item()

        if self.periodic_table_index:
            species = torch.tensor(self.atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        else:
            species = self.species_to_tensor(self.atoms.get_chemical_symbols()).to(self.device)

        species = species.unsqueeze(0)
        coordinates = torch.tensor(self.atoms.get_positions())
        coordinates = coordinates.to(self.device).to(self.dtype) \
                                 .requires_grad_('forces' in properties)

        if pbc_enabled:
            coordinates = utils.map2central(cell, coordinates, pbc)
            if self.overwrite and atoms is not None:
                atoms.set_positions(coordinates.detach().cpu().reshape(-1, 3).numpy())

        if 'stress' in properties:
            scaling = torch.eye(3, requires_grad=True, dtype=self.dtype, device=self.device)
            coordinates = coordinates @ scaling
        coordinates = coordinates.unsqueeze(0)

        if pbc_enabled:
            if 'stress' in properties:
                cell = cell @ scaling
            self.energy = self.model((species, coordinates), cell=cell, pbc=pbc).energies
        else:
            self.energy = self.model((species, coordinates)).energies
            
        mol_self_energy = self.CSE.CalcSelfE(self.atoms.get_chemical_symbols())
        self.energy += mol_self_energy

        self.energy *= ase.units.Hartree # ALL OF THESE ASE.UNITS.* ARE A MEANS OF CONVERTING TO eV
        self.results['energy'] = self.energy.item()
        self.results['free_energy'] = self.energy.item()

        if 'forces' in properties:
            forces = -torch.autograd.grad(self.energy.squeeze(), coordinates, retain_graph='stress' in properties)[0]
            self.results['forces'] = forces.squeeze(0).to('cpu').numpy() #* ase.units.Hartree

        if 'stress' in properties:
            volume = self.atoms.get_volume()
            stress = torch.autograd.grad(self.energy.squeeze(), scaling)[0] / volume
            self.results['stress'] = stress.cpu().numpy()

class SuperCalculator(ase.calculators.calculator.Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, SelfEnergies, model, nn, checkpoints, verbose=False, overwrite=False, next_gen=False):
        super(SuperCalculator, self).__init__()
        self.species_to_tensor = utils.ChemicalSymbolsToInts(list(SelfEnergies.index))
        self.model = model
        self.overwrite = overwrite
        self.nn = nn
        self.checkpoints = checkpoints
        self.verbose = verbose
        self.next_gen = next_gen
        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype
        try:
            # We assume that the model has a "periodic_table_index" attribute
            # if it doesn't we set the calculator's attribute to false and we
            # assume that species will be correctly transformed by
            # species_to_tensor
            self.periodic_table_index = self.model.periodic_table_index
        except AttributeError:
            self.periodic_table_index = False
        
        self.CSE = CorrectSelfE(SelfEnergies)


    def calculate(self, atoms=None, properties=['energy'], system_changes=ase.calculators.calculator.all_changes):
        super(SuperCalculator, self).calculate(atoms, properties, system_changes)
        cell = torch.tensor(self.atoms.get_cell(complete=True),
                            dtype=self.dtype, device=self.device)
        pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool,
                           device=self.device)
        pbc_enabled = pbc.any().item()

        # RIGHT NOT WE WILL ASSUME EVERYTHING IS A SINGLET
        NPROTONS = sum(self.atoms.get_atomic_numbers())
        Charge = torch.Tensor(1)
        if NPROTONS%2 == 1:
            Charge[0] = 1
        else:
            Charge[0] = 0
        
        
        if self.periodic_table_index:
            species = torch.tensor(self.atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        else:
            species = self.species_to_tensor(self.atoms.get_chemical_symbols()).to(self.device)

        species = species.unsqueeze(0)
        coordinates = torch.tensor(self.atoms.get_positions())
        coordinates = coordinates.to(self.device).to(self.dtype) \
                                 .requires_grad_('forces' in properties)

        if pbc_enabled:
            coordinates = utils.map2central(cell, coordinates, pbc)
            if self.overwrite and atoms is not None:
                atoms.set_positions(coordinates.detach().cpu().reshape(-1, 3).numpy())

        if 'stress' in properties:
            scaling = torch.eye(3, requires_grad=True, dtype=self.dtype, device=self.device)
            coordinates = coordinates @ scaling
        coordinates = coordinates.unsqueeze(0)
        
        self.EnsembleEnergy = np.ndarray((len(self.checkpoints),), dtype=np.float64)
        self.EnsembleForce = np.ndarray((len(self.checkpoints), coordinates.shape[1], coordinates.shape[2]), dtype=np.float32)
        self.EnsembleStress = np.ndarray((len(self.checkpoints), 3, 3), dtype=np.float32)
        
        mol_self_energy = self.CSE.CalcSelfE(self.atoms.get_chemical_symbols())
        


        for model_i, checkpoint in enumerate(self.checkpoints):
            self.nn.load_state_dict(checkpoint)
            if pbc_enabled:
                if 'stress' in properties:
                    cell = cell @ scaling
                self.energy = self.model((species, coordinates), cell=cell, pbc=pbc).energies
            else:
                if self.next_gen:
                    self.energy = self.model((species, coordinates, Charge)).energies
                else:
                    self.energy = self.model((species, coordinates)).energies
            
            self.energy += mol_self_energy
            self.energy *= ase.units.Hartree # ALL OF THESE ASE.UNITS.* ARE A MEANS OF CONVERTING TO eV
            self.results['energy'] = self.energy.item()
            self.results['free_energy'] = self.energy.item()

            self.EnsembleEnergy[model_i] = self.energy.item()
            
            if 'forces' in properties:
                forces = -torch.autograd.grad(self.energy.squeeze(), coordinates, retain_graph='stress' in properties)[0]
                self.results['forces'] = forces.squeeze(0).to('cpu').numpy() #* ase.units.Hartree
                self.EnsembleForce[model_i] = self.results['forces'] 
                if model_i == len(self.checkpoints):
                    self.results['forces'] = self.EnsembleForce.mean(axis=0)
                    pass
                
            if 'stress' in properties:
                volume = self.atoms.get_volume()
                stress = torch.autograd.grad(self.energy.squeeze(), scaling, retain_graph=True)[0] / volume
                self.results['stress'] = stress.cpu().numpy()
                if self.verbose:
                    print("coordinates.shape", coordinates.shape)
                    print("self.results['stress'].shape:", self.results['stress'].shape)
                self.EnsembleStress[model_i] = stress.cpu().numpy()
                if model_i == len(self.checkpoints):
                    #self.results['stress'] = EnsembleStress.mean(axis=0)
                    pass
                
        
        self.results['energy'] = self.EnsembleEnergy.mean()
        self.results['free_energy'] = self.EnsembleEnergy.mean()


class IrDNN:
    def load_molecules(self, fname, index=":"):
        mol_file_size = os.stat(fname).st_size /1024/1024/1024
        if mol_file_size > 4:
            print("Loading", fname, "with iread as its > 4 GB")
            self.mol = iread(fname, index=index, parallel = True)
            self.iread = True
        else:
            self.iread = False
            self.mol = read(fname, index=index, parallel = True)
            if len(self.mol) == 0:
                print("Couldn't load molecule(s):", fname)
                return False
            elif len(self.mol) == 1:
                self.mol = self.mol[0]
            if self.verbose:
                print("Ir loaded all molecules in:", fname)
                
    def get_calc(self):
        if len(self.checkpoints) > 1:
            #return self.SUPERSPLITCALC
            return self.SUPERCALC
        else:
            return self.CALC
        
    def set_calc(self):
        self.mol.set_calculator(self.get_calc())
        
    def set_calculator(self):
        self.mol.set_calculator(self.CALC)
    def set_SuperCalculator(self):
        self.mol.set_calculator(self.SUPERCALC)

    def GenModel(self, elements=["H", "C", "N", "O", "F",  "P", "Cl", "Ir"], aev_dim=None):
        if aev_dim is None:
            aev_dim = self.aev_computer.aev_length
        if self.verbose:
            print("aev_dim:", aev_dim)
        
        networks = {}
        networks["H_network"] = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 256),
            torch.nn.CELU(self.training_config["celu0"]),
            torch.nn.Linear(256, 192),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(160, 1)
        )
        networks["C_network"] = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 224),
            torch.nn.CELU(self.training_config["celu0"]),
            torch.nn.Linear(224, 192),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(160, 1)
        )
        networks["N_network"] = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 192),
            torch.nn.CELU(self.training_config["celu0"]),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(128, 1)
        )
        networks["O_network"] = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 192),
            torch.nn.CELU(self.training_config["celu0"]),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(128, 1)
        )
        networks["Cl_network"] = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(self.training_config["celu0"]),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(self.training_config["celu1"]),
            torch.nn.Linear(96, 1)
        )

        
        NNs = []
        for element in elements:
            NNs.append(networks[f"{element}_network"])
        
        try:
            self.nn = torchani.ANIModel(NNs, self.training_config["Min"], self.training_config["Max"])
        except TypeError:
            self.nn = torchani.ANIModel(NNs)

        def init_params(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=1.0)
                torch.nn.init.zeros_(m.bias)
        
        self.nn.apply(init_params)
        self.model = TNN.Sequential(self.aev_computer, self.nn).to(self.device)
    
    def load_checkpoint(self, checkpointfile, typ = "Both"):
        if typ in ["Both", "Energy", "Force"]:
            self.calculator_splits.append(typ)
        else:
            print(typ, "not in", ["Both", "Energy", "Force"], "Exiting...")
            sys.exit()
            
        if "*" in checkpointfile:
            files = glob.glob(checkpointfile)
            for file in files:
                print("Loading:", file)
                self._load_checkpoint(file)
        else:
            self._load_checkpoint(checkpointfile)
            
    def _load_checkpoint(self, checkpointfile):
        checkpoint = torch.load(checkpointfile, map_location=self.device)
        self.checkpoints.append(checkpoint)
        self.checkpointfiles.append(checkpointfile)
        self.SwitchCalculator(-1)
        #self.checkpoints = list(np.unique(self.checkpoints)) # just dont load multiple of the same checkpoint ??
        self.models.append(copy.copy(self.model))
        self.SUPERCALC = SuperCalculator(self.SelfEnergies, self.model, self.nn, self.checkpoints, 
                                            self.device, self.verbose, next_gen=self.next_gen)

    def SwitchCalculator(self, index):
        checkpoint = self.checkpoints[index]
        if self.verbose:
            print("checkpoint", checkpoint)
        #try:
        self.nn.load_state_dict(checkpoint) #best.pt
        #except RuntimeError:
        #self.nn.load_state_dict(checkpoint["nn"]) #latest.pt

        self.CALC = Calculator(self.SelfEnergies, self.model)
  
    def PredictEnergies(self, frame=":", units="kcal/mol", return_variance=False, TakeMean=True):
        if len(self.checkpoints) < 1:
            print("Ir: Not checkpoints loaded!")
            sys.exit()
        if len(self.checkpoints) < 2 and return_variance:
            print("Ir: Not enough checkpoints loaded to return a variance (min 2)!")
            sys.exit()
        
        if type(self.mol) == ase.atoms.Atoms:
            self.mol = [self.mol]
        
        if return_variance:
            Variances = np.ndarray((len(self.mol), ))
        if TakeMean == True:
            Energies = np.ndarray((len(self.mol), ))
        else:
            Energies = np.ndarray((len(self.mol), len(self.checkpoints)))
            
        for i,mol in enumerate(self.mol):
            if return_variance:
                E = np.ndarray((len(self.checkpoints),))
                for j in range(len(self.checkpoints)):
                    self.SwitchCalculator(j)
                    mol.set_calculator(self.CALC)
                    E[j] = mol.get_potential_energy() # eV
                Var = np.var(E)
                Variances[i] = Var
                if TakeMean:
                    Energies[i] = np.mean(E)
                else:
                    Energies[j] = E 
            else:
                mol.set_calculator(self.get_calc()) 
                E = mol.get_potential_energy() # eV
                Energies[i] = E 
        
        Energies = eV_to_X(Energies, units)
                
        if return_variance:
            return Energies, Variances
        else:
            return Energies
    
    
    def ProcessTensors(self, addSelfE = True, units="eV", return_all=False):
        #Currently is not setup to handle PBC!
        E = np.ndarray((0, self.MultiChemSymbols.shape[0]))
        mol_self_energy = map(self.CSE.CalcSelfE, self.MultiChemSymbols)
        mol_self_energy = np.array(list(mol_self_energy))
        i = 0
        while i < len(self.checkpoints):
            #checkpoint = torch.load(self.checkpoints[i], map_location=torch.device(self.device))
            try:
                self.nn.load_state_dict(self.checkpoints[i]) #best
            except RuntimeError:
                self.nn.load_state_dict(self.checkpoints[i]["nn"]) #latest.pt
            model_E = self.model((self.Multi_Species, self.Multi_Coords)).energies # This one will actually come out as Hartree
            if self.device != "cpu":
                model_E = model_E.cpu()
            model_E = model_E.detach().numpy()
            E = np.vstack((E, model_E))
            i+=1
        
        if addSelfE:
            E = E + mol_self_energy
        if not return_all:
            if len(self.checkpoints) > 1:
                var = np.var(E, axis=0)
                if (var > 0.01).any():
                    print("WARNING: High variance between DNN models, max. var. =", var.max())
            E = np.mean(E, axis=0)
            #E = E + mol_self_energy
    
        if units == "eV":
            E = E *ase.units.Hartree
        elif units == "Ha":
            pass
        elif "kcal" in units.lower():
            E = E * 627.5
        elif "kj" in units.lower():
            E = E * 2625.5
        else:
            print("ProcessTensors UNKNOWN UNITS:", units)
            return False
        return E#, var
    
    def MakeTensors(self, properties=['energy']):
        try:
            periodic_table_index = self.model.periodic_table_index
            if self.verbose:
                print("Found periodic_table_index:", periodic_table_index)
        except AttributeError:
            periodic_table_index = False
            if self.verbose:
                print("Found periodic_table_index:", periodic_table_index)
        
        #Currently is not setup to handle PBC!
        
        #cell = torch.tensor(self.mol[0].get_cell(complete=True), dtype=self.dtype, device=self.device)
        #pbc = torch.tensor(self.mol[0].get_pbc(), dtype=torch.bool, device=self.device)
        #pbc_enabled = pbc.any().item()
        
        
        if type(self.mol) == ase.atoms.Atoms:
            template_mol = self.mol
            self.mol = [self.mol]
        else:
            template_mol = self.mol[0]
        
        coord_shape = template_mol.get_positions().shape
        Multi_Species = np.ndarray((0, template_mol.get_atomic_numbers().shape[0]))
        self.MultiChemSymbols = np.ndarray((0, template_mol.get_atomic_numbers().shape[0]))
        Multi_Coords = np.ndarray((0, coord_shape[0], coord_shape[1]))
        for M in self.mol:
            if coord_shape[0] != self.species_to_tensor(M.get_chemical_symbols()).shape[0]:
                print("TO USE MakeTensors all species must have the same number of Atoms!!")
                return False
            if periodic_table_index:
                Multi_Species = np.vstack((Multi_Species, M.get_atomic_numbers()))
            else:
                Multi_Species = np.vstack((Multi_Species, self.species_to_tensor(M.get_chemical_symbols())))
            self.MultiChemSymbols = np.vstack((self.MultiChemSymbols, M.get_chemical_symbols()))
            Multi_Coords = np.vstack((Multi_Coords, M.get_positions().reshape(1, coord_shape[0], coord_shape[1])))
        
            #print(Multi_Species)
            
        self.Multi_Species = torch.tensor(Multi_Species, dtype=torch.long, device=self.device)
        self.Multi_Coords = torch.tensor(Multi_Coords)
        self.Multi_Coords = self.Multi_Coords.to(self.dtype).to(self.device).requires_grad_('forces' in properties)
        return True

    
    def __init__(self, SelfEnergies, verbose=False, device=torch.device("cpu"), training_config=dict(), next_gen=False):
        self.verbose = verbose
        self.species_order = list(SelfEnergies.index)
        self.species_to_tensor = utils.ChemicalSymbolsToInts(self.species_order)
        self.dtype = torch.float32
        self.training_config = training_config
        self.device = device
        self.next_gen = next_gen
        if self.verbose:
            print("Running on device:", self.device)
        
        Rcr = training_config["Rcr"]
        Rca = training_config["Rca"]
        EtaR = torch.tensor([1.6000000e+01], device=device)
        ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
        Zeta = torch.tensor([3.2000000e+01], device=device)
        ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
        EtaA = torch.tensor([8.0000000e+00], device=device)
        ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.8500000e+00, 2.2000000e+00], device=device)
        num_species = len(self.species_order)
        self.aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

        self.SelfEnergies = SelfEnergies
        self.CSE = CorrectSelfE(self.SelfEnergies)
        
        self.models = []
        self.checkpoints = []
        self.checkpointfiles = []
        self.calculator_splits = []
        



    