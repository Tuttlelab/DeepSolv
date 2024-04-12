# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:52:04 2024

@author: Alex
"""
import networkx as nx
import pathlib, json, pandas
import torch
from colour import Color
import orca_parser
from _DNN import *
import matplotlib.pyplot as plt
import numpy as np
from colour import Color

def get_color_gradient(value, min_value, max_value):
    """
    Generate an RGB color based on the given value within the scale of min_value to max_value.
    The color transitions smoothly from green (low) to red (high) using the colour module.

    Parameters:
    - value: The value to get the color for.
    - min_value: The minimum value of the scale.
    - max_value: The maximum value of the scale.

    Returns:
    A string representing the RGB color in hex format.
    """
    # Ensure the value falls within the given range
    value = max(min(value, max_value), min_value)
    
    # Calculate the position of the value in the range [0, 1]
    position = (value - min_value) / (max_value - min_value)
    
    # Define start (green) and end (red) colors
    green = Color("green")
    red = Color("blue")
    
    # Create a color gradient between green and red
    gradient = list(green.range_to(red, 101))  # Generate more points for a smoother transition
    
    # Select the color based on the position
    selected_color = gradient[int(position * 100)]
    
    return selected_color.rgb



device = torch.device("cpu")


state = "Gas"

# Load Neutral model
p = f"../TrainDNN/models/Final_model_set/{state}_Z=0_bs=2048"
self_energy = pathlib.Path(p, "Self_Energies.csv")
training_config = pathlib.Path(p, "training_config.json")
print("p:", p)
# Load traning_config to keep ANI parameters consistent
with open(training_config, 'r') as jin:
    training_config = json.load(jin)
NeutralSelfE = pandas.read_csv(self_energy, index_col=0)
species_order = NeutralSelfE.index
Neutral = IrDNN(NeutralSelfE, verbose=False, device=device, training_config=training_config, next_gen=False)
Neutral.GenModel(species_order)
Neutral.load_checkpoint(str(pathlib.Path(p, "best_L1.pt")), typ="Energy")
#Neutral.SwitchCalculator(0)

# Load Charged model
p = f"../TrainDNN/models/Final_model_set/{state}_Z=1_bs=2048"
self_energy = pathlib.Path(p, "Self_Energies.csv")
training_config = pathlib.Path(p, "training_config.json")
print("p:", p)
# Load traning_config to keep ANI parameters consistent
with open(training_config, 'r') as jin:
    training_config = json.load(jin)
ChargedSelfE = pandas.read_csv(self_energy, index_col=0)
species_order = ChargedSelfE.index
Charged = IrDNN(ChargedSelfE, verbose=False, device=device,training_config=training_config, next_gen=False)
Charged.GenModel(species_order)
Charged.load_checkpoint(str(pathlib.Path(p, "best_L1.pt")), typ="Energy")

MAXVAL = 0
#MAXVAL = 0.0558013916015625

viz = open("HowDoesAniHandleCharge.xyz", 'w')
for yates_i in range(1,12):
    investigate = f"../Data/DFT/{yates_i}+.out"
    op = orca_parser.ORCAParse(investigate)
    op.parse_charges()
    mol = op.asemol
    
    # Find the difference in per-atom energies
    Energies = pandas.DataFrame()
    species = Neutral.CALC.species_to_tensor(mol.get_chemical_symbols()).to(device)
    species = species.unsqueeze(0)
    print(species)
    coordinates = torch.tensor(mol.get_positions())
    coordinates = coordinates.to(device).to(next(Neutral.model.parameters()).dtype)
    coordinates = coordinates.unsqueeze(0)
    
    # Neutral
    AEV = Neutral.aev_computer((species, coordinates))
    for i in range(mol.positions.shape[0]):
        E = Neutral.nn((AEV.species[0,i].unsqueeze(0).unsqueeze(0), AEV.aevs[0,i].unsqueeze(0).unsqueeze(0)))
        Energies.at[i, "atom"] = mol.get_chemical_symbols()[i]
        Energies.at[i, "Neutral"] = E.energies.detach().numpy() + NeutralSelfE.at[mol.get_chemical_symbols()[i], "SelfEnergy"]
    # Charged    
    AEV = Charged.aev_computer((species, coordinates))
    for i in range(mol.positions.shape[0]):
        E = Charged.nn((AEV.species[0,i].unsqueeze(0).unsqueeze(0), AEV.aevs[0,i].unsqueeze(0).unsqueeze(0)))
        Energies.at[i, "Charged"] = E.energies.detach().numpy() + ChargedSelfE.at[mol.get_chemical_symbols()[i], "SelfEnergy"]
    
    Energies["d"] = Energies["Charged"] - Energies["Neutral"]
    Energies["Mulliken"] = op.charges["Mulliken"]
    print(Energies)
    plt.scatter(Energies["d"], Energies["Mulliken"])
    # Find the min max scale
# =============================================================================
#     max_value = MAXVAL
#     min_value = -MAXVAL
# =============================================================================
    if Energies["d"].max() > abs(Energies["d"].min()):
        max_value = abs(Energies["d"].max())
        min_value = -Energies["d"].max()
    else:
        max_value = abs(Energies["d"].min())
        min_value = Energies["d"].min()    
    if max_value > MAXVAL:
        MAXVAL = max_value
    
    
    viz.write(f"{mol.positions.shape[0]}\n\n")
    for i in range(mol.positions.shape[0]):
        color = get_color_gradient(Energies.at[i, "d"], min_value, max_value)
        viz.write(Energies.at[i, "atom"])
        viz.write("\t")
        viz.write("\t".join([str(x) for x in mol.positions[i]]))
        viz.write("\t")
        viz.write("\t".join([str(x) for x in color]))
        viz.write("\n")
        
viz.close()

print("Green = more negative G value")
print("blue = more positive G value")
print("Global maxval:", MAXVAL)




