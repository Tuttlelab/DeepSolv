# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:55:22 2022

@author: bwb16179 & rkb19187

This file is a stripped back script of the original iridium training as it has
all of the unnecessary lines taken out. For the original script look in the 
scripts repo for "Original_Iridium_Training.py"
"""

# =============================================================================
# Import all the requirements
# =============================================================================
import matplotlib, os, sys
if os.name != "nt":
    matplotlib.use('Agg')
import torch
import torchani

# Temp patch, somethings wrong with the way torchany installs
from torchani import nn as torachani_nn


import math, pandas, h5py, pickle, time, json, copy, argparse
import matplotlib.pyplot as plt
import numpy as np

### TorchAni uses the 'random' module to do the .shuffle of the data
import random, time

# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol

# sub-modules
from Logger import *

# =============================================================================
# Configure the basic variables that we need
# =============================================================================
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--cuda', action='store_true', help='Use CUDA if possible', default=True)
parser.add_argument('--multigpu', action='store_true', help='Use CUDA if possible', default=False)
parser.add_argument('--KMP_DUPLICATE_LIB_OK', action='store_true', default=True)
parser.add_argument('--output', type=str, help='Output folder', required=True)
parser.add_argument('--mixed', action='store_true', help='Does the dataset contains mixed explicit charges', default=False, required=False)
#parser.add_argument('-Z', type=int, help='Charge', required=True)
parser.add_argument('--batch_size', type=int, help='batch_size', required=True)
parser.add_argument('--training_frac', type=float, help='Fraction of the dataset to be used for training', default=0.8)
parser.add_argument('--ds', type=str, help='HDF5 dataset for training', required=True)
parser.add_argument('--valds', type=str, help='HDF5 dataset for validation', required=False)
parser.add_argument('--hard_reset', action='store_true', help='Delete all the file from previous run', default=False, required=False)
parser.add_argument('--log', type=str, help='Verbose Log file', default="Log.txt", required=False)
parser.add_argument('--logfile', type=str, help='Training Log file', default="Training.log", required=False)
parser.add_argument('--model_n', type=int, help='Model number', default=1, required=False)
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate', default=False, required=False)
parser.add_argument('--fc', type=float, help='Force coefficient', default=0.0, required=False)
parser.add_argument('--GraphEveryEpoch', type=int, help='Graph Every N Epochs', default=10, required=False)
parser.add_argument('--HasForces', action='store_true', help='Do the datasets have forces and are we training against them?', default=False, required=False)
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-3, required=False)
parser.add_argument('--preLoadANI',  action='store_true', help='Transfer learning from another DNN', default=False, required=False)
parser.add_argument('--preLoadANI_model_path', type=str, help='Folder containing the network information', default="ANIWEIGHTS/ani-2x_8x/train0/networks", required=False)
parser.add_argument('--early_stop', type=float, help='early stop (kcal/mol)', default=0.1, required=False)
parser.add_argument('--early_stopping_learning_rate', type=float, help='early_stopping_learning_rate', default=1e-5, required=False)
parser.add_argument('--Rcr', type=float, help='Rcr', default=5.2, required=False)
parser.add_argument('--Rca', type=float, help='Rca', default=3.1, required=False)
parser.add_argument('--remove_self_energy',  action='store_true', help='remove_self_energy', default=False, required=False)
parser.add_argument('--preAEV',  action='store_true', help='preAEV', default=False, required=False)
parser.add_argument('--sigmoid',  action='store_true', help='Apply a sigmoid and scale it between Min and Max over the output', default=False, required=False)

parser.add_argument('--celu0',  type=float, help='celu0', required=True)
parser.add_argument('--celu1',  type=float, help='celu1', required=True)

parser.add_argument('--L1',  action='store_true',  help='Use L1 loss function', default=False)

# Stratified DS
parser.add_argument('--Stratified',  action='store_true', help='Stratified', default=False, required=False)
parser.add_argument('--TrainDS', type=str, help='Training dataset for Stratified training', required=False)
parser.add_argument('--TestDS', type=str, help='Training dataset for Stratified training', required=False)

args = parser.parse_args()
print(args)



config = vars(args) # converts it to dictionary
config["species_order"] =  ['H', 'C', 'N', 'O', 'Cl']
config["random_seed"] = int(time.time())
# Z = 1
#config["celu0"] = 0.05
#config["celu1"] = 0.01
#config["celu0"] = 1.0
#config["celu1"] = 1.0

print(config)

assert os.path.exists(config["ds"]), "Dataset not found"

config["cmd"] = " ".join(sys.argv[1:])

random.seed(config["random_seed"])
os.makedirs(config["output"], exist_ok=True)



def indices2atoms(specie):
    return [config["species_order"][x] for x in specie if x >= 0]
            
# =============================================================================
# Define the 'Logger' class so that we can save information that we need
# =============================================================================
Log = Logger(f"{config['output']}/{config['log']}" if type(config["log"]) == str else None, verbose=True)

pickled_training = f"{config['output']}/Training_{config['model_n']}.pkl"
pickled_testing = f"{config['output']}/Testing_{config['model_n']}.pkl"
pickled_SelfEnergies = f"{config['output']}/SelfEnergies_{config['model_n']}.pkl"

# =============================================================================
# Config the operation of a hard reset if needed
# =============================================================================

if config["hard_reset"] == True:
    Log.Log("Performing a HARD RESET")
    config["hard_reset"] = False
    if type(config["logfile"]) == str and os.path.exists(config["logfile"]):
        os.remove(config["logfile"])
    for file in [f"{config['output']}/Training.log", f"{config['output']}/DNN_training.png",
                 f"{config['output']}/best.pt", f"{config['output']}/latest.pt", f"{config['output']}/Verbose.log",
                 pickled_training, pickled_testing, pickled_SelfEnergies]:
        if os.path.exists(file):
            Log.Log(f"Removing: {file}")
            os.remove(file)
    
os.environ["KMP_DUPLICATE_LIB_OK"] = str(config["KMP_DUPLICATE_LIB_OK"])
plt.ioff()

# =============================================================================
# Ensure learning rate and checkpoints reset are consistent with config
# =============================================================================
reset_lr = config["reset_lr"]
latest_checkpoint = config['output']+"/latest.pt"

Log.Log("reset_lr: "+str(reset_lr))
Log.Log("latest_checkpoint:"+latest_checkpoint)

# =============================================================================
# Set the device to cuda or to CPU
# =============================================================================

Log.Log("CUDA availibility: "+str(torch.cuda.is_available()))
if config["cuda"] == False or torch.cuda.is_available() == False:
    device = torch.device('cpu')   
    Log.Log("FORCING TO CPU")
else:
    device = torch.device('cuda')
Log.Log("Running on device: "+str(device))

# =============================================================================
# Now we want to initialise the variables for the aev equations (Eqn 3 from the
# ANI paper)
# =============================================================================


Rcr = config["Rcr"] #5.2000e+00  # Cut-off
Rca = config["Rca"] # original 3.5
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.8500000e+00, 2.2000000e+00], device=device)


num_species = len(config["species_order"])
try:
    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=True)
    cuaev = True
except:
    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
    cuaev = False
Log.Log("cuaev: "+str(cuaev))
    
#aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
aev_dim = aev_computer.aev_length
energy_shifter = torchani.utils.EnergyShifter(None)


config["aev_dim"] = aev_dim+1 if config["mixed"] else aev_dim
with open(os.path.join(config["output"], "training_config.json"), 'w') as jout:
    json.dump(config, jout, indent=4)
    

# =============================================================================
# Configure some more variables, start time, cwd, path to the dataset and batch
# size
# =============================================================================

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()


batch_size = config["batch_size"]


starttime = int(time.time())

# =============================================================================
# Now we need to !process the dataset!
# 
# We pickle the dataset after loading to ensure we use the same testing set
# each time we restart training, otherwise we risk mixing the testing and
# training sets on each restart.
# This is also where we define the test:train split.
# We also subtract the self_energies here too.
# =============================================================================


if config["Stratified"]:
    ###
    # We DONT NEED TO PICKLE A STRATIFIED DATASET
    ###
    Log.Log(f"Processing dataset: {config['TrainDS']}")
    training = torchani.data.load(config["TrainDS"], additional_properties=('forces',))
    if config["remove_self_energy"]:
        training.subtract_self_energies(energy_shifter, config["species_order"])
    
    training.species_to_indices(config["species_order"])
    training.shuffle()
    
    testing = torchani.data.load(config["TestDS"], additional_properties=('forces',))
    if config["remove_self_energy"]:
        testing.subtract_self_energies(energy_shifter, config["species_order"])
    testing.species_to_indices(config["species_order"])
    testing.shuffle()
    
    
    training = training.collate(batch_size).cache()
    testing = testing.collate(batch_size).cache()
    x =  round(time.time()-starttime, 3)
    Log.Log(f"Stratified dataset generated and saved in {x} s")
else:
    Log.Log("treating this as a non-stratified dataset")
    
    if os.path.isfile(pickled_training) and not config["hard_reset"]:
        if config["remove_self_energy"]:
            Log.Log(f'Unpickling preprocessed dataset found in {pickled_SelfEnergies}')
            energy_shifter.self_energies = pickle.load(open(pickled_SelfEnergies, 'rb')).to(device)
        else:
            Log.Log(f'Setting energry_shifter to zeros')
            energy_shifter.self_energies = torch.Tensor([0]*len(config["species_order"]))
    
        Log.Log(f'Unpickling preprocessed dataset found in {pickled_testing}')
        testing = pickle.load(open(pickled_testing, 'rb')).collate(config["batch_size"]).cache()
    
        Log.Log(f'Unpickling preprocessed dataset found in {pickled_training}')
        training = pickle.load(open(pickled_training, 'rb')).collate(config["batch_size"]).cache()  
    
        x =  round(time.time()-starttime, 3)
        Log.Log("Dataset "+pickled_training+" already made")
    else:    
        Log.Log(f"Processing dataset:" +config["ds"])
        if config["remove_self_energy"]:
            training, testing = torchani.data.load(config["ds"], additional_properties=('Charge',))\
                                                .subtract_self_energies(energy_shifter, config["species_order"])\
                                                .species_to_indices(config["species_order"])\
                                                .shuffle()\
                                                .split(config["training_frac"], None)
        else:
            training, testing = torchani.data.load(config["ds"], additional_properties=('Charge',))\
                                                .species_to_indices(config["species_order"])\
                                                .shuffle()\
                                                .split(config["training_frac"], None)

        with open(pickled_training, 'wb') as f:
            pickle.dump(training, f)
        with open(pickled_testing, 'wb') as f:
            pickle.dump(testing, f)
        if config["remove_self_energy"]:
            with open(pickled_SelfEnergies, 'wb') as f:
                pickle.dump(energy_shifter.self_energies.cpu(), f)
    
        training = training.collate(batch_size).cache()
        testing = testing.collate(batch_size).cache()
        x =  round(time.time()-starttime, 3)
        Log.Log(f"Pickled dataset generated and saved in {x} s")

# Find min-max
if config["sigmoid"]:
    Max = np.ndarray((0,))
    Min = np.ndarray((0,))
    for p in training:
        Max = np.hstack((Max, p["energies"].max().numpy()))
        Min = np.hstack((Min, p["energies"].min().numpy()))
    for p in testing:
        Max = np.hstack((Max, p["energies"].max().numpy()))
        Min = np.hstack((Min, p["energies"].min().numpy()))
    Max = Max.max()
    Min = Min.min()
    config['Max'] = Max
    config['Min'] = Min
    print(f"Will scale outputs via a sigmoid function to between {config['Min']} and {config['Max']}")
else:
    config['Max'] = None
    config['Min'] = None
with open(os.path.join(config["output"], "training_config.json"), 'w') as jout:
    json.dump(config, jout, indent=4)




### Load validation set if it is part of the config:
if config["valds"] is not None:
    Log.Log("Loading validation set from: " + config["valds"])
    if config["remove_self_energy"]:
        validation = torchani.data.load(config["valds"], additional_properties=('forces',))\
                                            .subtract_self_energies(energy_shifter, config["species_order"])\
                                            .species_to_indices(config["species_order"])\
                                            .shuffle()
    else:
        validation = torchani.data.load(config["valds"], additional_properties=('forces',))\
                                            .species_to_indices(config["species_order"])\
                                            .shuffle()
    validation = validation.collate(batch_size).cache()
    
###Write self energies
if config["remove_self_energy"]:
    Log.Log('Self atomic energies: '+str(energy_shifter.self_energies))
    with open(os.path.join(config["output"], "Self_Energies.csv"), "w") as f:
        selfenergies_tensors = [t for t in energy_shifter.self_energies]
        if len(selfenergies_tensors) != len(config["species_order"]):
            Log.Log("len(selfenergies_tensors) != len(config['species_order'])")
            Log.Log("Exiting...")
            sys.exit()
        senergies = [x.item() for x in selfenergies_tensors]
        se_dict = {}
        ds = config["output"]
        ds = ds.split('/')[-1]
        #f.write("Self Energies for dataset: %s \n" % ds)
        f.write("Atom,SelfEnergy\n")
        for key in config["species_order"]:
            for value in senergies:
                se_dict[key] = value
                senergies.remove(value)
                break
        for k in se_dict.keys():
            f.write("%s, %s \n" % (k, se_dict[k]))
        f.close()
else:
    SE = pandas.DataFrame(index=config["species_order"], columns=["SelfEnergy"])
    SE[:] = 0
    SE.to_csv(os.path.join(config["output"], "Self_Energies.csv"))
# =============================================================================
# Now we can actually configure the model
# This involves initialising all the individual networks and then combining
# them all with the AEV parameters before they all get combined and wrapped
# into the one function
# =============================================================================


# Make the last layer a sigmoid that scales between max and min


networks = {}
networks["H_network"] = torch.nn.Sequential(
    torch.nn.Linear(config["aev_dim"], 256),
    torch.nn.CELU(config["celu0"]),
    torch.nn.Linear(256, 192),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(160, 1)
)
networks["C_network"] = torch.nn.Sequential(
    torch.nn.Linear(config["aev_dim"], 224),
    torch.nn.CELU(config["celu0"]),
    torch.nn.Linear(224, 192),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(160, 1)
)
networks["N_network"] = torch.nn.Sequential(
    torch.nn.Linear(config["aev_dim"], 192),
    torch.nn.CELU(config["celu0"]),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(128, 1)
)
networks["O_network"] = torch.nn.Sequential(
    torch.nn.Linear(config["aev_dim"], 192),
    torch.nn.CELU(config["celu0"]),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(128, 1)
)
networks["Cl_network"] = torch.nn.Sequential(
    torch.nn.Linear(config["aev_dim"], 160),
    torch.nn.CELU(config["celu0"]),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(config["celu1"]),
    torch.nn.Linear(96, 1)
)




NNs = []
for element in config["species_order"]:
    if config["preLoadANI"]:
        source_NN_path = config["preLoadANI_model_path"]+f"/ANN-{element}.nnf"
        Log.Log("Pre-loadingin weights of: " + source_NN_path)
        source_NN = torchani.neurochem.load_atomic_network(source_NN_path)
        for i in range(len(source_NN)):
            if hasattr(source_NN[i], "bias"):
                our_layer_size = networks[f"{element}_network"][i].weight.size()
                networks[f"{element}_network"][i].bias = source_NN[i].bias
                Log.Log(str(networks[f"{element}_network"][i].weight.size()) + "<-" + str(source_NN[i].weight.size()))
                if networks[f"{element}_network"][i].weight.size() == source_NN[i].weight.size():
                    networks[f"{element}_network"][i].weight = source_NN[i].weight
                else:
                    Log.Log(f"Mismatch in network size (probably different AEV dimensions on the input layer), only copying over a slice of size: [:our_layer_size[0], :our_layer_size[1]]")
                    networks[f"{element}_network"][i].weight = torch.nn.parameter.Parameter(source_NN[i].weight[:our_layer_size[0], :our_layer_size[1]])

                
    NNs.append(networks[f"{element}_network"])
nn = torchani.ANIModel(NNs, config['Min'], config['Max'])

# Function to randomize weights
def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        #torch.nn.init.uniform_(m.bias)
        torch.nn.init.zeros_(m.bias)
# Apply the function to randomize weightd
nn.apply(init_params)


if config["preAEV"]:
    Log.Log("Setting up neural network for pre-cooked AEV dataset")
    model = nn.to(device)
else:
    model =    torachani_nn.Sequential(aev_computer, nn).to(device)
    
# =============================================================================
# Now we set up the optimiser Adam with decoupled weight decay updates the
# weights and SGD updates the biases
# =============================================================================
AdamW_params = []
for network in NNs:
    AdamW_params.append({'params': [network[0].weight]})
    AdamW_params.append({'params': [network[2].weight], 'weight_decay': 0.00001})
    AdamW_params.append({'params': [network[4].weight], 'weight_decay': 0.000001})
    AdamW_params.append({'params': [network[6].weight]})


AdamW = torch.optim.AdamW(AdamW_params, lr = config["lr"])

SGD_params = []
for network in NNs:
    for i in range(len(network)):
        if hasattr(network[i], "bias"):
            SGD_params.append({'params': [network[i].bias]})
        
    
    
SGD = torch.optim.SGD(SGD_params, lr = config["lr"])

# =============================================================================
# Set up schedulers to do the updating of the learning rate for us 
# =============================================================================
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0, verbose=False)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0, verbose=False)

# =============================================================================
# Sort out the resumption of training or reset the learning rate (in params)
# =============================================================================

if os.path.isfile(latest_checkpoint):
    if device.type == "cpu":
        checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])
    L1_started = checkpoint["L1_started"]
else:
    L1_started = False



if reset_lr:
    Log.Log("Reset learning rates, you should only do this at the begining of a continuation!")
    for x in AdamW.param_groups:
        x["lr"] = config["lr"]
    for x in SGD.param_groups:
        x["lr"] = config["lr"]
    AdamW_scheduler._last_lr=[]
    SGD_scheduler._last_lr=[]
    AdamW_scheduler.best = 10000
    
# =============================================================================
# Now we can set up our testing loop
# =============================================================================

def test_set():
    # run testing set
    if config["L1"] and L1_started:
        LOSSFN = torch.nn.L1Loss(reduction='sum')
    else:
        LOSSFN = torch.nn.MSELoss(reduction='sum')
        
        
    total_loss = 0.0
    count = 0
    model.eval()
    for properties in testing:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).float()
        if config["mixed"]:
            charges = properties['Charge'].to(device).float()
            _, predicted_energies = model((species, coordinates, charges))
        else:
            _, predicted_energies = model((species, coordinates))

        #plt.scatter(true_energies.cpu().detach().numpy(), predicted_energies.cpu().detach().numpy())
        total_loss += LOSSFN(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    #plt.show()
    model.train()
    return hartree2kcalmol(math.sqrt(total_loss / count))

def validation_set():
    # run testing set
    mse_sum = torch.nn.MSELoss(reduction='sum')
    mae_sum = torch.nn.L1Loss(reduction='sum')
    total_mse = 0.0
    total_mae = 0.0
    count = 0
    model.eval()
    for properties in validation:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).float()
        charges = properties['Charge'].to(device).float()
        _, predicted_energies = model((species, coordinates, charges))
        #plt.scatter(true_energies.cpu().detach().numpy(), predicted_energies.cpu().detach().numpy())
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    model.train()
    #plt.show()
    return hartree2kcalmol(math.sqrt(total_mse / count))

# =============================================================================
# And finally we define the training for the network
# =============================================================================

#if config["L1"]:
#    LOSS = torch.nn.L1Loss(reduction='none')
#else:
#    LOSS = torch.nn.MSELoss(reduction='none')



Log.Log("training starting from epoch " + str(AdamW_scheduler.last_epoch + 1))
max_epochs = 10000
best_model_checkpoint = config['output']+"/best.pt"

TrainingLog = config['output']+"/Training.log"

if os.path.exists(TrainingLog):
    training_log = pandas.read_csv(TrainingLog, index_col=0)
else:
    training_log = pandas.DataFrame(columns=["Epoch", "Train", "Test", "Val", "lr"])
    

# Nothing has crashed up to this point so write out to the config file that we
# don't want to restart from scratch again next time

best_i = 0

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    if config["HasForces"]:
        energy_rmse, force_rmse = test_set()
        EF_coef = (energy_rmse * (1-config["fc"])) + (force_rmse * config["fc"])
        if config["valds"] is not None:
            val_energy_rmse, val_force_rmse = validation_set()
        else:
            val_energy_rmse, val_force_rmse = -1,-1
    else:
        energy_rmse = test_set()
        EF_coef = energy_rmse 
        if config["valds"] is not None:
            val_energy_rmse = validation_set()
        else:
            val_energy_rmse = -1
    
    Epoch = AdamW_scheduler.last_epoch + 1
    
    if EF_coef < config["early_stop"]:
        print("rmse < early_stop, exiting...")
        break

    learning_rate = AdamW.param_groups[0]['lr']
    if learning_rate < config["early_stopping_learning_rate"]:
        if config["L1"] and not L1_started:
            L1_started = True
            config["early_stopping_learning_rate"] *= 0.5
            best_model_checkpoint = config['output']+"/best_L1.pt"
            Log.Log("learning_rate < early_stopping_learning_rate, switching to L1")
        else:
            Log.Log("learning_rate < early_stopping_learning_rate, exiting...")
            break

    # set a checkpoint
    if AdamW_scheduler.is_better(EF_coef, AdamW_scheduler.best):
        try:
            torch.save(nn.state_dict(), best_model_checkpoint)
        except PermissionError: # happens sometimes on windows for no good reason
            torch.save(nn.state_dict(), best_model_checkpoint)
    
    if config["L1"] and L1_started:
        LOSS = torch.nn.L1Loss(reduction='none')
    else:
        LOSS = torch.nn.MSELoss(reduction='none')

    AdamW_scheduler.step(EF_coef)
    SGD_scheduler.step(EF_coef)
    
    Train_E_rmse = 0
    Train_F_rmse = 0

    for i, properties in enumerate(training): #,desc="epoch {}".format(AdamW_scheduler.last_epoch)
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).float()
        if config["mixed"]:
            charges = properties['Charge'].to(device).float()
        if config["HasForces"]:
            true_forces = properties['forces'].to(device).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        if config["mixed"]:
            _, predicted_energies = model((species, coordinates, charges))
        else:
            _, predicted_energies = model((species, coordinates))


        # We can use torch.autograd.grad to compute force. Remember to
        # create graph so that the loss of the force can contribute to
        # the gradient of parameters, and also to retain graph so that
        # we can backward through it a second time when computing gradient
        # w.r.t. parameters.
        if config["HasForces"]:
            forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

        # Now the total loss has two parts, energy loss and force loss
        energy_loss = (LOSS(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        if config["HasForces"]:
            force_loss = (LOSS(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
            loss = (energy_loss * (1-config["fc"])) + (force_loss * config["fc"])
        else:
            loss = energy_loss
        

        Train_E_rmse += np.sqrt(energy_loss.cpu().detach().numpy())
        if config["HasForces"]:
            Train_F_rmse += np.sqrt(force_loss.cpu().detach().numpy())
        
        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()
    
    Train_E_rmse = hartree2kcalmol(Train_E_rmse/len(training))
    if config["HasForces"]:
        Train_F_rmse = hartree2kcalmol(Train_F_rmse/len(training))
    else:
        Train_F_rmse = -1
        force_rmse = -1
        val_force_rmse = -1
    training_log.loc[AdamW_scheduler.last_epoch + 1] = [Epoch, Train_E_rmse, energy_rmse, val_energy_rmse, learning_rate]
    
    if config["HasForces"]:
        Log.Log(f"Epoch: {Epoch} Train: {round(Train_E_rmse, 3)} kcal/mol Test: {round(energy_rmse, 3)} kcal/mol Test F RMSE: {round(force_rmse, 3)} kcal/mol Test EF_coef: {round(EF_coef, 3)} lr: {learning_rate}")
    else:
        #Log.Log(f"Epoch: {Epoch} Train: {round(Train_E_rmse, 3)} Test: {round(energy_rmse, 3)} Val: {round(val_energy_rmse, 3)} kcal/mol lr: {learning_rate}, Zero values: {Zero}")
        Log.Log(f"Epoch: {Epoch} Train: {round(Train_E_rmse, 3)} Test: {round(energy_rmse, 3)} Val: {round(val_energy_rmse, 3)} kcal/mol lr: {learning_rate}")

# =============================================================================
# Need to ensure everything saves nicely and then we can plot some pretty graphs
# =============================================================================

    try:
        torch.save({
            'nn': nn.state_dict(),
            'AdamW': AdamW.state_dict(),
            'SGD': SGD.state_dict(),
            'AdamW_scheduler': AdamW_scheduler.state_dict(),
            'SGD_scheduler': SGD_scheduler.state_dict(),
            "L1_started": L1_started,
        }, latest_checkpoint)
    except PermissionError: # happens sometimes on windows for no good reason
        Log.Log("Permission error in saving latest.pt, we'll just skip this one.")
    except OSError: # happens sometimes on windows for no good reason
        Log.Log("OSerror in saving latest.pt, we'll just skip this one.")
    
    training_log.to_csv(TrainingLog)
    
a="""
    if (AdamW_scheduler.last_epoch + 1) % config["GraphEveryEpoch"] == 0:
        fig, axs = plt.subplots(2)
        fig.suptitle(f"Iridium deep learning\nForce coef = {config['fc']}")
        axs[0].plot(training_log.index, training_log["Train"], lw=1.5, label="Training")
        axs[0].plot(training_log.index, training_log["Test"], lw=1.5, label="Testing")
        if config["valds"] is not None:
            axs[0].plot(training_log.index, training_log["Val"], lw=1.5, label="Validation")
        axs[0].scatter([training_log["Test"].argmin()-1], [training_log["Test"].min()], color="black", s=3)
        if config["HasForces"]:
            axs[1].plot(training_log.index, training_log["Train F RMSE"], lw=1.5, label="Training")
            axs[1].plot(training_log.index, training_log["Test F RMSE"], lw=1.5, label="Testing")
        #axs[2].plot(training_log.index, training_log["Energy MSE"], lw=1.5, label="Energy MSE")
       # axs[3].plot(training_log.index, training_log["Force MSE"], lw=1.5, label="Force MSE")

        axs[1].set_xlabel("Epoch")
        
        if training_log[["Train", "Test"]].max().max()-training_log[["Train", "Test"]].min().min() > 250:
            axs[0].set_yscale("log")    
            axs[0].set_ylabel("log E RMSE (kcal/mol)")
        else:
            axs[0].set_ylabel("E RMSE (kcal/mol)")
            
        axs[1].plot(training_log.index[-100:], training_log["Train"][-100:], lw=1.5, label="Training")
        axs[1].plot(training_log.index[-100:], training_log["Test"][-100:], lw=1.5, label="Testing")
        if config["valds"] is not None:
            axs[1].plot(training_log.index[-100:], training_log["Val"][-100:], lw=1.5, label="Validation")
        axs[1].set_ylabel("E RMSE (kcal/mol) [-100:]")
            
# =============================================================================
#         if config["HasForces"]:
#             if training_log[["Train F RMSE", "Test F RMSE"]].max().max()-training_log[["Train F RMSE", "Test F RMSE"]].min().min() > 250:
#                 axs[1].set_yscale("log")    
#                 axs[1].set_ylabel("log F RMSE (kcal/mol)")
#             else:
#                 axs[1].set_ylabel("F RMSE (kcal/mol)")
# =============================================================================
                

        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines_labels = [axs[0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)
        plt.tight_layout()
        plt.savefig(config['output']+"/DNN_training.png")
        plt.show()
        Log.Log(f"Training graph saved to: {config['output']}/DNN_training.png")
#"""



t = time.localtime()
Log.Log("Finished training in:" + str(starttime - time.time()) + "s")

# =============================================================================
# Finish the Training and save evrything to the log file
# =============================================================================

Log.close()
