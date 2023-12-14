# DeepSolv - ANI-based-carbene-pKa-prediction

```
$$$$$$$\                                 $$$$$$\            $$\            
$$  __$$\                               $$  __$$\           $$ |           
$$ |  $$ | $$$$$$\   $$$$$$\   $$$$$$\  $$ /  \__| $$$$$$\  $$ |$$\    $$\ 
$$ |  $$ |$$  __$$\ $$  __$$\ $$  __$$\ \$$$$$$\  $$  __$$\ $$ |\$$\  $$  |
$$ |  $$ |$$$$$$$$ |$$$$$$$$ |$$ /  $$ | \____$$\ $$ /  $$ |$$ | \$$\$$  / 
$$ |  $$ |$$   ____|$$   ____|$$ |  $$ |$$\   $$ |$$ |  $$ |$$ |  \$$$  /  
$$$$$$$  |\$$$$$$$\ \$$$$$$$\ $$$$$$$  |\$$$$$$  |\$$$$$$  |$$ |   \$  /   
\_______/  \_______| \_______|$$  ____/  \______/  \______/ \__|    \_/    
                              $$ |                                         
                              $$ |                                         
                              \__|                                         


```


# Setup

```bash
git clone
https://github.com/Tuttlelab/DeepSolv
cd DeepSolv
pip install -r requirements.txt
```

```bash
python DeepSolv.py
```

This is a generalizable method based on the free energies and solvation of molecules.

This method uses the ANI2x deep learning model to optimize molecules and predict their energies as well as an auxiliary model we have trained on solvation energy ($E_{solv}$).

$$ E_{solv} = E_{aq} - E_{g} $$

Such that the total energy is calculated as:

$$ E_{tot} = E_{ANI2x} + E_{ANI-SOLVATION-MODEL} $$

This model focus on reproducing the work of Magill *et al.* (https://doi.org/10.1021/ja038973x) in the investigation of carbene pKa values.

## Auxiliary model
##### The auxiliary solvation model differs in three key components from the original ANI model:
1. It is trained the CPCM(Water) solvation energy difference between conformers in the aqueous and gaseous phase
2. Molecular self-interaction energy is not removed before training (as it is already removed by equation #)
3. There is an additional partical given to positively charged molecule (Uranium as it need only be a placeholder that represents the presence of a +1 charge).



##### Limitations of the current solvation energy prediction model:
1. In this study we have focused on the prediction of carbene pKa's and thus that is what is represented in the dataset.
2. The addition of a charged particle has proven to work well though has only been tested on relatively small molecules of a +1 charge and has been placed at the COM.


## Training data
We used the [QMSpin dataset](https://archive.materialscloud.org/record/2020.0051/v1) which contains thousands of carbene conformers as the corpus of carbene molecules on which to train model.

These carbenes and their protonated forms, where optimized as well as run as MD trajectories (in the gas phase) to yield a large number of conformers. Each of these conformers was then simulated in CPCM water in order to calculate the $E_{solv}$ values.

In order to gain additional data the original ANI1 dataset was also used as a based to generate additional $E_{solv}$ values as well as water and hydronium clusters.

The training data (in HDF5 format) can be downloaded from the University of Strathclyde's KnowledgeBase at: [LINK](http://)

**The training set does not contain any of the carbenes for which are we predicting the pKa.**

## pKa calculation
We test several different methods of calculating pKa, which fall under two categories:
1. Using reference and calculated $G(H)$ and $\Delta G^{0}_{solv}(H)$ along with the carbene thermodynamic cycle energetic values
2. Using linear machine learning models to fit the carbene thermodynamic cycle energetic values

### Using the full thermodynamic cycle
```math
\Delta G^{0}_{solv}(H) = G(H_{3}O_{aq}) - ( G(H_{2}O_{aq}) + G^{0}(H))
```

Using the reference $G(H)$ and $\Delta G^{0}_{solv}(H)$ used by Magill *et al.* yielded very large systematic errors and a substancial RMSE values once corrected (~8.3 pKa units) for the systematic bias (MAE = 124.4 pKa units).

Using ANI model consistent values for $G(H)$ and $\Delta G^{0}_{solv}(H)$ (ie. using equation X to calculate these values within the ANI + Solvation model) we get similar errors.


### Using linear regression models to fit pKa based on free energies of solvation

This works well because the $G(H)$ and $\Delta G^{0}_{solv}(H)$ constants can be fit *via* embedding in the models intercept and coeficients which removes systematic error.

Ultimatley it doesnt matter what values are used for $G(H)$ and $\Delta G^{0}_{solv}(H)$ as long as they fit into and are consistent the model's "world view".

<p style="text-align: center;">![](Workflow.png)
</p>




