import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import *
from mordred import descriptors,is_missing
# import matplotlib.pyplot as plt
from tqdm import tqdm

dravpath = "data/dravnieks/raw/"
kellerpath = "data/keller/raw/"
datadrav = pd.read_csv(dravpath+"dravnieks.csv")
datadrav.dropna(inplace=True)
datakeller = pd.read_csv(kellerpath+"keller.csv")
datakeller.columns


# using mordred to compute molecular descriptors
dataset = [datadrav, datakeller]
df_descriptor = [[], []]
path = [dravpath+"molecular_descriptors.csv", kellerpath+"molecular_descriptors.csv"]
for i in np.arange(len(dataset)):
    df=dataset[i]
    data_descriptor = df_descriptor[i]
    Smiles = df["CanonicalSMILES"].to_numpy()
    calc = Calculator(descriptors)
    for smile in tqdm(Smiles):
        mol = Chem.MolFromSmiles(smile)
        calc_result = calc(mol)
        dict_result = calc_result.asdict()
        data_descriptor.append(dict_result)
    mol = pd.DataFrame(df_descriptor[i])
    mol.insert(0,"CanonicalSMILES",Smiles)
    mol.insert(0,"CID",df["CID"].to_numpy().astype(int))
    mol.to_csv(path[i])


mol_des = [pd.read_csv(path[0]), pd.read_csv(path[1])]
for i in np.arange(len(dataset)):
    unknownid = (mol_des[i]["Vabc"]!="unknown atom type (Vabc)")
    dataset[i] = dataset[i][unknownid]
    mol_des[i] = mol_des[i][unknownid]
    print(mol_des[i].shape)
    print(dataset[i].shape)


dataset[0].to_csv(dravpath+"processed_dravnieks.csv")
dataset[1].to_csv(kellerpath+"processed_keller.csv")


drop_cols = ["missing 3D","float division","min", "max","module network"]
labeltodrop = [[],[]]
for i in np.arange(len(dataset)):
    label = labeltodrop[i]
    mol = mol_des[i]
    for descriptor in mol.columns:
        for x in mol[descriptor].unique():
            if any(substr in str(x) for substr in drop_cols):
                label.append(descriptor)
                continue

label_drop = set(labeltodrop[0]+labeltodrop[1])
print(len(label_drop))
mol_des[0].drop(columns = label_drop, inplace=True)
mol_des[1].drop(columns = label_drop, inplace=True)


mol_des[0].replace(to_replace = "^invalid value", value=np.nan,inplace=True, regex=True)
mol_des[1].replace(to_replace = "^invalid value", value=np.nan,inplace=True, regex=True)


mol_des[0].to_csv(dravpath+"processed_descriptors.csv")
mol_des[1].to_csv(kellerpath+"processed_descriptors.csv")

