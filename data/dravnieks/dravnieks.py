import pandas as pd
import numpy as np
import pubchempy as pcp

# behavior_1.csv contains same numerical information as DravnieksGrid.csv and only differ in label name format
# dataset_g = pd.read_csv("./data/Dravnieks/raw/DravnieksGrid.csv")
# np.sum(dataset_g.to_numpy()==dataset_b.to_numpy(), axis=0)
dataset_b = pd.read_csv("./data/Dravnieks/raw/behavior_1.csv") # p.a.  
dataset_id = pd.read_csv("./data/Dravnieks/raw/identifiers.csv") # cid, conc

# drop molecules measured at low conc
low_conc = dataset_id.iloc[(dataset_id["Conc"]=="low").to_list(),:].index.to_numpy() # drop molecules measured at low conc
mixture = dataset_b.iloc[dataset_id["CID"].isnull().to_list(),:].index.to_numpy() # drop mixtures
drop_rows = np.union1d(low_conc,mixture)

dataset_b = dataset_b.drop(drop_rows)
dataset_id = dataset_id.drop(drop_rows)
dataset_b = (dataset_b.drop(columns = ["Stimulus"])).set_index(dataset_id["CID"].astype(int))

# retreiving SMILES information for each molecules
smiles = pcp.get_properties(["canonical_smiles","IUPACName"], dataset_id["CID"].astype(int).tolist(),as_dataframe=True)
assert (smiles.index.to_numpy()!=dataset_id["CID"].to_numpy()).sum()==0
dataset_b = smiles.join(dataset_b)
dataset_b.set_index(np.arange(len(dataset_b)),inplace=True)
print(dataset_b)
print("-"*35)
print(smiles)
print("-"*35)
print(dataset_id["CID"])
dataset_b = dataset_id["CID"].to_frame().join(dataset_b)

dataset_b.to_csv('./data/Dravnieks/raw/dravnieks.csv') 