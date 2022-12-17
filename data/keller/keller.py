import pandas as pd
import numpy as np
import pubchempy as pcp

dataset = pd.read_csv("./data/keller/raw/Keller_12868_2016_287_MOESM1_ESM.csv")
dataset.columns = dataset.iloc[1,:].to_list()
dataset = dataset.drop([0,1]) # the first and the second rows
odor_labels_name = ['HOW STRONG IS THE SMELL?', 'HOW PLEASANT IS THE SMELL?',
                    'EDIBLE ', 'BAKERY ', 'SWEET ', 'FRUIT ',
                   'FISH', 'GARLIC ', 'SPICES ', 'COLD', 'SOUR ', 'BURNT ', 'ACID ',
                   'WARM ', 'MUSKY ', 'SWEATY ', 'AMMONIA/URINOUS', 'DECAYED', 'WOOD ',
                   'GRASS ', 'FLOWER ', 'CHEMICAL '] # odor labels we are interested in
dataset = dataset[["C.A.S.","CID","Odor","Odor dilution","Subject # (this study) ",
                   "Subject # (DREAM challenge) "] + odor_labels_name]

# na values occur only in numerical rating columns
dataset = dataset.fillna(0) # by the logistic of Keller's study, na should be left with 0

dataset["CID"] = dataset["CID"].astype(int)
cids = dataset["CID"].unique()

# average ratings for each odor labels per molecule across subjects

# dataset.describe()
# 4 unique values in odor dilution

average_df = []
for i in (cids):
    df = dataset.groupby(["CID"]).get_group(i)
    dilution = df["Odor dilution"].unique() 
    high_dilution = dilution[0] if dilution[0].count("0") < dilution[1].count("0") else dilution[1] # pick the higher dilution
    df = df[df["Odor dilution"]==high_dilution] # select rows of higher dilution
    df = df[odor_labels_name].astype(float)
    if len(average_df)==0:
        average_df = df.mean()
    else:
        average_df = pd.concat([average_df,df.mean()],axis=1)

average_df = (average_df.T)
average_df.set_index(cids,inplace=True)

# get SMILES for each molecules
smiles = pcp.get_properties(["canonical_smiles","IUPACName"], cids.tolist(),as_dataframe=True)
average_df = smiles.join(average_df)
average_df=average_df.rename(columns = {'HOW STRONG IS THE SMELL?':'INTENSITY', 'HOW PLEASANT IS THE SMELL?':'PLEASANTNESS'})
average_df.set_index(np.arange(len(average_df)))
# average_df.to_csv('./data/keller/raw/keller.csv') 

### average on low dilution values
average_df_new = []
for i in (cids):
    df = dataset.groupby(["CID"]).get_group(i)
    dilution = df["Odor dilution"].unique() 
    low_dilution = dilution[0] if dilution[0].count("0") > dilution[1].count("0") else dilution[1] # pick the higher dilution
    df = df[df["Odor dilution"]==low_dilution] # select rows of higher dilution
    df = df[odor_labels_name].astype(float)
    if len(average_df_new)==0:
        average_df_new = df.mean()
    else:
        average_df_new = pd.concat([average_df_new,df.mean()],axis=1)
average_df_new = (average_df_new.T)
average_df_new.set_index(cids,inplace=True)

# get SMILES for each molecules
smiles = pcp.get_properties(["canonical_smiles","IUPACName"], cids.tolist(),as_dataframe=True)
average_df_new = smiles.join(average_df_new)
average_df_new=average_df_new.rename(columns = {'HOW STRONG IS THE SMELL?':'INTENSITY', 'HOW PLEASANT IS THE SMELL?':'PLEASANTNESS'})
average_df_new.set_index(np.arange(len(average_df_new)))
average_df_new.to_csv('./data/keller/raw/keller_low.csv') 