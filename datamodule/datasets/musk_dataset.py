import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import InMemoryDataset, Data
import torch.nn.functional as F
from . import ATOM_TYPES, BOND_TYPES
import torch

dfdata = pd.read_csv("data/musk/raw/musk_label_data.csv")

class MuskDataset(InMemoryDataset):
    def __init__(self, root: str = "data/musk", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['musk_label_data.csv']

    @property
    def processed_file_names(self):
        return ['main.pt']


    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        # df.dropna(inplace=True)
        # Read data into huge `Data` list.
        print(df)
        molecules_list = df["Smile"].tolist()
        molecules_name_list = df["Common Name"].tolist()

        lbe = LabelEncoder()
        target_val = np.expand_dims(lbe.fit_transform(df["cOR5A2 activity"]), axis=1)
        target_val = (np.concatenate((target_val,1-target_val), axis=1))
        

        # target_val = (df["cOR5A2 activity"]).to_numpy().astype(int)

        # process input molecules
        data_list = []
        for i, mol in enumerate(tqdm(molecules_list)):
            mol = Chem.MolFromSmiles(mol)
            mol = Chem.AddHs(mol)

            N = mol.GetNumAtoms()
            type_idx = []
            atomic_number = []
            aromatic = []
            sp, sp2, sp3 = [], [], []

            # check if a special atom appears
            flag = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ATOM_TYPES.keys():
                    flag = 1
                    continue
                type_idx.append(ATOM_TYPES[atom.GetSymbol()]) # encoded number for atom symbol
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            # skip molecules that contain special atom types
            if flag == 1:
                continue
        
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(BOND_TYPES)).to(
                        torch.float
                    )

            perm = (edge_index[0] * N + edge_index[1]).argsort() # permutation
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            
            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(ATOM_TYPES)) # atom type info
            x2 = (
                        torch.tensor([atomic_number, aromatic, sp, sp2, sp3], dtype=torch.float)
                        .t()
                        .contiguous()
                    ) # further info about atomic info, aromatic, sp, sp2, sp3
            x = torch.cat([x1.to(torch.float), x2], dim=-1)
            
            name = molecules_name_list[i]
            y = torch.unsqueeze(torch.tensor(target_val[i,:]),0)
            data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,
                        name=name,
                        idx=i,
                    )

            if self.pre_filter is not None and (not self.pre_filter(data)):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])