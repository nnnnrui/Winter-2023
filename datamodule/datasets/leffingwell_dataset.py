# %%
import warnings
from typing import Callable, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from . import ATOM_TYPES, BOND_TYPES

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="unknown class(es) ['', None] will be ignored",
)


class LeffingwellDataset(InMemoryDataset):
    def __init__(
        self,
        root: str = "data/leffingwell",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["graph_dataset.pt"]

    @property
    def raw_file_names(self):
        return ["leffingwell_data.csv"]

    @property
    def num_classes(self) -> int:
        target = self[0].y
        return target.size(-1)

    def process(self):
        data_df = pd.read_csv(self.raw_paths[0])

        # process target labels
        classes = (
            data_df["odor_labels_filtered"]
            .str.split(pat=r"\['|', '|'\]", expand=True)
            .values
        )[:, 1:]
        unique_classes = set(classes.reshape(-1).tolist())
        unique_classes.discard("")
        unique_classes.discard(" ")
        unique_classes.discard(None)
        unique_classes = list(unique_classes)
        mlb = MultiLabelBinarizer(classes=unique_classes)
        target = mlb.fit_transform(classes)
        target = torch.tensor(target, dtype=torch.float)

        molecules_list = data_df["smiles"].values.tolist()
        molecules_name_list = data_df["chemical_name"].values.tolist()
        assert len(target) == len(molecules_list)

        # process input molecules
        data_list = []
        for i, mol in enumerate(tqdm(molecules_list)):
            mol = Chem.MolFromSmiles(mol)
            mol = Chem.AddHs(mol)

            N = mol.GetNumAtoms()
            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            for atom in mol.GetAtoms():
                type_idx.append(ATOM_TYPES[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

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

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(ATOM_TYPES))
            x2 = (
                torch.tensor([atomic_number, aromatic, sp, sp2, sp3], dtype=torch.float)
                .t()
                .contiguous()
            )
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = molecules_name_list[i]

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
