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

from . import ATOM_TYPES, BOND_TYPES, FULL_ATOM_TYPES

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="unknown class(es) ['', None] will be ignored",
)

DFT_CORRECT = [1682, 3003]

FAULTY_SMILES = [4334, 4984, 4341, 4932, 4931, 3887, 3678, 4968, 4235, 4982]
WEIRD_LOOKING = [4919, 3483]
SPECIAL_CHARGE = [2900, 2160]
CONTAIN_BR_OR_CL = [921, 2792, 2820]

REMOVE_LIST = FAULTY_SMILES + WEIRD_LOOKING + SPECIAL_CHARGE + CONTAIN_BR_OR_CL


class LeffingwellGoodscentsDataset(InMemoryDataset):
    def __init__(
        self,
        root: str = "data/leffingwell_goodscents",
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
        return ["leffingwell_goodscents_data.csv", "dft_data.sdf"]

    @property
    def num_classes(self) -> int:
        target = self[0].y
        return target.size(-1)

    def process(self):
        label_df = pd.read_csv(self.raw_paths[0])
        dft_data = Chem.SDMolSupplier(self.raw_paths[1], removeHs=False, sanitize=True)

        # process target labels
        classes = (
            label_df["odor_labels"].str.split(pat=r"\['|', '|'\]", expand=True).values
        )[:, 1:]
        unique_classes = set(classes.reshape(-1).tolist())
        unique_classes.discard("")
        unique_classes.discard(" ")
        unique_classes.discard(None)
        unique_classes = list(unique_classes)
        mlb = MultiLabelBinarizer(classes=unique_classes)
        target = mlb.fit_transform(classes)
        target = torch.tensor(target, dtype=torch.float)

        smiles_list = label_df["smiles"].values.tolist()
        assert len(target) == len(smiles_list)

        # process input molecules
        data_list = []
        for i, mol in enumerate(tqdm(dft_data)):
            # There are only 4 Cl adn 1 Br atoms in this dataset, remove corresponding molecules

            idx = int(mol.GetProp("_Name"))
            if idx in REMOVE_LIST:
                continue

            N = mol.GetNumAtoms()

            pos = dft_data.GetItemText(i).split("\n")[4 : 4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

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

            y = target[idx].unsqueeze(0)

            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
