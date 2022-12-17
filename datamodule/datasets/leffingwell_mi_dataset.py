import os
import warnings
from typing import Callable, List, Optional

import numpy as np
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


class LeffingwellMiDataset(InMemoryDataset):
    def __init__(
        self,
        root: str = "data/leffingwell_mi",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        max_conformer: int = 512,
    ):
        self.max_conformers = max_conformer
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["graph_dataset.pt"]

    @property
    def raw_file_names(self):
        return ["leffingwell_data.csv", "leffingwell_conformers"]

    @property
    def num_classes(self) -> int:
        target = self[0].y
        return target.size(-1)

    def process(self):
        label_df = pd.read_csv(self.raw_paths[0])
        sdf_dir = os.listdir(self.raw_paths[1])

        # process target labels
        classes = (
            label_df["odor_labels_filtered"]
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

        smiles_list = label_df["smiles"].values.tolist()
        assert len(target) == len(smiles_list)

        # process input molecules
        data_list = []
        for sdf_file in tqdm(sdf_dir):

            idx = int(sdf_file.lstrip("LG").rstrip(".sdf"))
            sdf_path = os.path.join(self.raw_paths[1], sdf_file)
            conformers = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
            mol = conformers[0]
            smiles = mol.GetProp("_SMILES")
            assert smiles == smiles_list[idx]

            N = mol.GetNumAtoms()
            con_pos_padded = torch.zeros(N, self.max_conformers, 3)
            con_pos_mask = torch.ones(N, self.max_conformers, dtype=torch.bool)
            # True: fake padded conformer
            # False: real conformer

            con_pos = []
            for conformer in conformers:
                con_pos.append(conformer.GetConformer().GetPositions())
            con_pos = torch.tensor(np.stack(con_pos), dtype=torch.float).permute(
                1, 0, 2
            )
            # shape: (n_atom, n_conformer, 3)
            con_pos_padded[:, : len(conformers)] = con_pos
            con_pos_mask[:, : len(conformers)] = False
            con_pos_padded = con_pos_padded.reshape(
                N, self.max_conformers * 3
            ).contiguous()
            # shape: (n_atom, max_conformer x 3)

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
                con_pos=con_pos_padded,
                con_mask=con_pos_mask,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                smiles=smiles,
                idx=idx,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
