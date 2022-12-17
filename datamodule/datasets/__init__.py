from rdkit.Chem.rdchem import BondType as BT

ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "F": 5}
FULL_ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "F": 5, "Cl": 6, "Br": 7}
BOND_TYPES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
