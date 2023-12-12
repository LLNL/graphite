# Adopted and modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html

import torch

from rdkit import Chem

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def atom2features(atom):
    return [
        x_map['atomic_num'].index(atom.GetAtomicNum()),
        x_map['chirality'].index(str(atom.GetChiralTag())),
        # x_map['formal_charge'].index(atom.GetFormalCharge()),
        # x_map['num_hs'].index(atom.GetTotalNumHs()),
        # x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()),
        # x_map['hybridization'].index(str(atom.GetHybridization())),
        # x_map['is_aromatic'].index(atom.GetIsAromatic()),
    ]


def bond2features(bond):
    return [
        e_map['bond_type'].index(str(bond.GetBondType())),
        e_map['stereo'].index(str(bond.GetStereo())),
        e_map['is_conjugated'].index(bond.GetIsConjugated()),
    ]


def mol2graph(mol, atom2features=atom2features, bond2features=bond2features, with_hydrogen=False, kekulize=False):
    # from rdkit import Chem, RDLogger
    # RDLogger.DisableLog('rdApp.*')

    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    x = []
    for atom in mol.GetAtoms():
        x.append(atom2features(atom))

    x = torch.tensor(x, dtype=torch.long)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e = bond2features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    # if edge_index.numel() > 0:  # Sort indices.
    #     perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
    #     edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return x, edge_index, edge_attr


def features2atom(features):
    x = features
    atom = Chem.Atom(x[0])
    atom.SetChiralTag(Chem.rdchem.ChiralType.values[x[1]])
    # atom.SetFormalCharge(x_map['formal_charge'][x[2]])
    # atom.SetNumExplicitHs(x_map['num_hs'][x[3]])
    # atom.SetNumRadicalElectrons(x_map['num_radical_electrons'][x[4]])
    # atom.SetHybridization(Chem.rdchem.HybridizationType.values[x[5]])
    # atom.SetIsAromatic(x[6])
    return atom


def set_bond_features(bond, features):
    e = features
    bond.SetBondType(Chem.BondType.values[e[0]])
    
    stereo = Chem.rdchem.BondStereo.values[e[1]]
    if stereo != Chem.rdchem.BondStereo.STEREONONE:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond.SetStereoAtoms(j, i)
        bond.SetStereo(stereo)
    
    bond.SetIsConjugated(bool(e[2]))


def graph2mol(x, edge_index, edge_attr, features2atom=features2atom, set_bond_features=set_bond_features, kekulize=False):
    mol = Chem.RWMol()

    for x_i in x.tolist():
        atom = features2atom(x_i)
        mol.AddAtom(atom)

    visited = set()
    for (i, j), e in zip(edge_index.T.tolist(), edge_attr.tolist()):
        if tuple(sorted([i, j])) in visited:
            continue
        else:
            mol.AddBond(i, j)

        bond = mol.GetBondBetweenAtoms(i, j)
        set_bond_features(bond, e)

        visited.add(tuple(sorted([i, j])))

    mol = mol.GetMol()

    if kekulize:
        Chem.Kekulize(mol)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return mol