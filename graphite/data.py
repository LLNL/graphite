from torch_geometric.data import Data


class LineGraphPairData(Data):
    """Custom PyG data for representing atomic structures, with optional angular/line graph encoding.
    The following arguments assume an atomic graph of N_atm atoms, N_bnd bonds,
    and a line graph of N_ang angles (including dihedral angles).

    Args:
        edge_index_G (2 x N_atm Tensor): Edge index of the atomic graph in COO format.
        x_atm (N_atm x F_atm Tensor): Atom features.
        x_bnd (N_bnd x F_bnd Tensor): Bond features.
        edge_index_L (2 x N_ang Tensor, optional): Optional edge index of the angular/line graph in COO format.
        x_ang (N_ang x F_ang Tensor, optional): Optional angle features.
        mask_dih_ang (Tensor of bool, optional): If the angular graph encodes dihedral angles, the mask indicating
            which angles are dihedral angles must be provided.
    """
    def __init__(self,
        edge_index_G,
        x_atm,
        x_bnd,
        edge_index_L = None,
        x_ang        = None,
        mask_dih_ang = None,
    ):
        super().__init__()
        self.edge_index_G = edge_index_G
        self.edge_index_L = edge_index_L
        self.x_atm = x_atm
        self.x_bnd = x_bnd
        self.x_ang = x_ang
        self.mask_dih_ang = mask_dih_ang
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_G':
            return self.x_atm.size(0)
        if key == 'edge_index_L':
            return self.x_bnd.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
