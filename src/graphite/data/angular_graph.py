from torch_geometric.data import Data


class AngularGraphPairData(Data):
    """Custom PyG data for representing a pair of two graphs: one graph for regular atomic
    structure (atom and bonds) and the other for bond/dihedral angles.
    
    The following arguments assume an atomic graph of `N_atm` atoms with `N_bnd` bonds,
    and an angular graph of `N_ang` angles (including dihedral angles, if there's any).

    Args:
        edge_index_G (2 x N_atm Tensor): Edge index of the atomic graph in COO format.
        x_atm (N_atm x F_atm Tensor): Atom features.
        x_bnd (N_bnd x F_bnd Tensor): Bond features.
        edge_index_A (2 x N_ang Tensor): Edge index of the angular graph in COO format.
        x_ang (N_ang x F_ang Tensor): Angle features.
        mask_dih_ang (Tensor of bool, optional): If the angular graph contains dihedral
            angles, this mask indicates which angles are dihedral angles.
    """
    def __init__(self,
        edge_index_G,
        x_atm,
        x_bnd,
        edge_index_A,
        x_ang,
        mask_dih_ang = None,
    ):
        super().__init__()
        self.edge_index_G = edge_index_G
        self.edge_index_A = edge_index_A
        self.x_atm = x_atm
        self.x_bnd = x_bnd
        self.x_ang = x_ang
        self.mask_dih_ang = mask_dih_ang
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_G':
            return self.x_atm.size(0)
        if key == 'edge_index_A':
            return self.x_bnd.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)