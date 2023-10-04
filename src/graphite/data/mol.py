import torch
from torch_geometric.data import Data

from graphite.nn import bond_angles, dihedral_angles


class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in {'edge_index', 'bnd_index', 'aux_bnd_index'}:
            return self.x_atm.size(0)
        if key in {'ang_index', 'bnd_ang_index', 'dih_ang_index'}:
            return self.x_bnd.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def bond_features(self, bond_index_name='bnd_index'):
        """Does not account for periodic boundaries"""
        i, j = self.__getattr__(bond_index_name)
        bond_vec = self.pos[j] - self.pos[i]
        bond_len = bond_vec.norm(dim=-1, keepdim=True)
        return torch.hstack([bond_vec, bond_len])

    def bond_angle_features(self, bond_feature_name='x_bnd', bond_angle_index_name='bnd_ang_index'):
        x_bnd = self.__getattr__(bond_feature_name)
        bnd_ang_index = self.__getattr__(bond_angle_index_name)
        return bond_angles(x_bnd[:, :3], bnd_ang_index)
    
    def dihedral_angle_features(self, bond_index_name='bnd_index', dihehdral_angle_index_name='dih_ang_index'):
        """Does not account for periodict boundaries"""
        bnd_index = self.__getattr__(bond_index_name)
        dih_ang_index = self.__getattr__(dihehdral_angle_index_name)
        return dihedral_angles(self.pos, bnd_index, dih_ang_index)
    
    def concat_features_with_onehot(self, x1_name='x_bnd', x2_name='x_aux_bnd'):
        x1 = self.__getattr__(x1_name)
        x2 = self.__getattr__(x2_name)
        mask = torch.cat([
            torch.zeros(x1.size(0), dtype=torch.long, device=self.pos.device),
            torch.ones(x2.size(0),  dtype=torch.long, device=self.pos.device),
        ])
        onehot = torch.nn.functional.one_hot(mask)
        return torch.hstack([torch.vstack([x1, x2]), onehot])
    
    def get_bnd_ang_vals(self):
        cos_ang, _ = self.x_bnd_ang[:, :2].T
        return cos_ang.arccos().rad2deg()
    
    def get_dih_ang_vals(self):
        cos_ang, sin_ang = self.x_dih_ang[:, :2].T
        return torch.atan2(sin_ang, cos_ang).rad2deg()