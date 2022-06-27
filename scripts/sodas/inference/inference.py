import torch
import numpy as np
import ase.io

from graphite           import atoms2pygdata
from graphite.nn.models import EGCONV_GNN

from cmdline_args       import parse_cmdline


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------
    # Input parameters
    args =  parse_cmdline()

    FNAME      = args.filename
    CHECKPOINT = args.checkpoint


    # ----------------------------------------------------------------------------------
    # Convert atoms to graph
    atoms = ase.io.read(FNAME)
    data = atoms2pygdata(atoms, cutoff=3.5, edge_dist=True)

    print('Structure:')
    print(f'  {atoms}')
    print(f'  {data}')
    print(f'  Number of nodes:      {data.num_nodes}')
    print(f'  Number of edges:      {data.num_edges}')
    print(f'  Average node degree:  {data.num_edges / data.num_nodes:.2f}')
    print(f'  Has isolated nodes:   {data.has_isolated_nodes()}')
    print(f'  Has self-loops:       {data.has_self_loops()}')
    print(f'  Is undirected:        {data.is_undirected()}')


    # ----------------------------------------------------------------------------------
    # Load saved model weights
    model = EGCONV_GNN(dim=100, num_interactions=3, num_species=1, cutoff=3.5)
    checkpoint = torch.load(CHECKPOINT)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model:')
    print(model)


    # ----------------------------------------------------------------------------------
    # Apply model and save
    print('Applying model...')
    model = model.to('cpu')
    model.eval()

    pred = model(data).detach().numpy()

    atoms.info['cutoff'] = 3.5
    atoms.arrays['orderness'] = pred.reshape(-1)
    
    save_fname = FNAME + '.extxyz'
    ase.io.write(save_fname, atoms, format='extxyz')
    print(f'Model output saved to {save_fname}')
