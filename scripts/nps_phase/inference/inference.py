import torch
import numpy as np
import ase.io

from graphite           import atoms2pygdata
from graphite.nn.models import E3NN_PhaseNet_simple

from cmdline_args       import parse_cmdline


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------
    # Input parameters
    args = parse_cmdline()
    FNAME  = args.filename
    MODEL  = args.model
    CUTOFF = args.cutoff
    SCALE  = args.scale


    # ----------------------------------------------------------------------------------
    # Convert atoms to graph
    atoms = ase.io.read(FNAME)
    atoms.cell *= SCALE
    atoms.positions *= SCALE
    data = atoms2pygdata(atoms, cutoff=CUTOFF)
    print('Structure:')
    print(f'  {atoms}')
    print(f'  {data}')
    print(f'  Number of nodes:      {data.num_nodes}')
    print(f'  Number of edges:      {data.num_edges}')
    print(f'  Average node degree:  {data.num_edges / data.num_nodes:.2f}')
    print(f'  Has isolated nodes:   {data.has_isolated_nodes()}')
    print(f'  Has self-loops:       {data.has_self_loops()}')


    # ----------------------------------------------------------------------------------
    # Apply model and save
    model = torch.load(MODEL).to('cpu')
    model.eval()

    print('Model:')
    print(f'  {model}')

    print('Applying model...')
    with torch.no_grad():
        pred, _ = model(data)
        pred    = torch.nn.functional.softmax(pred, dim=1).numpy()

    atoms.info['cutoff'] = CUTOFF
    atoms.arrays['BCC']  = pred[:, 0].reshape(-1)
    atoms.arrays['FCC']  = pred[:, 1].reshape(-1)
    atoms.arrays['HCP']  = pred[:, 2].reshape(-1)
    atoms.arrays['LIQ']  = pred[:, 3].reshape(-1)

    save_fname = FNAME + '.extxyz'
    ase.io.write(save_fname, atoms, format='extxyz')
    print(f'Model output saved to {save_fname}')
