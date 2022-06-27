import torch
import numpy as np
from sklearn.model_selection    import train_test_split
from torch_geometric.loader     import DataLoader
from pathlib                    import Path
from time                       import perf_counter
from datetime                   import datetime

from graphite.nn.models         import EGCONV_GNN

from cmdline_args               import parse_cmdline


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------
    # Input parameters
    args =  parse_cmdline()

    DATASET     = args.dataset
    LOG_PATH    = args.log_path
    RUN_ID      = args.run_id
    CUTOFF      = args.cutoff
    NUM_LAYERS  = args.num_layers
    LEARN_RATE  = args.learn_rate
    EPOCHS      = args.epochs
    BATCH_SIZE  = args.batch_size
    CHANNELS    = args.num_channels

    model = EGCONV_GNN(dim=CHANNELS, num_interactions=NUM_LAYERS, num_species=1, cutoff=CUTOFF)
    print(f'Dataset: {DATASET}')
    print(f'Model:   {str(model)}')


    # ----------------------------------------------------------------------------------
    # Load data (X and y) and set up data loaders
    dataset    = torch.load(DATASET)

    ds_train, ds_valid = train_test_split(dataset, train_size=0.9, random_state=12345)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=False)
    print(f'Number of train graphs: {len(loader_train.dataset)}')
    print(f'Number of valid graphs: {len(loader_valid.dataset)}')


    # ----------------------------------------------------------------------------------
    # Train!
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    loss_fn   = torch.nn.BCELoss()
    print(f'Training on {device}')

    def train():
        model.train()
        for data in loader_train:
            data = data.to(device)
            pred = model(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test(loader):
        model.eval()
        total_loss = 0.0
        for data in loader:
            data = data.to(device)
            pred = model(data)
            batch_loss  = loss_fn(pred, data.y).detach().cpu().numpy()
            total_loss += batch_loss * data.num_graphs
        return total_loss / len(loader.dataset)

    L_train, L_valid, T_perf = [], [], []
    for epoch in range(EPOCHS):
        t_0 = perf_counter()
        train()
        t_i = perf_counter() - t_0; T_perf.append(t_i)
        loss_train = test(loader_train); L_train.append(loss_train)
        loss_valid = test(loader_valid); L_valid.append(loss_valid)
        print(f'{epoch:>5d} | loss_train: {loss_train:>10.5f} | loss_valid: {loss_valid:>10.5f} | perf_counter: {t_i:>8.4f}')


    # ----------------------------------------------------------------------------------
    # Log stuff
    dir_name = Path(LOG_PATH)/RUN_ID
    now      = datetime.now().strftime('%Y%m%d-%H%M%S')
    if not os.path.exists(dir_name): os.makedirs(dir_name)

    torch.save({
        'model_summary'       : model.__repr__(),
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, str(dir_name/f'{now}-model-params.pt'))

    np.savetxt(
        str(dir_name/f'{now}-train-hist.txt'),
        np.array((L_train, L_valid, T_perf)).T,
        fmt='%.8f',
        header='L_train L_valid T_perf',
    )
    print('Model parameters and training history have been logged.')
