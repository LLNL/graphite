import torch
import numpy as np
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import MinMaxScaler
from torch_geometric.loader     import DataLoader
from pathlib                    import Path
from time                       import perf_counter
from datetime                   import datetime

from graphite.nn.models         import EGCNN, ALIGNN, ALIGNN_d, ALIGNN_d_interpretable

from cmdline_args               import parse_cmdline


if __name__ == '__main__':
    # Input parameters
    args =  parse_cmdline()

    DATASET     = args.dataset
    LOG_PATH    = args.log_path
    MODEL       = args.model
    RUN_ID      = args.run_id
    CUTOFF      = args.cutoff
    NUM_LAYERS  = args.num_layers
    LEARN_RATE  = args.learn_rate
    EPOCHS      = args.epochs
    BATCH_SIZE  = args.batch_size
    CHANNELS    = args.num_channels

    if MODEL == 'EGCNN':
        model = EGCNN(dim=CHANNELS, num_interactions=NUM_LAYERS, cutoff=CUTOFF)
    if MODEL == 'ALIGNN':
        model = ALIGNN(dim=CHANNELS, num_interactions=NUM_LAYERS, cutoff=CUTOFF)
    if MODEL == 'ALIGNN_d':
        model = ALIGNN_d(dim=CHANNELS, num_interactions=NUM_LAYERS, cutoff=CUTOFF)
    if MODEL == 'ALIGNN_d_interpretable':
        model = ALIGNN_d_interpretable(dim=CHANNELS, num_interactions=NUM_LAYERS, cutoff=CUTOFF)
    print(f'Dataset: {DATASET}')
    print(f'Model:   {str(model)}')


    # Load data (X) and define `data.y`
    dataset    = torch.load(DATASET)
    scaler     = MinMaxScaler()
    y_unscaled = np.array([(data.loc, data.amp, data.sig) for data in dataset]).reshape(-1, 3)
    y_scaled   = scaler.fit_transform(y_unscaled)
    for data, y in zip(dataset, y_scaled):
        data.y  = torch.tensor(y, dtype=torch.float).view(1, -1)


    # Partition data and set up data loaders
    ds_train, ds_valid = train_test_split(dataset, train_size=0.9, random_state=12345)
    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(dataset[0], 'x_ang') else ['x_atm']
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  follow_batch=follow_batch)
    loader_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=False, follow_batch=follow_batch)
    print(f'Number of train graphs: {len(loader_train.dataset)}')
    print(f'Number of valid graphs: {len(loader_valid.dataset)}')


    # Train!
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(loader_train),
        epochs=EPOCHS,
    )
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    loss_fn   = torch.nn.MSELoss()
    print(f'Training on {device}')

    def train(loader):
        model.train()
        total_loss = 0.0
        for data in loader:
            optimizer.zero_grad()
            data    = data.to(device)
            pred, _ = model(data)
            loss    = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(loader)


    @torch.no_grad()
    def test(loader):
        model.eval()
        total_loss = 0.0
        for data in loader:
            data = data.to(device)
            pred, _     = model(data)
            loss  = loss_fn(pred, data.y)
            total_loss += loss.item()
        return total_loss / len(loader)

    L_train, L_valid, T_perf = [], [], []
    for epoch in range(EPOCHS):
        time_0 = perf_counter()
        loss_train = train(loader_train)
        time_i = perf_counter() - time_0
        loss_valid = test(loader_valid)

        L_train.append(loss_train)
        L_valid.append(loss_valid)
        T_perf.append(time_i)

        print(
            f'epoch: {epoch:>4d} | '
            f'loss_train: {loss_train:>10.5f} | '
            f'loss_valid: {loss_valid:>10.5f} | '
            f'time: {time_i:>8.4f}'
        )


    # Log stuff
    dir_name = Path(LOG_PATH)/RUN_ID
    if not os.path.exists(dir_name): os.makedirs(dir_name)

    now = datetime.now().strftime('%Y%m%d-%H%M%S')

    torch.save(model, str(dir_name/f'{now}-model.pt'))

    np.savetxt(
        str(dir_name/f'{now}-train-hist.txt'),
        np.array((L_train, L_valid, T_perf)).T,
        fmt='%.8f',
        header='mse_train mse_valid epoch_time',
    )

    print('Model and training history have been logged. Good bye.')
