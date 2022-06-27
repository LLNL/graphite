import numpy as np
import torch
from sklearn.model_selection    import train_test_split
from torch_geometric.loader     import DataLoader
from pathlib                    import Path
from time                       import perf_counter
from datetime                   import datetime

from graphite.nn.models         import E3NN_PhaseNet_simple, E3NN_PhaseNet_NequIP
from graphite.util              import index2mask

from args_phasenet              import parse_args


if __name__ == '__main__':
    # Read command line arguments
    args =  parse_args()

    DATASET        = args.dataset
    LOG_PATH       = args.log_path
    RUN_ID         = args.run_id
    LEARN_RATE     = args.learn_rate
    EPOCHS         = args.epochs
    BATCH_SIZE     = args.batch_size
    IRREPS_IN      = args.irreps_in
    IRREPS_HIDDEN  = args.irreps_hidden
    IRREPS_EMB     = args.irreps_emb
    IRREPS_EDGE    = args.irreps_edge
    NUM_CONVS      = args.num_convs
    NUM_SPECIES    = args.num_species
    NUM_NEIGHBORS  = args.num_neighbors
    MAX_RADIUS     = args.max_radius
    RADIAL_NEURONS = [int(i) for i in args.radial_neurons.split()]
    HEAD_NEURONS   = [int(i) for i in args.head_neurons.split()]


    # Initialize model
    model = E3NN_PhaseNet_NequIP(
        irreps_in      = IRREPS_IN,
        irreps_hidden  = IRREPS_HIDDEN,
        irreps_emb     = IRREPS_EMB,
        irreps_edge    = IRREPS_EDGE,
        num_convs      = NUM_CONVS,
        num_species    = NUM_SPECIES,
        num_neighbors  = NUM_NEIGHBORS,
        max_radius     = MAX_RADIUS,
        radial_neurons = RADIAL_NEURONS,
        head_neurons   = HEAD_NEURONS,
    )
    print('Model:')
    print(f'  {model}')


    # Prepare dataset
    ## Cu dataset
    ds_bcc, ds_fcc, ds_hcp, ds_liq, ds_omega = torch.load(DATASET)
    dataset = ds_bcc + ds_fcc + ds_hcp + ds_liq + ds_omega
    phase2label = {'bcc': 0, 'fcc': 1, 'hcp': 2, 'liq': 3, 'omega': 4}
    for data in dataset:
        data.y = torch.empty_like(data.x).fill_(phase2label[data.phase])

    ## ICE dataset
    # dataset = torch.load(DATASET)
    # print(f'{len(dataset) = }')
    # phase2label = {'1c': 0, '1h': 1, '2': 2, '3': 3 , '6': 4, '7': 5, 'sI': 6 , 'T': 7, 'w': 8}
    # for data in dataset:
    #     data.y = torch.empty_like(data.x).fill_(phase2label[data.phase])


    # Prepare data loader
    ## Assign train-validation masks
    for data in dataset:
        idx_train, idx_valid = train_test_split(range(len(data.x)), train_size=0.8, random_state=12345)
        mask_train = index2mask(idx_train, len(data.x))
        mask_valid = index2mask(idx_valid, len(data.x))
        data.mask_train = torch.tensor(mask_train)
        data.mask_valid = torch.tensor(mask_valid)

    ## Store data into PyTorch DataLoader
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, exclude_keys=['T'])

    ## Get numbers of train/valid atoms
    num_train = np.sum([data.mask_train.sum().item() for data in dataset])
    num_valid = np.sum([data.mask_valid.sum().item() for data in dataset])
    print(f'{num_train = }')
    print(f'{num_valid = }')


    # Prepare for training
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model     = model.to(device)
    loss_fn   = torch.nn.CrossEntropyLoss()
    print(f'Training on {device}')


    def train(loader):
        model.train()
        total_loss = 0.0
        for data in loader:
            optimizer.zero_grad()
            data = data.to(device)
            pred = model(data)
            loss = loss_fn(pred[data.mask_train], data.y[data.mask_train])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)


    @torch.no_grad()
    def test(loader):
        model.eval()
        total_loss = 0.0
        for data in loader:
            data = data.to(device)
            pred, _ = model(data)
            loss = loss_fn(pred[data.mask_valid], data.y[data.mask_valid])
            total_loss += loss.item()
        return total_loss / len(loader)


    # Train!
    L_train, L_valid, T_perf = [], [], []
    print('# Epoch  Loss_train  Loss_valid  Time_perf')
    for epoch in range(EPOCHS):
        time_0     = perf_counter()
        loss_train = train(loader)
        time_i     = perf_counter() - time_0
        loss_valid = test(loader)

        L_train.append(loss_train)
        L_valid.append(loss_valid)
        T_perf.append(time_i)

        print(
            f'{epoch:>7d} '
            f'{loss_train:>10.6f} '
            f'{loss_valid:>10.6f} '
            f'{time_i:>10.3f}'
        )


    # Save model and train history
    dir_name = Path(LOG_PATH)/RUN_ID
    if not os.path.exists(dir_name): os.makedirs(dir_name)

    now = datetime.now().strftime('%Y%m%d-%H%M%S')

    torch.save(model, str(dir_name/f'{now}-model.pt'))

    np.savetxt(
        str(dir_name/f'{now}-train-hist.txt'),
        np.array((L_train, L_valid, T_perf)).T,
        fmt='%.8f',
        header='L_train L_valid T_perf',
    )

    print('Model and training history have been logged. Good bye.')
