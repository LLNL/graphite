import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="GNN training script template")

    parser.add_argument(
        '-d',
        '--dataset',
        required=True,
        type=str,
        help="Path to the dataset file (.pt file).",
        metavar=''
    )

    parser.add_argument(
        '-lp',
        '--log_path',
        required=False,
        type=str,
        default='./train_log/',
        help="Parent directory for log files (default: ./train_log/).",
        metavar=''
    )

    parser.add_argument(
        '-id',
        '--run_id',
        required=True,
        type=str,
        help="Identifier string for this training session.",
        metavar=''
    )

    parser.add_argument(
        '-lr',
        '--learn_rate',
        required=False,
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001).",
        metavar=''
    )

    parser.add_argument(
        '-E',
        '--epochs',
        required=False,
        type=int,
        default=100,
        help="Number of epochs to run (default: 1000).",
        metavar=''
    )

    parser.add_argument(
        '-B',
        '--batch_size',
        required=False,
        type=int,
        default=8,
        help="Batch size per node (default: 8).",
        metavar=''
    )

    parser.add_argument(
        '--irreps_in',
        required=False,
        type=str,
        default='1x0e',
        help="Irreps of input node features.",
        metavar=''
    )

    parser.add_argument(
        '--irreps_hidden',
        required=False,
        type=str,
        default='8x4e + 8x6e',
        help="Irreps of node features at the hidden layers.",
        metavar=''
    )

    parser.add_argument(
        '--irreps_emb',
        required=False,
        type=str,
        default='64x0e',
        help="Irreps of embedding after convolutions. Must be scalars.",
        metavar=''
    )

    parser.add_argument(
        '--irreps_edge',
        required=False,
        type=str,
        default='1x4e + 1x6e',
        help="Irreps of spherical harmonics.",
        metavar=''
    )

    parser.add_argument(
        '--num_convs',
        required=False,
        type=int,
        default=3,
        help="Number of GNN conv/interaction layers (default: 3).",
        metavar=''
    )

    parser.add_argument(
        '--num_species',
        required=False,
        type=int,
        default=1,
        help="Number of elements/species in the atomic data.",
        metavar=''
    )

    parser.add_argument(
        '-nn',
        '--num_neighbors',
        required=False,
        type=int,
        default=12,
        help="Typical/average node degree (default: 12).",
        metavar=''
    )

    parser.add_argument(
        '--max_radius',
        required=True,
        type=float,
        help="Cutoff radius for bond length encoding.",
        metavar=''
    )

    parser.add_argument(
        '--radial_neurons',
        required=False,
        type=str,
        default='16 64',
        help="Number of neurons per layers in the FC network that learns from bond distances. For first and hidden layers, not the output layer.",
        metavar=''
    )

    parser.add_argument(
        '--head_neurons',
        required=False,
        type=str,
        default='64 4',
        help="Number of neurons per layers in the FC network that projects to final output. For hidden and last layers, not the first layer.",
        metavar=''
    )

    return parser.parse_args()
