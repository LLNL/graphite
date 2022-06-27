import argparse


def parse_cmdline():
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
        default='./log/',
        help="Parent directory for log files (default: ./log/).",
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
        '-c',
        '--cutoff',
        required=False,
        type=float,
        default=3.5,
        help="Cutoff radius for bond length encoding (default: 3.5).",
        metavar=''
    )

    parser.add_argument(
        '-L',
        '--num_layers',
        required=False,
        type=int,
        default=3,
        help="Number of GNN conv/interaction layers (default: 3).",
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
        default=150,
        help="Number of epochs to run (default: 150).",
        metavar=''
    )

    parser.add_argument(
        '-B',
        '--batch_size',
        required=False,
        type=int,
        default=16,
        help="Batch size per node (default: 16).",
        metavar=''
    )

    parser.add_argument(
        '-C',
        '--num_channels',
        required=False,
        type=int,
        default=100,
        help="Number of hidden channels (default: 100).",
        metavar=''
    )

    return parser.parse_args()
