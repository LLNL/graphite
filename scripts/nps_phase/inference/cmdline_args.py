import argparse


def parse_cmdline():
    parser = argparse.ArgumentParser(description="GNN prediction script")

    parser.add_argument(
        '-f',
        '--filename',
        required=True,
        type=str,
        help="Path to the structure file (readable by ASE).",
        metavar=''
    )

    parser.add_argument(
        '-m',
        '--model',
        required=True,
        type=str,
        help="Path to the trained model (.pt file).",
        metavar=''
    )

    parser.add_argument(
        '-c',
        '--cutoff',
        required=False,
        default=3.15,
        type=float,
        help="Cutoff radius for graph construction (default: 3.15).",
        metavar=''
    )

    parser.add_argument(
        '-s',
        '--scale',
        required=True,
        type=float,
        help="Scaling factor for the input structure.",
        metavar=''
    )

    return parser.parse_args()
