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
        '-c',
        '--checkpoint',
        required=True,
        type=str,
        help="Model checkpoint (.pt file)",
        metavar=''
    )

    return parser.parse_args()
