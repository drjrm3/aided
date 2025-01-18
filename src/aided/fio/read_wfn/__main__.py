"""
This script reads a wfn file and prints the data structure.
"""
import argparse
import sys

from .read_wfn import read_wfn_file, read_wfn_files

# Get one or more input files.
parser = argparse.ArgumentParser(description="Test wfn reading.")
parser.add_argument("-i", "--input", type=str, nargs="+", help="Input wfn file(s)")
args = parser.parse_args()

if not args.input:
    parser.print_help()
    sys.exit(1)

if len(args.input) > 1:
    wfn_rep = read_wfn_files(args.input)
    print(f"{wfn_rep=}")
else:
    wfns_rep = read_wfn_file(args.input[0])
    print(f"{wfns_rep=}")
