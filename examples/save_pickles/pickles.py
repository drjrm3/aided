#!/usr/bin/env python3
"""Read large number of files and pickl them to save on reading time in the future."""

from datetime import datetime
import pickle
import sys

from aided.core.edwfns import EDWfns


def main():
    """Main routine"""

    # wfns.1k.tmp has 1K .wfn files in it to read.
    """
    tic = datetime.now()
    print("[*] Reading 1K .wfn files ... ", end="")
    sys.stdout.flush()

    ed_wfns = EDWfns("../../test/wfns.1k.tmp")
    pickle_file = "wfns.1k.pickle"

    toc = datetime.now()
    print(f"Done [{(toc - tic).total_seconds():10.5f} seconds.]")
    sys.stdout.flush()

    # Save the ed_wfns object to a pickle file.
    print("[*] Saving 1K EDWfn object to pickle file ... ", end="")
    sys.stdout.flush()

    with gzip.open(pickle_file, "wb") as f:
        pickle.dump(ed_wfns, f)
    print("Done.")

    # wfns.100k.tmp has 100K .wfn files in it to read.
    tic = datetime.now()
    print("[*] Reading 100K .wfn files ... ", end="")
    sys.stdout.flush()

    ed_wfns = EDWfns("../../test/wfns.100k.tmp")
    pickle_file = "wfns.100k.pickle"

    toc = datetime.now()
    print(f"Done [{(toc - tic).total_seconds():10.5f} seconds.]")
    sys.stdout.flush()

    # Save the ed_wfns object to a pickle file.
    print("[*] Saving 1K EDWfn object to pickle file ... ", end="")
    sys.stdout.flush()

    with open(pickle_file, "wb") as f:
        pickle.dump(ed_wfns, f)
    print("Done.")
    """

    # wfns.500k.tmp has 500K .wfn files in it to read.
    tic = datetime.now()
    print("[*] Reading 500K .wfn files ... ", end="")
    sys.stdout.flush()

    ed_wfns = EDWfns("../../test/wfns.500k.tmp")
    pickle_file = "wfns.500k.pickle"

    toc = datetime.now()
    print(f"Done [{(toc - tic).total_seconds():10.5f} seconds.]")
    sys.stdout.flush()

    # Save the ed_wfns object to a pickle file.
    print("[*] Saving 1K EDWfn object to pickle file ... ", end="")
    sys.stdout.flush()

    with open(pickle_file, "wb") as f:
        pickle.dump(ed_wfns, f)
    print("Done.")


if __name__ == "__main__":
    main()
