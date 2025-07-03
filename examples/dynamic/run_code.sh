#!/usr/bin/env bash

echo -n "[*] Installing ... "
pip install ../.. 1> .build.out 2> .build.err
if [ $? -ne 0 ]; then
    echo "failed."
    echo "Check .build.out and .build.err for details."
    exit 1
fi
echo "done."

python3 utils.py
