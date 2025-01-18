#!/usr/bin/env bash

set -e

# Change to the directory containing this script
cd $(dirname ${0})
pwd

echo -n "[TEST] Running test with one WFN file ... "
pushd ../src > /dev/null
python3 -m aided.fio.read_wfn -i ../test/data/wfns/formamide/formamide.6311gss.b3lyp.wfn
popd > /dev/null
echo "Done."

echo -n "[TEST] Running test with multiple WFN file ... "
pushd ../src > /dev/null
python3 -m aided.fio.read_wfn -i \
  ../test/data/wfns/formamide/form000001.wfn \
  ../test/data/wfns/formamide/form000002.wfn \
  ../test/data/wfns/formamide/form000003.wfn \
  ../test/data/wfns/formamide/form000004.wfn \
  ../test/data/wfns/formamide/form000005.wfn
popd > /dev/null
echo "Done."
