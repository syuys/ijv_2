#!/bin/sh

# first run CW baseline simulation
../../bin/mcx -f baseline.json --seed -1 -q 1 --bc aaaaaa -D P "$@"

# then run RF replay to get RF Jacobian, output data has 6 dimensions
../../bin/mcx -f rfreplay.json -E baseline.mch --bc aaaaaa --savedetflag DPXVW -F mc2 -D P "$@"
