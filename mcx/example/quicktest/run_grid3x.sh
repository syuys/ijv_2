#!/bin/sh
time ../../bin/mcx -n 1e6 --gpu 2 -f grid3x_source.json -D P --seed 1 --save2pt 1 --outputtype F --outputformat jnii
