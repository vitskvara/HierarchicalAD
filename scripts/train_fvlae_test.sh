#!/bin/bash
SAVEPATH="test"

JL="/home/skvara/julia-1.5.3/bin/julia"
$JL train_fvlae.jl 2 8 16 32 64 --seed 1 --digit 0 1 2 3 4 --test --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 5
