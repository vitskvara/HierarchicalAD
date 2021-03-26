#!/bin/bash
SAVEPATH=$1

JL="/home/skvara/julia-1.5.3/bin/julia"
$JL train_fvlae.jl 2 8 16 32 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 30
$JL train_fvlae.jl 2 8 16 32 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 60
$JL train_fvlae.jl 2 8 16 32 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 100
$JL train_fvlae.jl 2 512 16 32 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 30
$JL train_fvlae.jl 2 512 16 32 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 60
$JL train_fvlae.jl 2 512 16 32 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 100
$JL train_fvlae.jl 2 8 32 64 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 30
$JL train_fvlae.jl 2 512 32 64 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 60
$JL train_fvlae.jl 2 1024 32 64 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 100
$JL train_fvlae.jl 2 8 32 64 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 30 --gamma 1
$JL train_fvlae.jl 2 512 32 64 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 60 --gamma 1
$JL train_fvlae.jl 2 1024 32 64 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 100 --gamma 1
$JL train_fvlae.jl 2 8 32 64 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 30 --gamma 0.01
$JL train_fvlae.jl 2 512 32 64 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 60 --gamma 0.01
$JL train_fvlae.jl 2 1024 32 64 64 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 0 --savepath $SAVEPATH --nepochs 100 --gamma 0.01
