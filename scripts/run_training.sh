#!/bin/bash
JL="/home/skvara/julia-1.5.3/bin/julia"
$JL train_model_morpho_mnist.jl 2 16 32 64 --seed 2 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 1
$JL train_model_morpho_mnist.jl 2 32 64 128 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 1
$JL train_model_morpho_mnist.jl 2 8 16 32 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 1
$JL train_model_morpho_mnist.jl 2 8 16 16 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 1
$JL train_model_morpho_mnist.jl 2 8 16 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 1
$JL train_model_morpho_mnist.jl 2 8 16 16 --seed 2 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 1
$JL train_model_morpho_mnist.jl 2 8 16 32 --seed 1 --digit 0 1 2 3 4 --thickness "<=" 3 --slant ">=" 0 --gpu_id 1 --lambda 0.1
