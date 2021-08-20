JL="/home/skvara/julia-1.5.3/bin/julia"

$JL train_fvlae_shapes2D.jl 2 128 8 8 16 16 32 --savepath gamma_test --kernelsizes 5 5 3 3 1  --xdist gaussian --gpu-id 2 --nepochs 20 --seed 1 --gamma 200 --lr 0.001 --epochsize=100000
$JL train_fvlae_shapes2D.jl 2 128 8 8 16 16 32 --savepath gamma_test --kernelsizes 5 5 3 3 1 --xdist gaussian --gpu-id 2 --nepochs 20 --seed 1 --gamma 0.1 --lr 0.001 --epochsize=100000
$JL train_fvlae_shapes2D.jl 2 128 8 8 16 16 32 --savepath gamma_test --kernelsizes 5 5 3 3 1 --xdist gaussian --gpu-id 2 --nepochs 20 --seed 1 --gamma 50 --lr 0.001 --epochsize=100000
$JL train_fvlae_shapes2D.jl 2 128 8 8 16 16 32 --savepath gamma_test --kernelsizes 5 5 3 3 1 --xdist gaussian --gpu-id 2 --nepochs 20 --seed 1 --gamma 1 --lr 0.001 --epochsize=100000
$JL train_fvlae_shapes2D.jl 2 128 8 8 16 16 32 --savepath gamma_test --kernelsizes 5 5 3 3 1 --xdist gaussian --gpu-id 2 --nepochs 20 --seed 1 --gamma 100 --lr 0.001 --epochsize=100000
$JL train_fvlae_shapes2D.jl 2 128 8 8 16 16 32 --savepath gamma_test --kernelsizes 5 5 3 3 1 --xdist gaussian --gpu-id 2 --nepochs 20 --seed 1 --gamma 10 --lr 0.001 --epochsize=100000
$JL train_fvlae_shapes2D.jl 2 128 8 8 16 16 32 --savepath gamma_test --kernelsizes 5 5 3 3 1 --xdist gaussian --gpu-id 2 --nepochs 20 --seed 1 --gamma 20 --lr 0.001 --epochsize=100000