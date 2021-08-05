JL="/home/skvara/julia-1.5.3/bin/julia"

$JL train_fvlae_shapes2D_light.jl 8 128 32 32 64 --savepath full_train --kernelsizes 4 4 4 --xdist gaussian --gpu_id 0 --nepochs 10 --seed 1 --gamma 50 --lr 0.0001
$JL train_fvlae_shapes2D_light.jl 8 128 32 32 64 64 --savepath full_train --kernelsizes 4 4 4 4 --xdist gaussian --gpu_id 0 --nepochs 10 --seed 1 --gamma 50 --lr 0.001

$JL train_fvlae_shapes2D_light.jl 2 128 32 32 64 64 --savepath full_train --kernelsizes 4 4 4 4 --stride 2 --xdist gaussian --gpu_id 0 --nepochs 20 --seed 1 --gamma 50 --lr 0.001 --pad 1
$JL train_fvlae_shapes2D_light.jl 2 128 32 32 64 --savepath full_train --kernelsizes 4 4 4 --stride 2 --xdist gaussian --gpu_id 0 --nepochs 20 --seed 1 --gamma 50 --lr 0.0001 --pad 1
$JL train_fvlae_shapes2D_light.jl 8 128 32 32 64 64 --savepath full_train --kernelsizes 4 4 4 4 --stride 2 --xdist gaussian --gpu_id 0 --nepochs 20 --seed 1 --gamma 50 --lr 0.001 --pad 1
$JL train_fvlae_shapes2D_light.jl 8 128 32 32 64 --savepath full_train --kernelsizes 4 4 4 --stride 2 --xdist gaussian --gpu_id 0 --nepochs 20 --seed 1 --gamma 50 --lr 0.0001 --pad 1

$JL train_fvlae_shapes2D_light.jl 2 128 32 32 64 64 --savepath full_train --kernelsizes 4 4 4 4 --stride 2 --xdist bernoulli --gpu_id 0 --nepochs 20 --seed 1 --gamma 50 --lr 0.001 --pad 1
$JL train_fvlae_shapes2D_light.jl 2 128 32 32 64 --savepath full_train --kernelsizes 4 4 4 --stride 2 --xdist bernoulli --gpu_id 0 --nepochs 20 --seed 1 --gamma 50 --lr 0.0001 --pad 1
$JL train_fvlae_shapes2D_light.jl 8 128 32 32 64 64 --savepath full_train --kernelsizes 4 4 4 4 --stride 2 --xdist bernoulli --gpu_id 0 --nepochs 20 --seed 1 --gamma 50 --lr 0.001 --pad 1
$JL train_fvlae_shapes2D_light.jl 8 128 32 32 64 --savepath full_train --kernelsizes 4 4 4 --stride 2 --xdist bernoulli --gpu_id 0 --nepochs 20 --seed 1 --gamma 50 --lr 0.0001 --pad 1

