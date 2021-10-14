JL="/home/skvara/julia-1.5.3/bin/julia"

# even larger gamma
$JL --project train_fvlae_cifar10.jl 32 64 8 8 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=40 --gamma=50 \
	--xdist=bernoulli --disentangle-per-latent --last-conv
# even kernelsizes
$JL --project train_fvlae_cifar10.jl 4 64 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 4 4 4 --stride 2 \
	--seed=1 --nepochs=40 --gamma=50 \
	--xdist=bernoulli --disentangle-per-latent --last-conv
# general run
$JL --project train_fvlae_cifar10.jl 4 64 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 5 3 3 \
	--seed=1 --nepochs=40 --gamma=1 \
	--xdist=bernoulli --disentangle-per-latent --last-conv
# larger gamma
$JL --project train_fvlae_cifar10.jl 4 64 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 5 3 3 \
	--seed=1 --nepochs=40 --gamma=10 \
	--xdist=bernoulli --disentangle-per-latent --last-conv
# larger latent
$JL --project train_fvlae_cifar10.jl 16 64 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 5 3 3 \
	--seed=1 --nepochs=40 --gamma=50 \
	--xdist=bernoulli --disentangle-per-latent --last-conv
# more latents
$JL --project train_fvlae_cifar10.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=40 --gamma=50 \
	--xdist=bernoulli --disentangle-per-latent --last-conv
# more latents, smaller gamma
$JL --project train_fvlae_cifar10.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=40 --gamma=10 \
	--xdist=bernoulli --disentangle-per-latent --last-conv

# now the same but with gaussian decoder
# general run
$JL --project train_fvlae_cifar10.jl 4 64 16 32 64 --savepath=contaminated_training/cifar10 \
	--seed=1 --nepochs=40 --gamma=1 \
	--xdist=gaussian --disentangle-per-latent
# larger gamma
$JL --project train_fvlae_cifar10.jl 4 64 16 32 64 --savepath=contaminated_training/cifar10 \
	--seed=1 --nepochs=40 --gamma=10 \
	--xdist=gaussian --disentangle-per-latent
# even larger gamma
$JL --project train_fvlae_cifar10.jl 4 64 16 32 64 --savepath=contaminated_training/cifar10 \
	--seed=1 --nepochs=40 --gamma=50 \
	--xdist=gaussian --disentangle-per-latent
# larger latent
$JL --project train_fvlae_cifar10.jl 16 64 16 32 64 --savepath=contaminated_training/cifar10 \
	--seed=1 --nepochs=40 --gamma=50 \
	--xdist=gaussian --disentangle-per-latent
# more latents
$JL --project train_fvlae_cifar10.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=40 --gamma=50 \
	--xdist=gaussian --disentangle-per-latent
# more latents, smaller gamma
$JL --project train_fvlae_cifar10.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=40 --gamma=10 \
	--xdist=gaussian --disentangle-per-latent
# more latents, disentanglement per latent
$JL --project train_fvlae_cifar10.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/cifar10 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=40 --gamma=50 \
	--xdist=gaussian --disentangle-per-latent
