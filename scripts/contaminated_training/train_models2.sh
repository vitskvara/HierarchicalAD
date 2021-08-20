JL="/home/skvara/julia-1.5.3/bin/julia"

# even larger gamma
#$JL --project ../train_fvlae_shapes2D.jl 4 64 16 32 64 --savepath=contaminated_training/run2 \
#	--kernelsizes 5 3 3 \
#	--seed=1 --nepochs=20 --gamma=50 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
#	--xdist=bernoulli --nepochs=20
# more latents, disentanglement per latent
$JL --project ../train_fvlae_shapes2D.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=20 --gamma=50 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=bernoulli --nepochs=20 --disentangle-per-latent
# even kernelsizes
$JL --project ../train_fvlae_shapes2D.jl 4 64 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 4 4 4 --stride 2 \
	--seed=1 --nepochs=20 --gamma=50 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=bernoulli --nepochs=20
# general run
$JL --project ../train_fvlae_shapes2D.jl 4 64 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 5 3 3 \
	--seed=1 --nepochs=20 --gamma=1 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=bernoulli --nepochs=20
# larger gamma
$JL --project ../train_fvlae_shapes2D.jl 4 64 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 5 3 3 \
	--seed=1 --nepochs=20 --gamma=10 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=bernoulli --nepochs=20
# larger latent
$JL --project ../train_fvlae_shapes2D.jl 16 64 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 5 3 3 \
	--seed=1 --nepochs=20 --gamma=50 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=bernoulli --nepochs=20
# more latents
$JL --project ../train_fvlae_shapes2D.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=20 --gamma=50 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=bernoulli --nepochs=20
# more latents, smaller gamma
$JL --project ../train_fvlae_shapes2D.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=20 --gamma=10 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=bernoulli --nepochs=20

# now the same but with gaussian decoder
# general run
$JL --project ../train_fvlae_shapes2D.jl 4 64 16 32 64 --savepath=contaminated_training/run2 \
	--seed=1 --nepochs=20 --gamma=1 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=gaussian --nepochs=20
# larger gamma
$JL --project ../train_fvlae_shapes2D.jl 4 64 16 32 64 --savepath=contaminated_training/run2 \
	--seed=1 --nepochs=20 --gamma=10 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=gaussian --nepochs=20
# even larger gamma
$JL --project ../train_fvlae_shapes2D.jl 4 64 16 32 64 --savepath=contaminated_training/run2 \
	--seed=1 --nepochs=20 --gamma=50 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=gaussian --nepochs=20
# larger latent
$JL --project ../train_fvlae_shapes2D.jl 16 64 16 32 64 --savepath=contaminated_training/run2 \
	--seed=1 --nepochs=20 --gamma=50 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=gaussian --nepochs=20
# more latents
$JL --project ../train_fvlae_shapes2D.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=20 --gamma=50 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=gaussian --nepochs=20
# more latents, smaller gamma
$JL --project ../train_fvlae_shapes2D.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=20 --gamma=10 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=gaussian --nepochs=20
# more latents, disentanglement per latent
$JL --project ../train_fvlae_shapes2D.jl 4 64 8 8 16 32 64 --savepath=contaminated_training/run2 \
	--kernelsizes 5 5 3 3 1 \
	--seed=1 --nepochs=20 --gamma=50 --shape 1 2 3 --scale ">=" 0.5 --normalized-orientation ">=" 0 \
	--xdist=gaussian --nepochs=20 --disentangle-per-latent
