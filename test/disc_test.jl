using DrWatson
@quickactivate
using HierarchicalAD
using Flux
using StatsBase
using HierarchicalAD: FVLAE, reconstruct, encode, encode_all, decode,
	basic_model_constructor, sample_tensor

dataset = "morpho_mnist"
ratios = (0.8,0.199,0.001)
(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y), (a_x, a_y) = 
	HierarchicalAD.load_train_val_test_data(dataset, Dict(); ratios=ratios)

discriminator_nlayers = 3
data = sample_tensor(tr_x, 1000);
val_x = sample_tensor(val_x, 100);

layer_depth=1
var = :dense
activation = "relu"
lr = 0.001f0
λ = 0.0f0
γ = 0.1f0
epochsize = size(data,4)
batchsize = 64

model = gpu(FVLAE(zdim, hdim, ks, ncs, str, size(data), layer_depth=layer_depth, var=var, 
	activation=activation, nlayers=discriminator_nlayers))
nl = length(model.e)

aeps = Flux.params(model.e, model.d)
cps = Flux.params(model.c)
aeopt = ADAM(lr)
copt = ADAM(lr)

aeloss(x) = factor_aeloss(model, gpu(x), γ) + λ*sum(HierarchicalAD.l2, aeps)
closs(x) = factor_closs(model, gpu(x))

faeps = deepcopy(collect(aeps))
fcps = deepcopy(collect(cps))

# first update the autoencoder	
gs = gradient(aeps) do
	aeloss(x)
end
Flux.update!(aeopt, aeps, gs)
map(x->all(x[1] .== x[2]), zip(collect(aeps), faeps)) # this should be all zeros
map(x->all(x[1] .== x[2]), zip(collect(cps), fcps)) # this should be all ones

# then the discriminator/critic
faeps = deepcopy(collect(aeps))
fcps = deepcopy(collect(cps))
gs = gradient(cps) do
	closs(x)
end
Flux.update!(copt, cps, gs)
map(x->all(x[1] .== x[2]), zip(collect(aeps), faeps)) # this should be all ones
map(x->all(x[1] .== x[2]), zip(collect(cps), fcps)) # this should be all zeros
