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

zdim = 2
hdim = 16
batchsize = 64
ks = ((5,5), (3,3), (3,3))
ncs = (16, 32, 32)
str = 1
nepochs = 20
λ = 0.0f0
γ = 0.1f0
epochsize = size(data,4)
layer_depth = 1
lr = 0.001f0
var = :dense
activation = "relu"
discriminator_nlayers = 3

model, hist, rdata, zs = HierarchicalAD.train_fvlae(zdim, hdim, batchsize, ks, ncs, str,
 nepochs, tr_x, val_x, tst_x; λ=λ, γ=γ, epochsize = epochsize, layer_depth=layer_depth, 
 	lr=lr, var=var, activation=activation, discriminator_nlayers=discriminator_nlayers)

