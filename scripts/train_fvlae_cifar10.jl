using DrWatson
@quickactivate
using ArgParse
using CSV, DataFrames
using HierarchicalAD
using Flux, CUDA
using MLDatasets

arg_table = ArgParseSettings(;autofix_names=true)
@add_arg_table arg_table begin
    "latent_dim"
        arg_type = Int
        default = 32
        help = "dimensionality of latent spaces"
    "hdim"
        arg_type = Int
        default = 128
        help = "width of the hidden layers in the discriminator"
    "channels"
        arg_type = Int
        nargs = '*'
        default = [16, 32, 64]
        help = "channel sizes"
    "--disentangle-per-latent"
        action = :store_true
        help = "disentangle representations per latent space instead of per latent dim"
    "--savepath"
        arg_type = String
        default = "test"
        help = "subdir of data/models"
    "--discriminator_nlayers"
        arg_type = Int
        default = 3
        help = "number of discriminator layers"
    "--kernelsizes"
        arg_type = Int
        nargs = '*'
        default = [5, 3, 3]
        help = "kernelsizes"
    "--stride"
        help = "stride length, *to be implemented*"
        arg_type = Int
        default = 1
    "--layer_depth"
        arg_type = Int
        default = 1
        help = "depth of individual Conv blocks between latent spaces, *to be implemented*"
    "--seed"
        default = nothing
        help = "data split seed"
    "--nepochs"
        default = 20
        arg_type = Int
        help = "number of epochs"
    "--gamma"
        default = 50f0
        arg_type = Float32
        help = "scaling constant of total correlation in loss"
    "--lambda"
        default = 0f0
        arg_type = Float32
        help = "L2 regularization constant"
    "--test"
        action = :store_true
        help = "test run with limited data"
    "--batchsize"
        default = 128
        arg_type = Int
        help = "batchsize"
    "--activation"
        default = "relu"
        arg_type = String
        help = "activation function"
    "--xdist"
        default = :bernoulli
        arg_type = Symbol
        help = "decoder distribution, one of gaussian/bernoulli"
    "--last_conv"
        action = :store_true
        help = "should the last layer of decoder be a dense or a conv layer"
    "--pad"
        arg_type = Int
        default = 0
        help = "padding"
    "--lr"
        help = "learning rate"
        arg_type = Float32
        default = 0.001f0
    "--gpu_id"
        help = "GPU device switch"
        arg_type = Int
        default = 0
    "--epochsize"
        help = "number of samples used in each epoch"
        default = nothing
end
args = parse_args(arg_table)
@unpack latent_dim, hdim, channels, kernelsizes, stride, layer_depth, last_conv, seed, test,
    lambda, batchsize, nepochs, gpu_id, epochsize, savepath, lr, activation, gamma, xdist,
    pad, discriminator_nlayers, disentangle_per_latent = args
latent_count = length(channels)
(latent_count <= length(kernelsizes)) ? nothing : error("number of kernels and channels does not match.")
out_var = last_conv ? :conv : :dense
if seed != nothing
    seed = eval(Meta.parse(seed))
end
CUDA.device!(gpu_id)

# get the data
dataset = "cifar10"
train_x, train_y = CIFAR10.traindata();
test_x,  test_y  = CIFAR10.testdata();
all_x = Float32.(cat(train_x, test_x, dims=4));
all_y = Float32.(vcat(train_y, test_y));
ratios = (0.8,0.199,0.001)
split_inds = HierarchicalAD.train_val_test_inds(1:size(all_x,4), ratios; seed=seed);
tr_x, val_x, tst_x = map(is->all_x[:,:,:,is], split_inds);
tr_y, val_y, tst_y = map(is->all_y[is], split_inds);
if epochsize == nothing
    epochsize = size(tr_x, 4)
end

# now train the model
ncs = channels
ks = [(k,k) for k in kernelsizes][1:latent_count]

savepath = "contaminated_training/cifar10"

# repeat training until convergence
max_retries = 10
RETRY = true
itries = 0
while RETRY
    global model, training_history, reconstructions, latent_representations = 
        HierarchicalAD.train_fvlae(latent_dim, hdim, batchsize, ks, ncs, stride, nepochs, tr_x, 
            val_x;
            disentangle_per_latent = disentangle_per_latent,
            λ=lambda, γ=gamma, epochsize=epochsize, 
            layer_depth=layer_depth, lr=lr, 
            var=out_var, activation=activation, xdist=xdist, pad=pad, 
            discriminator_nlayers=discriminator_nlayers,
            initial_convergence_epochs=nepochs, convergence_threshold=0.0,
            early_stopping=false);
    if training_history[:autoencoder_loss].values[end] < 2000 || itries >= max_retries
        global RETRY = false
    else
        @info  "Restarting training..."
        global itries += 1
    end
end

Flux.Zygote.ignore() do
    # compute scores
    tr_scores, val_scores, tst_scores = 
        map(x->cpu(HierarchicalAD.reconstruction_probability(gpu(model), x, 10, batchsize)), (tr_x, val_x, tst_x))

    # compute encodings
    #_encode_all(m,x,batchsize) = (size(x, ndims(x)) == 0) ? Float32[] : 
    #    cpu(HierarchicalAD.encode_all(m,x,batchsize=batchsize))
    #tr_encodings, val_encodings, tst_encodings, a_encodings = 
    #    map(x->_encode_all(model,x,batchsize),(tr_x, val_x, tst_x, a_x))

    # now save everything
    model_id = HierarchicalAD.timetag()
    experiment_args = (model_id=model_id, data=dataset, latent_count=latent_count, 
        latent_dim=latent_dim, channels=ncs, kernelsizes=ks, stride=stride, layer_depth=layer_depth, 
        last_conv=last_conv, lr=lr, activation=activation, seed=seed, lambda=lambda, 
        batchsize=batchsize, nepochs=nepochs, gpu_id=gpu_id, epochsize=epochsize, gamma=gamma, 
        hdim=hdim, xdist=xdist, pad=pad, discriminator_nlayers=discriminator_nlayers,
        disentangle_per_latent=disentangle_per_latent)
    save_args = (model_id=model_id, data=dataset, model="fvlae", latent_dim=latent_dim,
        channels=channels,gamma=gamma, lambda=lambda, activation=activation, xdist=xdist)
    svn = HierarchicalAD.safe_savename(save_args, "bson", digits=5)
    svn = joinpath(datadir("models/$savepath"), svn)
    tagsave(svn, Dict(
            :model => cpu(model),
            :experiment_args => experiment_args,
            :training_history => training_history,
            :reconstructions => reconstructions,
            :latent_representations => latent_representations,
            :ratios => ratios,
            :tr_scores => tr_scores,
            :val_scores => val_scores,
            :tst_scores => tst_scores,
            :tr_labels => tr_y,
            :val_labels => val_y,
            :tst_labels => tst_y,
            :savepath => svn,
        ))
    @info "Results saved to $svn"
end
