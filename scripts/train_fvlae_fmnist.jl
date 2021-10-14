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
        default = 2
        help = "dimensionality of latent spaces"
    "hdim"
        arg_type = Int
        default = 8
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
        default = 0.1f0
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
        default = :gaussian
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
        default = 500000
        arg_type = Int
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
dataset = "fmnist"
train_x, train_y = FashionMNIST.traindata()
test_x,  test_y  = FashionMNIST.testdata()

train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); 
    seed=nothing, contamination::Real=0.0)

ratios = (0.8,0.199,0.001)
if test 
    data = HierarchicalAD.load_shapes2D()
    tr_x = HierarchicalAD.sample_tensor(data, 1000)
    val_x = HierarchicalAD.sample_tensor(data, 1000)
    tst_x = HierarchicalAD.sample_tensor(data, 100)
    a_x = HierarchicalAD.sample_tensor(data, 1000)
    tr_y, val_y, tst_y, a_y = nothing, nothing, nothing, nothing
else
    (tr_x, tr_y), (val_x, val_y), (tst_x, tst_y), (a_x, a_y) = 
        HierarchicalAD.load_train_val_test_data(dataset, filter_dict; ratios=ratios, seed=seed,
            categorical_key="shape")
end
if epochsize == nothing
    epochsize = size(tr_x, 4)
end

# now train the model
ncs = channels
ks = [(k,k) for k in kernelsizes][1:latent_count]
factors = [[:shape, :scale, :posX, :posY, :normalized_orientation], [:posX, :posY], [:shape, :posX, :posY]]

# repeat training until convergence
max_retries = 10
RETRY = true
itries = 0
while RETRY
    global model, training_history, reconstructions, latent_representations = 
        HierarchicalAD.train_fvlae(latent_dim, hdim, batchsize, ks, ncs, stride, nepochs, tr_x, 
            val_x;  tr_y=tr_y, val_y=val_y, factors=factors, 
            disentangle_per_latent = disentangle_per_latent,
            λ=lambda, γ=gamma, epochsize=epochsize, 
            layer_depth=layer_depth, lr=lr, 
            var=out_var, activation=activation, xdist=xdist, pad=pad, 
            discriminator_nlayers=discriminator_nlayers,
            initial_convergence_epochs=2, convergence_threshold=0.2);
    if training_history[:autoencoder_loss].values[end] < 400 || itries >= max_retries
        global RETRY = false
    else
        @info  "Restarting training..."
        global itries += 1
    end
end

Flux.Zygote.ignore() do
    # compute scores
    tr_scores, val_scores, tst_scores, a_scores = 
        map(x->cpu(HierarchicalAD.reconstruction_probability(gpu(model), x, 10, batchsize)), (tr_x, val_x, tst_x, a_x))

    # compute encodings
    _encode_all(m,x,batchsize) = (size(x, ndims(x)) == 0) ? Float32[] : 
        cpu(HierarchicalAD.encode_all(m,x,batchsize=batchsize))
    tr_encodings, val_encodings, tst_encodings, a_encodings = 
        map(x->_encode_all(model,x,batchsize),(tr_x, val_x, tst_x, a_x))

    # now save everything
    model_id = HierarchicalAD.timetag()
    experiment_args = (model_id=model_id, data=dataset, latent_count=latent_count, 
        latent_dim=latent_dim, channels=ncs, kernelsizes=ks, stride=stride, layer_depth=layer_depth, 
        last_conv=last_conv, lr=lr, activation=activation, seed=seed, lambda=lambda, 
        batchsize=batchsize, nepochs=nepochs, gpu_id=gpu_id, epochsize=epochsize, gamma=gamma, 
        hdim=hdim, xdist=xdist, test=test, pad=pad, discriminator_nlayers=discriminator_nlayers,
        disentangle_per_latent=disentangle_per_latent, factors=factors)
    save_args = (model_id=model_id, data=dataset, model="fvlae", latent_dim=latent_dim,
        channels=channels,gamma=gamma, lambda=lambda, activation=activation, xdist=xdist)
    svn = HierarchicalAD.safe_savename(save_args, "bson", digits=5)
    svn = joinpath(datadir("models/$savepath"), svn)
    tagsave(svn, Dict(
            :model => cpu(model),
            :experiment_args => experiment_args,
            :filter_dict => filter_dict,
            :training_history => training_history,
            :reconstructions => reconstructions,
            :latent_representations => latent_representations,
            :ratios => ratios,
            :tr_scores => tr_scores,
            :val_scores => val_scores,
            :tst_scores => tst_scores,
            :a_scores => a_scores,
            :tr_encodings => tr_encodings, 
            :val_encodings => val_encodings, 
            :tst_encodings => tst_encodings, 
            :a_encodings => a_encodings,
            :tr_labels => tr_y,
            :val_labels => val_y,
            :tst_labels => tst_y,
            :a_labels => a_y,
            :savepath => svn,
            :non_default_filters => non_default_filters        
        ))
    @info "Results saved to $svn"
end