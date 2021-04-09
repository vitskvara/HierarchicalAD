using DrWatson
@quickactivate
using ArgParse
using CSV, DataFrames
using HierarchicalAD
using Flux, CUDA

s = ArgParseSettings()
@add_arg_table s begin
    "latent_dim"
        arg_type = Int
        default = 2
        help = "dimensionality of latent spaces"
    "channels"
        arg_type = Int
        nargs = '*'
        default = [16, 32, 64]
        help = "channel sizes"
    "--savepath"
        arg_type = String
        default = "test"
        help = "subdir of data/models"
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
    "--digit"
        help = "which digits to include"
        arg_type = Int
        nargs = '*'
        default = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    "--area"
        help = "define the included area which is a number in [0,350] with mean of ~100, e.g. `--area >= 150`"
        arg_type = String
        nargs = 2
        default = [">=", "0"]
   "--length"
        help = "define the included digit length which is a number in [0,100] with mean of ~40"
        arg_type = String
        nargs = 2
        default = [">=", "0"]
   "--thickness"
        help = "define the included thickness which is a number in [0,10] with mean of ~3"
        arg_type = String
        nargs = 2
        default = [">=", "0"]
   "--slant"
        help = "define the included slant which is a number in [-1,1] with mean of ~0.1"
        arg_type = String
        nargs = 2
        default = [">=", "-1"]
   "--width"
        help = "define the included width which is a number in [0,25] with mean of ~14"
        arg_type = String
        nargs = 2
        default = [">=", "0"]
   "--height"
        help = "define the included height which is a number in [0,21] with mean of ~19.5"
        arg_type = String
        nargs = 2
        default = [">=", "0"]
end
args = parse_args(s)
@unpack latent_dim, channels, kernelsizes, stride, layer_depth, last_conv, seed, lambda, batchsize, nepochs, 
    gpu_id, epochsize, savepath, lr, activation, xdist, test = args
latent_count = length(channels)
(latent_count <= length(kernelsizes)) ? nothing : error("number of kernels and channels does not match.")
out_var = last_conv ? :conv : :dense
if seed != nothing
	seed = eval(Meta.parse(seed))
end
CUDA.device!(gpu_id)

# get filters
filter_keys = filter(k->!(k in 
    ["latent_dim", "channels", "kernelsizes", "stride", "layer_depth", "last_conv", "seed", 
    "lambda", "batchsize", "nepochs", "gpu_id", "epochsize", "savepath", "lr", "activation", 
    "xdist", "test"]),keys(args))
filter_dict = Dict(zip(filter_keys, [args[k] for k in filter_keys]))

# also, set which arguments are non-default
non_default_filters = []
for k in filter_keys
    argind = findfirst(map(f->f.dest_name == k,s.args_table.fields))
    (s.args_table.fields[argind].default == filter_dict[k]) ? nothing : push!(non_default_filters, k)
end

# get the data
dataset = "morpho_mnist"
ratios = (0.8,0.199,0.001)
if test 
    data = HierarchicalAD.load_mnist("train")
    tr_x = HierarchicalAD.sample_tensor(data, 1000)
    val_x = HierarchicalAD.sample_tensor(data, 1000)
    tst_x = HierarchicalAD.sample_tensor(data, 100)
    tr_y, val_y, tst_y = nothing, nothing, nothing
    a_x, a_y = HierarchicalAD.sample_tensor(HierarchicalAD.load_mnist("test"), 1000), nothing
else
    (tr_x, tr_y), (val_x, val_y), (tst_x, tst_y), (a_x, a_y) = 
        HierarchicalAD.load_train_val_test_data(dataset, filter_dict; ratios=ratios, seed=seed,
            categorical_key="digit")
end
if epochsize == nothing
    epochsize = size(tr_x, 4)
end

# now train the model
ncs = channels
ks = [(k,k) for k in kernelsizes][1:latent_count]
model, training_history, reconstructions, latent_representations = 
    HierarchicalAD.train_vlae(latent_dim, batchsize, ks, ncs, stride, nepochs, tr_x, val_x, tst_x; 
        λ=lambda, epochsize=epochsize, layer_depth=layer_depth, lr=lr, var=out_var, 
        activation=activation, xdist=xdist)

# compute scores
tr_scores, val_scores, tst_scores, a_scores = 
    map(x->HierarchicalAD.reconstruction_probability(gpu(model), x, 5, batchsize), (tr_x, val_x, tst_x, a_x))

# compute encodings
tr_encodings, val_encodings, tst_encodings, a_encodings = 
    map(x->HierarchicalAD.encode_all(model,x,batchsize),(tr_x, val_x, tst_x, a_x))

# now save everything
model_id = HierarchicalAD.timetag()
experiment_args = (model_id=model_id, data=dataset, latent_count=latent_count, latent_dim=latent_dim, 
    channels=ncs, kernelsizes=ks, stride=stride, layer_depth=layer_depth, last_conv=last_conv, lr=lr, 
    activation=activation, seed=seed, lambda=lambda, batchsize=batchsize, nepochs=nepochs, gpu_id=gpu_id, 
    epochsize=epochsize, xdist=xdist, test=test)
save_args = (model_id=model_id, data=dataset, model="vlae", latent_dim=latent_dim,
    channels=channels, lambda=lambda, activation=activation, xdist=xdist)
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
