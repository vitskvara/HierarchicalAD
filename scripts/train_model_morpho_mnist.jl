using DrWatson
@quickactivate
using ArgParse
using CSV, DataFrames
using HierarchicalAD

s = ArgParseSettings()
@add_arg_table s begin
    "latent_count"
        arg_type = Int
        default = 3
        help = "number of latent spaces"
    "latent_dim"
        arg_type = Int
        default = 2
        help = "dimensionality of latent spaces"
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
    "--batchsize"
        default = 128
        arg_type = Int
        help = "batchsize"
    "--last_conv"
        action = :store_true
        help = "should the last layer of decoder be a dense or a conv layer"
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
@unpack latent_count, latent_dim, last_conv, seed, lambda, batchsize, nepochs = args
if seed != nothing
	seed = eval(Meta.parse(seed))
end

# get full labels and data
full_data = HierarchicalAD.load_mnist()
full_labels = CSV.read(datadir("morpho_mnist/labels.csv"), DataFrame)
filter_keys = filter(k->!(k in 
    ["latent_count", "latent_dim", "last_conv", "seed", "lambda", "batchsize", "nepochs"]),keys(args))
filter_dict = Dict(zip(filter_keys, [args[k] for k in filter_keys]))
included_inds = HierarchicalAD.filter_data(full_labels, filter_dict)

# now split the data
normal_data = full_data[:,:,:,included_inds]
anomalous_data = full_data[:,:,:,.!included_inds]

# further split the normal into train and val
trinds, valinds, tstinds = HierarchicalAD.train_val_test_inds(1:size(normal_data,4), 
    (0.6,0.399,0.001); seed=seed)
tr_x = normal_data[:,:,:,trinds]
val_x = normal_data[:,:,:,valinds]
tst_x = normal_data[:,:,:,tstinds]

# now train the model
ks = [(5,5), (3,3), (3,3), (1,1)][1:latent_count]
ncs = [16,32,64,128][1:latent_count]
model, training_history, reconstructions, latent_representations = 
    HierarchicalAD.train_vlae(latent_dim, batchsize, ks, ncs, 1, nepochs, tr_x, val_x, tst_x; λ=lambda)

# now save everything
experiment_args = (data="morpho-mnist", latent_count=latent_count,latent_dim=latent_dim, last_conv=last_conv, 
    seed=seed, lambda=lambda, batchsize=batchsize, nepochs=nepochs)
svn = savename(xperiment_args, "bson")
svn = joinpath(datadir("models/initial_models"), svn)
tagsave(svn, Dict(
        :model => cpu(model),
        :experiment_args => experiment_args,
        :filter_dict => filter_dict,
        :training_history => training_history,
        :reconstructions => reconstructions,
        :latent_representations => latent_representations
    ))
# TODO also add anomaly scores for tr/val/tst and for anomalous data