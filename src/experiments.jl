digits_bw_labels() = CSV.read(datadir("digits_bw/labels.csv"), DataFrame)
digits_rgb_labels() = CSV.read(datadir("digits_rgb/labels.csv"), DataFrame)
morpho_mnist_labels() = CSV.read(datadir("morpho_mnist/labels.csv"), DataFrame)

"""
	filter_data(full_labels::DataFrame, filter_dict::Dict)

Filter the data given the full labels and a dict of filters.
"""
function filter_data(full_labels, filter_dict)
    ks = keys(filter_dict)
    bin_inds = ("digit" in ks) ? map(x-> x in filter_dict["digit"], full_labels[!,:digit]) : 
        Bool.(ones(size(full_labels,1)))
    for k in filter(k->k != "digit", ks)
        kp = filter_dict[k]
        ff(x) = eval(Meta.parse("$x $(kp[1]) $(kp[2])"))
        bin_inds = bin_inds .& map(ff, full_labels[!,Symbol(k)])
    end
    return bin_inds
end

function _load_mnist(type::String)
	(type in ["train", "test"]) ? nothing : error("Suggested MNIST data type $type not available.") 
    x = eval(Meta.parse("MLDatasets.MNIST.$(type)tensor(Float32)"))
    x = permutedims(x, (2,1,3))
    s = size(x)
    reshape(x, s[1], s[2], 1, s[3])
end

"""
	load_mnist(type="")

Loads either all/test/train MNIST data.
"""
function load_mnist(type="")
	if type == ""
		return cat(_load_mnist("train"), _load_mnist("test"), dims=4)
	else
		return _load_mnist(type)
	end
end


"""
    train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)

Split indices.
"""
function train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)
    (sum(ratios) ≈ 1 && length(ratios) == 3) ? nothing :
    	error("ratios must be a vector of length 3 that sums up to 1")

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # set number of samples in individual subsets
    n = length(indices)
    ns = cumsum([x for x in floor.(Int, n .* ratios)])

    # scramble indices
    _indices = sample(indices, n, replace=false)

    # restart seed
    (seed == nothing) ? nothing : Random.seed!()

    # return the sets of indices
    _indices[1:ns[1]], _indices[ns[1]+1:ns[2]], _indices[ns[2]+1:ns[3]]
end

"""
	load_train_val_test_data(dataset, data_args=Dict(); ratios=(0.6,0.2,0.2), seed=nothing)

Dataset is one of "morhpo_mnist"/"digits_bw"/"digits_rgb".
"""
function load_train_val_test_data(dataset, data_args=Dict(); ratios=(0.6,0.2,0.2), seed=nothing)
    if dataset == "morpho_mnist"
	    full_data = load_mnist()
	else
		raw_data = load(datadir("$(dataset)/digits.bson"))[:digits]
		N = length(raw_data)
		full_data = zeros(Float32, size(raw_data[1])..., N);
		for i in 1:N
		    full_data[:,:,:,i] .= raw_data[i]
		end
	end
    full_labels = CSV.read(datadir("$(dataset)/labels.csv"), DataFrame)
    filter_keys = filter(k->!(k in 
        ["latent_count", "latent_dim", "last_conv", "seed", "lambda", "batchsize", "nepochs"]),
    	keys(data_args))
    filter_dict = Dict(zip(filter_keys, [data_args[k] for k in filter_keys]))
    included_inds = filter_data(full_labels, filter_dict)

    # now split the data
    normal_data = full_data[:,:,:,included_inds]
    anomalous_data = full_data[:,:,:,.!included_inds]

    # further split the normal into train and val
    trinds, valinds, tstinds = train_val_test_inds(1:size(normal_data,4), 
        ratios; seed=seed)
    tr_x = normal_data[:,:,:,trinds]
    val_x = normal_data[:,:,:,valinds]
    tst_x = normal_data[:,:,:,tstinds]
    ax = anomalous_data
    
    tr_y = full_labels[trinds,:]
	val_y = full_labels[valinds,:]
	tst_y = full_labels[tstinds,:]
	a_y = full_labels[.!included_inds,:]

    (tr_x, tr_y), (val_x, val_y), (tst_x, tst_y), (a_x, a_y)
end