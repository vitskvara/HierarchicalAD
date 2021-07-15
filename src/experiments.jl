digits_bw_labels() = CSV.read(datadir("digits_bw/labels.csv"), DataFrame)
digits_rgb_labels() = CSV.read(datadir("digits_rgb/labels.csv"), DataFrame)
morpho_mnist_labels() = CSV.read(datadir("morpho_mnist/labels.csv"), DataFrame)

"""
	filter_data(full_labels::DataFrame, filter_dict::Dict)

Filter the data given the full labels and a dict of filters.

Example:
filter_dict_a = Dict(
    "shape" => [1],
    "normalized_orientation" => ["<=", 0.1],
    "scale" => ["==", 0.8],
    "posX" => ["<", 0.5],
    "posY" => ["<", 0.5],
    )
"""
function filter_data(full_labels, filter_dict; categorical_key=nothing)
    ks = keys(filter_dict)
    bin_inds = !isnothing(categorical_key)  ? 
        map(x-> x in filter_dict[categorical_key], full_labels[!,Symbol(categorical_key)]) : 
        Bool.(ones(size(full_labels,1)))
    for k in filter(k->k != categorical_key, ks)
        kp = filter_dict[k]
        #ff(x) = eval(Meta.parse("$x $(kp[1]) $(kp[2])")) # this is slow
        val = Float64(Meta.parse("$(kp[2])"))
        # this is ugly but fast
        ff(x) = if kp[1] == "=="
            x == val
        elseif kp[1] == "<="
            x <= val
        elseif kp[1] == "<"
            x < val
        elseif kp[1] == ">="
            x >= val
        elseif kp[1] == ">"
            x > val
        elseif kp[1] == "!="
            x != val
        else
            error("Unknown operator, please support one of [==, <, <=, >, >=, !=]")
        end 
        bin_inds = bin_inds .& map(ff, full_labels[!,Symbol(k)])
    end
    return bin_inds
end

hyphen_join(x,y) = "$(x)-$(y)"

"""
	safe_savename(params, args...; kwargs...)

Equivalent to DrWatson.savename but preserves arrays and tuples.
"""
function safe_savename(params, args...; kwargs...)
    outparams = Dict()
    for (k,v) in pairs(params)
        if typeof(v) <: Vector
            outparams[k] = reduce(hyphen_join,v)
        else
            outparams[k] = v
        end
    end
    return replace(savename(outparams, args...; kwargs...), " "=>"")
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
	load_train_val_test_data(dataset, filter_dict=Dict(); ratios=(0.6,0.2,0.2), seed=nothing,
        categorical_key=nothing)

Dataset is one of "morhpo_mnist"/"digits_bw"/"digits_rgb".
"""
function load_train_val_test_data(dataset, filter_dict=Dict(); ratios=(0.6,0.2,0.2), 
    seed=nothing, categorical_key=nothing)
    if dataset == "morpho_mnist"
	    full_data = load_mnist()
    elseif dataset == "shapes2D"
        full_data,_ = load_shapes2D()
	else
		raw_data = load(datadir("$(dataset)/digits.bson"))[:digits]
		N = length(raw_data)
		full_data = zeros(Float32, size(raw_data[1])..., N);
		for i in 1:N
		    full_data[:,:,:,i] .= raw_data[i]
		end
	end
    full_labels = CSV.read(datadir("$(dataset)/labels.csv"), DataFrame)
    included_inds = filter_data(full_labels, filter_dict; categorical_key=categorical_key)

    # now split the data
    normal_data = full_data[:,:,:,included_inds]
    anomalous_data = full_data[:,:,:,.!included_inds]

    # further split the normal into train and val
    trinds, valinds, tstinds = train_val_test_inds(1:size(normal_data,4), 
        ratios; seed=seed)
    tr_x = normal_data[:,:,:,trinds]
    val_x = normal_data[:,:,:,valinds]
    tst_x = normal_data[:,:,:,tstinds]
    a_x = anomalous_data
    
    tr_y = full_labels[included_inds,:][trinds,:]
	val_y = full_labels[included_inds,:][valinds,:]
	tst_y = full_labels[included_inds,:][tstinds,:]
	a_y = full_labels[.!included_inds,:]

    (tr_x, tr_y), (val_x, val_y), (tst_x, tst_y), (a_x, a_y)
end

# stuff for run scripts
function get_filter_info(experiment_argnames, args, arg_table)
    # construct the Dictionary with filters and their values
    filter_keys = filter(k->!(k in experiment_argnames),keys(args))
    filter_dict = Dict(zip(filter_keys, [args[k] for k in filter_keys]))

    # also, set which arguments are non-default
    non_default_filters = []
    for k in filter_keys
        argind = findfirst(map(f->f.dest_name == k,arg_table.args_table.fields))
        (arg_table.args_table.fields[argind].default == filter_dict[k]) ? nothing : 
            push!(non_default_filters, k)
    end

    filter_dict, non_default_filters
end