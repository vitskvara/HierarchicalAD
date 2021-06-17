"""
	load_uci_data(dataset::String)

Load a single UCI dataset.
"""
function load_uci_data(dataset::String)
	# I have opted for the original Loda datasets, use of multiclass problems in all vs one case
	# does not necessarily represent a good anomaly detection scenario
	data, _, _ = UCI.get_loda_data(dataset)
	# return only easy and medium anomalies
	UCI.normalize(data.normal, hcat(data.easy, data.medium)) # data (standardized)
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
	train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); seed=nothing,
	    contamination::Real=0.0)
Split data.
"""
function train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); 
	seed=nothing, contamination::Real=0.0)

	# split the normal data, add some anomalies to the train set and divide
	# the rest between validation and test
	(0 <= contamination <= 1) ? nothing : error("contamination must be in the interval [0,1]")
	nd = ndims(data_normal) # differentiate between 2D tabular and 4D image data

	# split normal indices
	indices = 1:size(data_normal, nd)
	split_inds = train_val_test_inds(indices, ratios; seed=seed)

	# select anomalous indices
	na = size(data_anomalous, nd)
	indices_anomalous = 1:na
	na_tr = floor(Int, length(split_inds[1])*contamination/(1-contamination))
	(na_tr > na) ? error("selected contamination rate $contamination is too high, not enough anomalies available") : nothing
	tr = na_tr/length(indices_anomalous) # training ratio
	vtr = (1 - tr)/2 # validation/test ratio
	split_inds_anomalous = train_val_test_inds(indices_anomalous, (tr, vtr, vtr); seed=seed)

	# this can be done universally - how?
	if nd == 2
		tr_n, val_n, tst_n = map(is -> data_normal[:,is], split_inds)
		tr_a, val_a, tst_a = map(is -> data_anomalous[:,is], split_inds_anomalous)
	elseif nd == 4
		tr_n, val_n, tst_n = map(is -> data_normal[:,:,:,is], split_inds)
		tr_a, val_a, tst_a = map(is -> data_anomalous[:,:,:,is], split_inds_anomalous)
	end

	# cat it together
	tr_x = cat(tr_n, tr_a, dims = nd)
	val_x = cat(val_n, val_a, dims = nd)
	tst_x = cat(tst_n, tst_a, dims = nd)

	# now create labels
	tr_y = vcat(zeros(Float32, size(tr_n, nd)), ones(Float32, size(tr_a,nd)))
	val_y = vcat(zeros(Float32, size(val_n, nd)), ones(Float32, size(val_a,nd)))
	tst_y = vcat(zeros(Float32, size(tst_n, nd)), ones(Float32, size(tst_a,nd)))

	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y)
end

"""
	load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, 
	method="leave-one-out", contamination::Real=0.0, category)
Returns 3 tuples of (data, labels) representing train/validation/test part. Arguments are the splitting
ratios for normal data, seed and training data contamination.
For a list of available datasets, check `GenerativeAD.Datasets.uci_datasets`, `GenerativeAD.Datasets.other_datasets`,
`GenerativeAD.Datasets.mldatasets`. For MNIST-C and MVTec-AD datasets, the categories can be obtained by
 `GenerativeAD.Datasets.mnist_c_categories()` and `GenerativeAD.Datasets.mvtec_ad_categories()`.
"""
function load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, contamination::Real=0.0, kwargs...)
    data_normal, data_anomalous = load_uci_data(dataset; kwargs...)
    return train_val_test_split(data_normal, data_anomalous, ratios; seed=seed, contamination=contamination)
end