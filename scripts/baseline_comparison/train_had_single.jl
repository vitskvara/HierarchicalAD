using DrWatson
@quickactivate
using HierarchicalAD
using Flux, CUDA
using BSON, FileIO
using ValueHistories, DataFrames
using Plots
using ArgParse
using IterTools
using StatsBase

using HierarchicalAD: autoencoder_params_outputs_from_fvlae_data, create_input_data, knn_constructor
using HierarchicalAD: TrainingParameters, TrainingOutputs, HAD
using HierarchicalAD: fit_detectors!, fit_classifier!, predict
using HierarchicalAD: auc_val, classifier_score

arg_table = ArgParseSettings(;autofix_names=true)
@add_arg_table arg_table begin
	"--modeldir"
		default = "models/contaminated_training/run2/"
		help = "dir where the models are saved"
	"--model-id"
		default = nothing
		help = "id of the FVLAE model which is to be used for construction of HAD"
	"--model-ind"
		default = nothing
		help = "index of the FVLAE model which is to be used for constructionof of HAD"
end
args = parse_args(arg_table)
@unpack modeldir, model_id, model_ind = args

# get the data
X, y = HierarchicalAD.load_shapes2D();

# set device id
CUDA.device!(0)

# setup paths
modeldir = datadir(modeldir)
outpath = datadir("baseline_comparison/had_simple")
mkpath(outpath)

# what models are we gonna use
inmfs = readdir(modeldir)
if !isnothing(model_ind)
	mfs = [inmfs[Meta.parse(model_ind)]] 
elseif !isnothing(model_id)
	mfs = filter(x->occursin(model_id, x), mfs)
else
	mfs = inmfs # iterate over all 
end

mfs = ["activation=relu_channels=8-8-16-32-64_data=shapes2D_gamma=50.0_lambda=0.0_latent_dim=4_model=fvlae_model_id=20210817174044364_xdist=bernoulli.bson"]

# some params
nns = floor.(Int, 10.0 .^ range(1, 5, length=10))
#nns = floor.(Int, 10.0 .^ range(1, 2, length=2))
nas = floor.(Int, 10.0 .^ range(1, 4, length=4))
#nas = floor.(Int, 10.0 .^ range(1, 2, length=2))
ntst = 25000
label_vec = [
	y[!,:shape] .== 2, y[!,:shape] .!= 2, 
	y[!,:scale] .< 0.75, y[!,:scale] .> 0.75,
	y[!,:posX] .> 0.5, y[!,:posY] .> 0.5,
	y[!,:normalized_orientation] .< 0.1, 
	.!((y[!,:shape] .== 1) .| ((y[!,:posX] .> 0.5) .& (y[!,:posY] .> 0.5)))
	]
label_desc = [
	"ovals normal", "ovals anomalous",
	"small normal", "large normal",
	"right normal", "bottom normal",
	"rotated anomalous", 
	"squares or right bottom anomalous"
	]

# ovals normal
ilabel = 1
labels = label_vec[ilabel]
desc = label_desc[ilabel]
nns = range(400, 3000, step=200)
nas = range(100, 1000, step = 100)

# ovals anomalous
ilabel = 2
labels = label_vec[ilabel]
desc = label_desc[ilabel]
nns = range(400, 3000, step=200)
nas = range(100, 1000, step = 100)


function train_and_evaluate_had(mf, autoencoder_data)
	model_id = split(split(mf, "model_id=")[2], "_")[1]
	autoencoder_parameters, autoencoder_outputs = 
	    autoencoder_params_outputs_from_fvlae_data(autoencoder_data, size(X));
	
	# setup some other params
	lambda = 0.1f0
	batchsize = 128
	latent_count = autoencoder_data[:experiment_args].latent_count

	
	for seed = 1:1
		aucs_per_samples = []
		scores_per_samples = []
		tst_ys_per_samples = []
		for (nn, na) in product(nns, nas)
			@info "Training classifier using $(desc), seed=$seed."
			@info "# normal training/validation = $nn, # anomalous training/validation = $na, # test = $(ntst*2)"
			try
				global tr_nX, (tr_X, tr_y), (val_X, val_y), (tst_X, tst_y) = create_input_data(X, y, labels, nn, nn, na, na,
				    ntst; seed=seed);
			catch e
				if isa(e, BoundsError)
					@info "Not enough data, skipping training..."
					push!(aucs_per_samples, NaN)
					push!(scores_per_samples, [NaN])
					push!(tst_ys_per_samples, [NaN])
					continue
				else
					rethrow(e)
				end
			end
			k = min(21, size(tr_X,4))
			classifier_parameters = (λ=lambda, batchsize=batchsize, nepochs=200, val_ratio=0.2, n_candidates=100)    
			detector_parameters = (k=k, v=:kappa)

			# create the model
			model = HAD(
			    gpu(autoencoder_data[:model]),
			    [knn_constructor(;detector_parameters...) for _ in 1:latent_count],
			    Dense(latent_count+1, 2),
			    TrainingParameters(Dict(autoencoder_parameters), Dict(detector_parameters), Dict(classifier_parameters)),
			    TrainingOutputs(autoencoder_outputs, Dict(), Dict())
			    )
			# train detectors
			fit_detectors!(model, tr_nX);

			# fit classifier
			fit_classifier!(model, tr_X, tr_y, val_X, val_y);

			scores = StatsBase.predict(model, tst_X)
			auc = auc_val(tst_y, scores)
			@info "Finished training, test AUC = $(auc)"
			push!(aucs_per_samples, auc)
			push!(scores_per_samples, scores)
			push!(tst_ys_per_samples, tst_y)
		end
		df = replace(desc, " " => "_")
		savef = joinpath(outpath, "had_output_$(df)_model_id=$(model_id)_seed=$(seed).bson")
		outdata = Dict(
			:nns => nns, :nas => nas, :ntst => ntst,
			:model_id => model_id,
			:aucs => aucs_per_samples,
			:scores => scores_per_samples,
			:tst_ys => tst_ys_per_samples,
			:desc => desc
			)
		save(savef, outdata)
		@info "Saved to $savef"
	end
end

##### the most complete run

outpath = datadir("baseline_comparison/had_simple")
mkpath(outpath)

nns = vcat(range(30, 370, step=30), range(400, 3000, step=200))
nas = vcat([10], collect(range(100, 1000, step = 200)))

mfs = vcat(["activation=relu_channels=8-8-16-32-64_data=shapes2D_gamma=50.0_lambda=0.0_latent_dim=4_model=fvlae_model_id=20210817174044364_xdist=bernoulli.bson"],
	readdir(modeldir))
for mf in mfs
	for ilabel in 1:8
		labels = label_vec[ilabel]
		desc = label_desc[ilabel]
		autoencoder_data = load(joinpath(modeldir, mf))
		train_and_evaluate_had(mf, autoencoder_data)
	end
end


for mf in mfs
	autoencoder_data = load(joinpath(modeldir, mf))
	train_and_evaluate_had(mf, autoencoder_data)
end

nns = range(400, 3000, step=200)
nas = range(100, 1000, step = 100)

# 
ilabel = 8
labels = label_vec[ilabel]
desc = label_desc[ilabel]

for mf in mfs
	autoencoder_data = load(joinpath(modeldir, mf))
	train_and_evaluate_had(mf, autoencoder_data)
end

# 
ilabel = 5
labels = label_vec[ilabel]
desc = label_desc[ilabel]

for mf in mfs
	autoencoder_data = load(joinpath(modeldir, mf))
	train_and_evaluate_had(mf, autoencoder_data)
end

# 
outpath = datadir("baseline_comparison/had_simple")
ilabel = 3
labels = label_vec[ilabel]
desc = label_desc[ilabel]
nns = range(400, 3000, step=200)
nas = range(100, 1000, step = 200)

for mf in mfs
	autoencoder_data = load(joinpath(modeldir, mf))
	train_and_evaluate_had(mf, autoencoder_data)
end


# ovals normal
outpath = datadir("baseline_comparison/had_simple_smaller_ns")
mkpath(outpath)

ilabel = 1
labels = label_vec[ilabel]
desc = label_desc[ilabel]
nns = range(10, 400, step=30)
nas = vcat([10], collect(range(100, 1000, step = 200)))

for mf in mfs
	autoencoder_data = load(joinpath(modeldir, mf))
	train_and_evaluate_had(mf, autoencoder_data)
end


















######## this is some random bullshit
outpath = datadir("baseline_comparison/had_simple_even_batches")
mkpath(outpath)

ilabel = 1
labels = label_vec[ilabel]
desc = label_desc[ilabel]
nns = range(400, 3000, step=200)
nas = [300]

for mf in mfs
	autoencoder_data = load(joinpath(modeldir, mf))
	train_and_evaluate_had(mf, autoencoder_data)
end


