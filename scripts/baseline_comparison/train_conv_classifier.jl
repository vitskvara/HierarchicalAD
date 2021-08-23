using DrWatson
@quickactivate
using HierarchicalAD
using Flux, CUDA
using BSON, FileIO
using ValueHistories, DataFrames
using Plots
using ArgParse
using IterTools

using HierarchicalAD: construct_conv_classifier, train_conv_classifier!, create_input_data
using HierarchicalAD: auc_val, classifier_score

arg_table = ArgParseSettings(;autofix_names=true)
@add_arg_table arg_table begin
	"--modeldir"
		default = "models/contaminated_training/run2/"
		help = "dir where the models are saved"
	"--model-id"
		default = nothing
		help = "id of the FVLAE model whose architecture is to be used for construction fo the classifier"
	"--model-ind"
		default = nothing
		help = "index of the FVLAE model whose architecture is to be used for construction fo the classifier"
end
args = parse_args(arg_table)
@unpack modeldir, model_id, model_ind = args

# get the data
X, y = HierarchicalAD.load_shapes2D();

# set device id
CUDA.device!(2)

# setup paths
modeldir = datadir(modeldir)
plotpath = datadir("plots/baseline_comparison/conv_classifier")
outpath = datadir("data/baseline_comparison/conv_classifier")
mkpath(outpath)
mkpath(plotpath)

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
conv_classifier_params = (lr = 0.001f0, λ = 0f0, batchsize = 64, nepochs = 1000, patience = 1, verb = true)

function train_and_evaluate_classifier(mf)
	model_id = split(split(mf, "model_id=")[2], "_")[1]
	autoencoder_data = load(joinpath(modeldir, mf))
	
	for seed = 1:10
		aucs_per_split = []
		scores_per_split = []
		tst_ys_per_split = []
		for (labels, desc) in zip(label_vec, label_desc)
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
				classifier = construct_conv_classifier(
					autoencoder_data[:experiment_args].kernelsizes,
					autoencoder_data[:experiment_args].activation,
					autoencoder_data[:experiment_args].stride,
					autoencoder_data[:experiment_args].channels,
					size(X))

				history, opt = train_conv_classifier!(classifier, tr_X, tr_y, val_X, val_y; conv_classifier_params...);
				scores = cpu(classifier_score(classifier, tst_X, 128))
				auc = auc_val(tst_y, scores)
				@info "Finished training, test AUC = $(auc)"
				push!(aucs_per_samples, auc)
				push!(scores_per_samples, scores)
				push!(tst_ys_per_samples, tst_y)
			end
			push!(aucs_per_split, aucs_per_samples)
			push!(scores_per_split, scores_per_samples)
			push!(tst_ys_per_split, tst_ys_per_samples)
		end
		savef = joinpath(outpath, "classifier_output_model_id=$(model_id)_seed=$(seed).bson")
		outdata = Dict(
			:nns => nns, :nas => nas, :ntst => ntst,
			:model_id => model_id,
			:conv_classifier_params => conv_classifier_params,
			:aucs => aucs_per_split,
			:scores => scores_per_split,
			:tst_ys => tst_ys_per_split,
			:label_vec => label_vec,
			:label_desc => label_desc
			)
		save(savef, outdata)
		@info "Saved to $savef"
	end
end

for mf in mfs
	train_and_evaluate_classifier(mf)
end

