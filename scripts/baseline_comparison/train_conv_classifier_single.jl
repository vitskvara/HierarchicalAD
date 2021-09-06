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

# get the data
X, y = HierarchicalAD.load_shapes2D();
datasize = size(X)

# set device id
CUDA.device!(2)

# setup paths
outpath = datadir("baseline_comparison/conv_simple")
mkpath(outpath)

# original defaults
nns = floor.(Int, 10.0 .^ range(1, 5, length=10))
nas = floor.(Int, 10.0 .^ range(1, 4, length=4))
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
nns = floor.(Int, 10.0 .^ range(1, 4, length=8))
nns = range(400, 3000, step=200)
nas = range(100, 1000, step = 100)

# ovals anomalous
ilabel = 2
labels = label_vec[ilabel]
desc = label_desc[ilabel]
nns = floor.(Int, 10.0 .^ range(1, 4, length=8))
nns = range(400, 3000, step=200)
nas = range(100, 1000, step = 100)

# ovals anomalous
ilabel = 8
labels = label_vec[ilabel]
desc = label_desc[ilabel]
nns = floor.(Int, 10.0 .^ range(1, 4, length=8))
nns = range(400, 3000, step=200)
nas = range(100, 1000, step = 100)

conv_classifier_params = (lr = 0.001f0, λ = 0f0, batchsize = 64, nepochs = 1000, patience = 100, verb = true)

function train_and_evaluate_classifier()
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
			# construct the classifier
			classifier_conv_part = [Conv((5,5), 1=>32, relu), MaxPool((3,3))]
			odims = outdims(classifier_conv_part, datasize)
			odim = reduce(*, odims[1:3])
			classifier = gpu(Chain(classifier_conv_part..., x->reshape(x, odim, size(x,4)),  Dense(odim, 2)))

			# train it
			classifier, history, opt = train_conv_classifier!(classifier, tr_X, tr_y, val_X, val_y; conv_classifier_params...);
			scores = cpu(classifier_score(classifier, tst_X, 128))
			auc = auc_val(tst_y, scores)
			@info "Finished training, test AUC = $(auc)"
			push!(aucs_per_samples, auc)
			push!(scores_per_samples, scores)
			push!(tst_ys_per_samples, tst_y)
		end
		df = replace(desc, " " => "_")
		savef = joinpath(outpath, "classifier_output_$(df)_seed=$(seed).bson")
		outdata = Dict(
			:nns => nns, :nas => nas, :ntst => ntst,
			:conv_classifier_params => conv_classifier_params,
			:aucs => aucs_per_samples,
			:scores => scores_per_samples,
			:tst_ys => tst_ys_per_samples,
			:desc => desc
			)
		save(savef, outdata)
		@info "Saved to $savef"
	end
end

outpath = datadir("baseline_comparison/conv_simple")
mkpath(outpath)

nns = vcat(range(30, 370, step=30), range(400, 3000, step=200))
nas = vcat([10], collect(range(100, 1000, step = 200)))

for ilabel in 1:8
	labels = label_vec[ilabel]
	desc = label_desc[ilabel]
	train_and_evaluate_classifier()
end


train_and_evaluate_classifier()

nns = range(400, 3000, step=200)
nas = range(100, 1000, step = 100)

# 
ilabel = 8
labels = label_vec[ilabel]
desc = label_desc[ilabel]

train_and_evaluate_classifier()

# 
ilabel = 5
labels = label_vec[ilabel]
desc = label_desc[ilabel]

train_and_evaluate_classifier()

# 
ilabel = 3
labels = label_vec[ilabel]
desc = label_desc[ilabel]

train_and_evaluate_classifier()



########
outpath = datadir("baseline_comparison/conv_simple_smaller_ns")
mkpath(outpath)

ilabel = 1
labels = label_vec[ilabel]
desc = label_desc[ilabel]
nns = range(10, 400, step=30)
nas = vcat([10], collect(range(100, 1000, step = 200)))
train_and_evaluate_classifier()



# the most compelte run
outpath = datadir("baseline_comparison/had_simple")
mkpath(outpath)

nns = vcat(range(10, 370, step=30), range(400, 3000, step=200))
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
