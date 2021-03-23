using DrWatson
@quickactivate
using ArgParse
using CSV, DataFrames
using HierarchicalAD
using FileIO, BSON
using Flux
using CUDA
CUDA.device!(1)

include(scriptsdir("plotting_code.jl"))

s = ArgParseSettings()
@add_arg_table s begin
	"path"
		help = "either a modelfile or a dir containing multiple modelfiles"
		arg_type = String
end
args = parse_args(s)
@unpack path = args

function evaluate_model(modeldata, modelfile)
	@unpack model, training_history = modeldata
	gmodel = gpu(model)

	# now print some info
	println("\n")
	@info "Loaded $modelfile."
	println("  model_id => $(modeldata[:experiment_args].model_id)")
	println("  data => $(modeldata[:experiment_args].data)")
	println("  non-default filters => ")
	for f in modeldata[:non_default_filters]
		fv = modeldata[:filter_dict][f]
		if f != "digit"
			println("    $(f) $(fv[1]) $(fv[2])")
		else
			println("    $(f) $(fv)")
		end
	end
	println("\n")

	# create the plot path
	modeltype = split(dirname(modelfile), "/")[end]
	plotpath = plotsdir(modeltype)
	mkpath(plotpath)
	evalpath = datadir("eval/$modeltype")
	mkpath(evalpath)
	model_id = modeldata[:experiment_args].model_id

	# load the same data that were used for creation of the model
	dataset = modeldata[:experiment_args].data
	ratios = modeldata[:ratios]
	filter_dict = modeldata[:filter_dict]
	seed = modeldata[:experiment_args].seed
	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y), (a_x, a_y) = 
	    HierarchicalAD.load_train_val_test_data(dataset, filter_dict; ratios=ratios, seed=seed)

	# plots
	# 01 plot training
	p = plot_training(training_history, plotsize = (600,300))
	svn = joinpath(plotpath, "$(model_id)_01_training_history.png")
	savefig(svn)
	@info "Saved $svn"

	# 02 reconstructions and generated digits
	p = compare_reconstructions(gmodel, tst_x)
	svn = joinpath(plotpath, "$(model_id)_02_reconstructions.png")
	savefig(svn)
	@info "Saved $svn"

	# 03 latent spaces
	p = plot_latents(gmodel)
	svn = joinpath(plotpath, "$(model_id)_03_latent_spaces.png")
	savefig(svn)
	@info "Saved $svn"

	### anomaly scores
	@unpack tr_scores, val_scores, tst_scores, a_scores, non_default_filters = modeldata;

	# digit anomalies
	p=plot_anomalies_digit(tr_y, a_y, tr_scores, val_scores, a_scores)
	svn = joinpath(plotpath, "$(model_id)_04_anomalies_digits.png")
	savefig(svn)
	@info "Saved $svn"

	# other anomalies
	p=plot_anomalies_other(non_default_filters, filter_dict, a_y, tr_scores, a_scores)
	svn = joinpath(plotpath, "$(model_id)_05_anomalies_others.png")
	savefig(svn)
	@info "Saved $svn"

	### latent spaces
	γ = 0.1f0
	k = GaussianKernel(γ)
	@unpack tr_encodings, val_encodings, tst_encodings, a_encodings = modeldata;
	nl = length(model.e)
	mmd_dict = Dict()

	# digit identity
	cat_vals = sort(unique(tr_y[!,:digit]))
	category = :digit
	p,mmds=plot_latent(cat_vals, category, tr_y, k, tr_encodings..., dims=[1,2])
	plot(p)
	svn = joinpath(plotpath, "$(model_id)_06_encodings_train_digits.png")
	savefig(svn)
	@info "Saved $svn"
	print_mmd_overview(category, cat_vals, mmds)
	mmd_dict[String(category)] = mmds

	# other filters
	nbins = 5
	p, mmds = mmd_overview_other(nbins, non_default_filters, tr_y, tr_encodings, k, model) 
	plot(p)
	svn = joinpath(plotpath, "$(model_id)_07_encodings_train_other.png")
	savefig(svn)
	@info "Saved $svn"
	mmd_dict = merge(mmd_dict, mmds)

	# now compare normal vs anomalous
	p, mmds = mmd_overview_anomalies(tr_y, a_y, tr_encodings, a_encodings, non_default_filters, 
		k, filter_dict)
	print_mmd_overview_anomalies(mmds)
	plot(p)
	svn = joinpath(plotpath, "$(model_id)_08_encodings_anomalies.png")
	savefig(svn)
	@info "Saved $svn"
	mmd_dict = merge(mmd_dict, mmds)

	# save results (mmds)
	svn = joinpath(evalpath, "$(model_id)_mmds_train.bson")
	save(svn, mmd_dict)
	@info "Saved $svn"
end

### run from here
# ooening of bson files has to be done in global scope, otherwise strange things happen :(
if isfile(path)
	modeldata = try
		load(path);
	catch e
		@info "File $path could not be opened!"
		return 
	end
	evaluate_model(modeldata, path)
else
	for f in readdir(path, join=true) 
		modeldata = try
			load(f);
		catch e
			@info "File $f could not be opened!"
			continue
		end
		evaluate_model(modeldata, f)
	end
end