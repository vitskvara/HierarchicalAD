@testset "experiments" begin 
	indices = collect(1:1000)
	ratios = (0.8,0.1,0.1)
	randsplit = HierarchicalAD.train_val_test_inds(indices, ratios)

	# basic index splitting
	seed = 1
	splits = [HierarchicalAD.train_val_test_inds(indices, ratios, seed=seed) for _ in 1:2]
	@test all(map(spl->all(spl[1] .== spl[2]), zip(splits[1], splits[2])))

	# data splitting
	seed = 1
	dataset = "morpho_mnist"
	filter_dict = Dict()
	ratios = (0.8,0.1,0.1)
	splits = [HierarchicalAD.load_train_val_test_data(dataset, filter_dict; ratios=ratios, 
		seed=seed) for _ in 1:2];
	# test the labels
	@test all(splits[1][1][2][!,:length] .== splits[2][1][2][!,:length])
	# and the data
	@test all(splits[1][1][1][:,:,:,1:100] .== splits[2][1][1][:,:,:,1:100])
end