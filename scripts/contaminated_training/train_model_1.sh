julia --project train_model.jl 2 64 16 32 64 --savepath=contaminated_training/run1 --layer_depth=2 \
	--seed=1 --nepochs=1 --gamma=10 --shape 1 2 3 --scale ">=" 0.5 --normalized_orientation ">=" 0 \
	--posX ">=" 0.5 --posY ">=" 0.5