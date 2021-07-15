struct FVLAE{E,D,C,F,G,XD,ZD,V,XDIST<:Val} <: AbstractVLAE
    e::E
    d::D
    c::C # critic
    f::F # extracts latent variables
    g::G # concatenates latent vars with the rest
    xdim::XD # (h,w,c)
    zdim::ZD # scalar
    var::V # dense or conv last layer (ignore for dense fvlae)
    xdist::XDIST # gaussian or bernoulli
end
FVLAE(e,d,c,f,g,xdim,zdim,var,xdist::Symbol=:gaussian) = 
	FVLAE(e,d,c,f,g,xdim,zdim,var,Val(xdist))
Flux.@functor FVLAE
(m::FVLAE)(x) = reconstruct(m, x)

function Base.show(io::IO, m::FVLAE)
    msg = """$(nameof(typeof(m))):
        encoder = 
            e - $(repr(m.e))
            f - $(repr(m.f))
        decoder = 
            d - $(repr(m.d))
            g - $(repr(m.g))
        critic = 
            $(repr(m.c))
    """
    print(io, msg)
end

"""
	FVLAE(zdim::Int, hdim::Int, ks, ncs, stride, datasize; discriminator_nlayers::Int=3, 
		layer_depth::Int=1, var=:dense, activation="relu", xdist=:gaussian)

    FVLAE(zdim::Int, hdims, discriminator_hdim::Int, datasize; discriminator_nlayers::Int=2, 
    	activation=relu, layer_depth=1)

Factor VLAE constructor.
"""
function FVLAE(zdim::Int, hdim::Int, ks, ncs, strd, datasize; discriminator_nlayers::Int=3,
		var=:dense, xdist=:gaussian, kwargs...)
	(xdist in [:gaussian, :bernoulli]) ? nothing : 
		error("xdist must be either :gaussian or :bernoulli")

	# get encoder, decoder and latent extractors and reshapers
	e,d,f,g = basic_model_constructor(zdim, ks, ncs, strd, datasize; var=var, xdist=xdist, 
		kwargs...)

	# now construct the critic (discriminator)
	c = discriminator_constructor(zdim*length(ks), hdim, discriminator_nlayers; kwargs...)

    return FVLAE(e,d,c,f,g,datasize[1:3],zdim,var,xdist)
end
function FVLAE(zdim::Int, hdims, d_hdim::Int, datasize; discriminator_nlayers::Int=2, kwargs...)
    # get encoder, decoder and latent extractors and reshapers
    e,d,f,g = basic_model_constructor(zdim, hdims, datasize[1]; kwargs...)

	# now construct the critic (discriminator)
	c = HierarchicalAD.discriminator_constructor(zdim*length(hdims), d_hdim, 
		discriminator_nlayers; kwargs...)

    return HierarchicalAD.FVLAE(e,d,c,f,g,(datasize[1],),zdim,nothing,:gaussian)
end

function logpdf(m::FVLAE{E,D,C,F,G,X,Z,V,XD}, x::AbstractArray{T,2}, zs...) where 
		{E,D,C,F,G,X,Z,V,XD<:Val{:gaussian}} where T
    μx, σx = _decoded_mu_var(m, zs...)
    _x = (m.var == :dense) ? vectorize(x) : x      
    return  gauss_logpdf(_x, μx, σx)
end
function logpdf(m::FVLAE{E,D,C,F,G,X,Z,V,XD}, x::AbstractArray{T,4}, zs...) where 
	{E,D,C,F,G,X,Z,V,XD<:Val{:gaussian}} where T
    μx, σx = _decoded_mu_var1(m, zs...)
    _x = (m.var == :dense) ? vectorize(x) : x      
    return  gauss_logpdf(_x, μx, σx)
end
function logpdf(m::FVLAE{E,D,C,F,G,X,Z,V,XD}, x::AbstractArray{T,4}, zs...) where 
	{E,D,C,F,G,X,Z,V,XD<:Val{:bernoulli}} where T
    _x = _decoder_out(m, zs...)
    return  bernoulli_logpdf(_x, x)
end

dloss(d,z::AbstractArray{T,2},zp::AbstractArray{T,2}) where T  = 
	-0.5f0*Flux.mean(log.(d(z) .+ eps(Float32)) .+ log.(1 .- d(zp) .+ eps(Float32)))
function total_correlation(d,z::AbstractArray{T,2}) where T 
	dz = d(z)
	Flux.mean(log.(dz .+ eps(Float32)) .- log.(1 .- dz .+ eps(Float32)))
end

function factor_aeloss(m, x, γ::Float32)
	# encoder pass - KL divergence
	μzs_σzs = _encoded_mu_vars(m, x)
	zs = map(y->rptrick(y...), μzs_σzs)
	kldl = sum(map(y->Flux.mean(kld(y...)), μzs_σzs))
		
	# decoder pass - logpdf
	lpdf = Flux.mean(logpdf(m, x, zs...))
	elbo = -kldl + lpdf

	# now the discriminator loss
	_zs = cat(zs...,dims=1)
	tc = total_correlation(m.c, _zs)
		
	-elbo + γ*tc
end

reconstruction_loss(m,x)  = 
	-Flux.mean(logpdf(m, x, encode_all(m, x)...))
kld_loss(m, x) = 
    kldl = sum(map(y->Flux.mean(kld(y...)), _encoded_mu_vars(m, x)))
tc_loss(m, x) = 
	total_correlation(m.c, cat(encode_all(m, x)...,dims=1))

function permutation_mat_cols(x::AbstractArray{T,2}) where T # permutation per latent
	m,n = size(x)
	perm_mat = fill!(similar(x, n, n), 0)
	for (j,i) in zip(1:n, sample(1:n, n, replace=false))
		perm_mat[j,i] = 1
	end
	perm_mat
end
function permutation_mat_rows(x::AbstractArray{T,2}) where T 
	m,n = size(x)
	perm_mat = fill!(similar(x, m, m), 0)
	for (j,i) in zip(1:m, sample(1:m, m, replace=false))
		perm_mat[j,i] = 1
	end
	perm_mat
end
function permute_rows(x::AbstractArray{T,2}) where T # permutation per latent dim
    vcat(map(i->HierarchicalAD.permute_mat_cols(x[i:i,:]), 1:size(x,1))...)
end
permute_mat_rows(x::AbstractArray{T,2}) where T = permutation_mat_rows(x)*x
permute_mat_cols(x::AbstractArray{T,2}) where T = x*permutation_mat_cols(x)
permute_mat(x::AbstractArray{T,2}) where T = permute_rows(x)  
Flux.Zygote.@nograd permute_mat_rows, permute_mat_cols, permute_mat, permute_rows

function factor_closs(m, x)
	zs = _encode_all(m, x; mean=false)
	zps = map(zs) do z
		zp = permute_mat(z)
		zp
	end
	_zs = cat(zs..., dims=1)
	_zps = cat(zps..., dims=1)
	dloss(m.c, _zs, _zps)
end


#### THIS IS FOR IMAGE DATA ####
"""
	train_fvlae(zdim, hdim, batchsize, ks, ncs, stride, nepochs, tr_x, val_x; 
	λ=0.0f0, γ=1.0f0, epochsize = size(tr_x,4), layer_depth=1, lr=0.001f0, var=:dense, 
	activation="relu", discriminator_nlayers=3, xdist=:gaussian, pad=0,
	initial_convergence_threshold=0)

Train a factored VLAE.
"""
function train_fvlae(zdim, hdim, batchsize, ks, ncs, strd, nepochs, tr_x::AbstractArray{T,4}, 
	val_x::AbstractArray{T,4}; λ=0.0f0, γ=1.0f0, epochsize = size(tr_x,4), 
	layer_depth=1, lr=0.001f0, var=:dense, 
	activation="relu", discriminator_nlayers=3, xdist=:gaussian, pad=0,
	initial_convergence_threshold=0f0, initial_convergence_epochs=10,
    max_retrain_tries=10) where T
    # this is to ensure that the model converges to something meaningful
	hist, rdata, model, zs, aeopt, copt = nothing, nothing, nothing, nothing, nothing, nothing
    
    ntries = 0
    
    while isnothing(hist) && ntries<=max_retrain_tries
        if ntries > 0
            @info "Restarting training automatically..."
        end
        
        # construct the model
        model = gpu(FVLAE(zdim, hdim, ks, ncs, strd, size(tr_x), layer_depth=layer_depth, var=var, 
            activation=activation, nlayers=discriminator_nlayers, xdist=xdist, pad=pad))
        aeopt = ADAM(lr)
        copt = ADAM(lr)
    
        # train it
        hist, rdata, zs = train!(model, nepochs, batchsize, tr_x, val_x, aeopt, copt; 
            λ=λ, γ=γ, epochsize = epochsize, 
            initial_convergence_threshold=initial_convergence_threshold,
            initial_convergence_epochs=initial_convergence_epochs)
        ntries += 1
    end
    
    # if the training was still unsuccesfull
    if isnothing(hist)  
        @info "Gave up the training after $ntries tries."
        return model, hist, rdata, zs, aeopt, copt
    end

    return model, hist, rdata, zs, aeopt, copt
end

"""
	train!(model::FVLAE, nepochs, batchsize, tr_x::AbstractArray{T,4}, 
		val_x::AbstractArray{T,4}, aeopt, copt; 
		λ=0.0f0, γ=1.0f0, epochsize = size(tr_x,4), initial_convergence_threshold=0f0,
	    initial_convergence_epochs=10)
"""
function train!(model::FVLAE, nepochs, batchsize, tr_x::AbstractArray{T,4}, 
	val_x::AbstractArray{T,4}, aeopt, copt; 
	λ=0.0f0, γ=1.0f0, epochsize = size(tr_x,4), initial_convergence_threshold=0f0,
    initial_convergence_epochs=10) where T
	# data an initial reconstruction loss	
	gval_x = gpu(val_x[:,:,:,1:min(1000, size(val_x,4))]);
	local data_itr
	@suppress begin # suppress the batchsize warning
		data_itr = Flux.Data.DataLoader(sample_tensor(tr_x, epochsize), batchsize=batchsize)
	end
    init_rloss = batched_loss(x->reconstruction_loss(model,x), gval_x, batchsize)

    # params
    aeps = Flux.params(model.e, model.d, model.f, model.g)
    cps = Flux.params(model.c)
    
    # losses
    aeloss(x) = factor_aeloss(model, gpu(x), γ) + λ*sum(l2, aeps)
    closs(x) = factor_closs(model, gpu(x))

    # rest of the setup
    rdata = []
    hist = MVHistory()
    nl = length(model.e)
    zs = [[] for _ in 1:nl]
    
	println("Training in progress...")
	for epoch in 1:nepochs
		for x in data_itr
			# first update the autoencoder
			param_update!(aeloss, aeps, x, aeopt)

			# then update the critic/discriminator
			param_update!(closs, cps, x, copt)
		end

		 # sometimes the model is stuck in some local minima and cant get out - better restart the training
        if epoch == initial_convergence_epochs
            rloss = batched_loss(x->reconstruction_loss(model, x), gval_x, batchsize)
            if (init_rloss - rloss)/init_rloss < initial_convergence_threshold
                @info "Initial improvement after $(initial_convergence_epochs) epochs is $(init_rloss) => $rloss, model probably stuck in local minima, terminating training. Try restarting the model."
                return nothing, nothing, nothing
            end
        end

		# logging
		Flux.Zygote.ignore() do 
			ael=round(batched_loss(aeloss, gval_x, batchsize), digits=2)
			cl=round(batched_loss(closs, gval_x, batchsize), digits=4)
			rl=round(batched_loss(x->reconstruction_loss(model, x), gval_x, batchsize), digits=2)
			kldl=round(batched_loss(x->kld_loss(model, x), gval_x, batchsize), digits=2)
			tcl=round(batched_loss(x->tc_loss(model, x), gval_x, batchsize), digits=4)
			
			println("Epoch $(epoch)/$(nepochs), validation loss: AE=$ael | C=$cl | R=$rl | KLD=$kldl | TC=$tcl")
			for i in 1:nl
				z = encode(model, gval_x, i)
				push!(zs[i], cpu(z))
			end
			push!(hist, :autoencoder_loss, epoch, ael)
			push!(hist, :critic_loss, epoch, cl)
			push!(hist, :reconstruction_loss, epoch, rl)
			push!(hist, :kld, epoch, kldl)
			push!(hist, :total_correlation, epoch, tcl)
			push!(rdata, cpu(reconstruct(model, gval_x)))
		end
	end
	
	return hist, rdata, zs
end

#### THIS IS FOR TAB DATA ####
"""
	train_fvlae(zdim, hdim, batchsize, discriminator_hdim, nepochs, tr_x, val_x; 
	λ=0.0f0, γ=1.0f0, epochsize = size(tr_x,2), layer_depth=1, lr=0.001f0,
	activation="relu", discriminator_nlayers=3,
	initial_convergence_threshold=0, initial_convergence_epochs=10,
    max_retrain_tries=10)

Train a factored dense VLAE.
"""
function train_fvlae(zdim, hdims, batchsize, d_hdim, nepochs, tr_x::AbstractArray{T,2}, 
	val_x::AbstractArray{T,2}; λ=0.0f0, γ=1.0f0, epochsize = size(tr_x,2), 
	layer_depth=1, lr=0.001f0, activation="relu", discriminator_nlayers=3,
	initial_convergence_threshold=0f0, initial_convergence_epochs=10,
    max_retrain_tries=10, kwargs...) where T

    # this is to ensure that the model converges to something meaningful
	hist, rdata, model, zs, aeopt, copt = nothing, nothing, nothing, nothing, nothing, nothing
    
    ntries = 0
    
    while isnothing(hist) && ntries<=max_retrain_tries
        if ntries > 0
            @info "Restarting training automatically..."
        end
        
        # construct the model
        model = FVLAE(zdim, hdims, d_hdim, size(tr_x); activation=activation,
        	discriminator_nlayers=discriminator_nlayers, 
        	layer_depth=layer_depth)
        aeopt = ADAM(lr)
        copt = ADAM(lr)
    
        # train it
        hist, rdata, zs = train!(model, nepochs, batchsize, tr_x, val_x, aeopt, copt; 
            λ=λ, γ=γ, epochsize = epochsize, 
            initial_convergence_threshold=initial_convergence_threshold,
            initial_convergence_epochs=initial_convergence_epochs)
        ntries += 1
    end
    
    # if the training was still unsuccesfull
    if isnothing(hist)  
        @info "Gave up the training after $ntries tries."
        return model, hist, rdata, zs, aeopt, copt
    end

    return model, hist, rdata, zs, aeopt, copt
end

"""
	train!(model::FVLAE, nepochs, batchsize, tr_x::AbstractArray{T,2}, 
		val_x::AbstractArray{T,2}, aeopt, copt; 
		λ=0.0f0, γ=1.0f0, epochsize = size(tr_x,2), initial_convergence_threshold=0f0,
    	initial_convergence_epochs=50)
"""
function train!(model::FVLAE, nepochs, batchsize, tr_x::AbstractArray{T,2}, 
	val_x::AbstractArray{T,2}, aeopt, copt; 
	λ=0.0f0, γ=1.0f0, epochsize = size(tr_x,2), initial_convergence_threshold=0f0,
    initial_convergence_epochs=50) where T
	
	# data an initial reconstruction loss	
	subval_x = val_x[:,1:min(1000, size(val_x,2))]
	local data_itr
	@suppress begin # suppress the batchsize warning
		data_itr = Flux.Data.DataLoader(sample_tensor(tr_x, epochsize), batchsize=batchsize)
    end
    init_rloss = batched_loss(x->reconstruction_loss(model,x), subval_x, batchsize)

    # params
    aeps = Flux.params(model.e, model.d, model.f, model.g)
    cps = Flux.params(model.c)
    
    # losses
    aeloss(x) = factor_aeloss(model, x, γ) + λ*sum(l2, aeps)
    closs(x) = factor_closs(model, x)

    # rest of the setup
    rdata = []
    hist = MVHistory()
    nl = length(model.e)
    zs = [[] for _ in 1:nl]
    
	println("Training in progress...")
	for epoch in 1:nepochs
		for x in data_itr
			# first update the autoencoder
			param_update!(aeloss, aeps, x, aeopt)

			# then update the critic/discriminator
			param_update!(closs, cps, x, copt)
		end

		 # sometimes the model is stuck in some local minima and cant get out - better restart the training
        if epoch == initial_convergence_epochs
            rloss = batched_loss(x->reconstruction_loss(model, x), subval_x, batchsize)
            if (init_rloss - rloss)/init_rloss < initial_convergence_threshold
                @info "Initial improvement after $(initial_convergence_epochs) epochs is $(init_rloss) => $rloss, model probably stuck in local minima, terminating training. Try restarting the model."
                return nothing, nothing, nothing
            end
        end

		# logging
		Flux.Zygote.ignore() do 
			ael=round(batched_loss(aeloss, subval_x, batchsize), digits=2)
			cl=round(batched_loss(closs, subval_x, batchsize), digits=4)
			rl=round(batched_loss(x->reconstruction_loss(model, x), subval_x, batchsize), digits=2)
			kldl=round(batched_loss(x->kld_loss(model, x), subval_x, batchsize), digits=2)
			tcl=round(batched_loss(x->tc_loss(model, x), subval_x, batchsize), digits=4)
			
			println("Epoch $(epoch)/$(nepochs), validation loss: AE=$ael | C=$cl | R=$rl | KLD=$kldl | TC=$tcl")
			for i in 1:nl
				z = encode(model, subval_x, i)
				push!(zs[i], z)
			end
			push!(hist, :autoencoder_loss, epoch, ael)
			push!(hist, :critic_loss, epoch, cl)
			push!(hist, :reconstruction_loss, epoch, rl)
			push!(hist, :kld, epoch, kldl)
			push!(hist, :total_correlation, epoch, tcl)
			push!(rdata, reconstruct(model, subval_x))
		end
	end
	
	return hist, rdata, zs
end