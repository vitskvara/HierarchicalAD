struct FVLAE <: AbstractVLAE
    e
    d
    c # critic
    g # extracts latent variables
    f # concatenates latent vars with the rest
    xdim # (h,w,c)
    zdim # scalar
    var # dense or conv last layer
end

Flux.@functor FVLAE
(m::FVLAE)(x) = reconstruct(m, x)

"""
	FVLAE(zdim::Int, hdim::Int, ks, ncs, stride, datasize; 
		discriminator_nlayers::Int=3, layer_depth::Int=1, var=:dense, activation="relu")

Factor VLAE constructor.
"""
function FVLAE(zdim::Int, hdim::Int, ks, ncs, str, datasize; discriminator_nlayers::Int=3,
		var=:dense, kwargs...)
	# get encoder, decoder and latent extractors and reshapers
	e,d,g,f = basic_model_constructor(zdim, ks, ncs, str, datasize; var=var, kwargs...)

	# now construct the critic (discriminator)
	c = discriminator_constructor(zdim*length(ks), hdim, discriminator_nlayers; kwargs...)

    return FVLAE(e,d,c,g,f,datasize[1:3],zdim,var)
end

dloss(d,z::AbstractArray{T,2},zp::AbstractArray{T,2}) where T  = 
	-0.5f0*Flux.mean(log.(d(z) .+ eps(Float32)) .+ log.(1 .- d(zp) .+ eps(Float32)))
function total_correlation(d,z::AbstractArray{T,2}) where T 
	dz = d(z)
	Flux.mean(log.(dz .+ eps(Float32)) .- log.(1 .- dz .+ eps(Float32)))
end

function factor_aeloss(m, x::AbstractArray{T,4}, γ::Float32) where T
	# encoder pass
	μzs_σzs = _encoded_mu_vars(m, x)
	zs = map(y->rptrick(y...), μzs_σzs)
	kldl = sum(map(y->Flux.mean(kld(y...)), μzs_σzs))
		
	# decoder pass
	μx, σx = _decoded_mu_var(m, zs...)
	_x = (m.var == :dense) ? vectorize(x) : x
	elbo = -kldl + Flux.mean(logpdf(_x, μx, σx))

	# now the discriminator loss
	_zs = cat(zs...,dims=1)
	tcl = total_correlation(m.c, _zs)
		
	-elbo + γ*tcl
end

function factor_closs(m, x::AbstractArray{T,4}) where T
	zs = encode_all(m, x)
	zps = map(zs) do z
		pm = permute_mat(z)
		z*pm
	end
	_zs = cat(zs..., dims=1)
	_zps = cat(zps..., dims=1)
	dloss(m.c, _zs, _zps)
end

function permute_mat(x::AbstractArray{T,2}) where T
	m,n = size(x)
	perm_mat = fill!(similar(x, n, n), 0)
	for (j,i) in zip(1:n, sample(1:n, n, replace=false))
		perm_mat[j,i] = 1
	end
	perm_mat
end
Flux.Zygote.@nograd permute_mat

"""
	train_fvlae(zdim, hdim, batchsize, ks, ncs, stride, nepochs, data, val_x, tst_x; 
	λ=0.0f0, γ=1.0f0, epochsize = size(data,4), layer_depth=1, lr=0.001f0, var=:dense, 
	activation=activation, discriminator_nlayers=3)

Train a factored VLAE.
"""
function train_fvlae(zdim, hdim, batchsize, ks, ncs, str, nepochs, data, val_x, tst_x; 
	λ=0.0f0, γ=1.0f0, epochsize = size(data,4), layer_depth=1, lr=0.001f0, var=:dense, 
	activation=activation, discriminator_nlayers=3)

	gval_x = gpu(val_x[:,:,:,1:min(1000, size(val_x,4))]);
	gtst_x = gpu(tst_x);
	
	model = gpu(FVLAE(zdim, hdim, ks, ncs, str, size(data), layer_depth=layer_depth, var=var, 
		activation=activation, nlayers=discriminator_nlayers))
	nl = length(model.e)
	
	aeps = Flux.params(model.e, model.d)
	cps = Flux.params(model.c)
	aeopt = ADAM(lr)
	copt = ADAM(lr)

	aeloss(x) = factor_aeloss(model, gpu(x), γ) + λ*sum(l2, aeps)
	closs(x) = factor_closs(model, gpu(x))
	rdata = []
	hist = MVHistory()
	zs = [[] for _ in 1:nl]
	data_itr = Flux.Data.DataLoader(data[:,:,:,sample(1:size(data,4), epochsize)], batchsize=batchsize)

	println("Training in progress...")
	for epoch in 1:nepochs
		for x in data_itr
			# first update the autoencoder
			param_update!(aeloss, aeps, x, aeopt)

			# then update the critic/discriminator
			param_update!(closs, cps, x, copt)
		end

		ael = Flux.mean(map(x->aeloss(x), Flux.Data.DataLoader(val_x, batchsize=batchsize)))
		cl = Flux.mean(map(x->closs(x), Flux.Data.DataLoader(val_x, batchsize=batchsize)))
		#elbo = Flux.mean(map(x->aeloss(x), Flux.Data.DataLoader(val_x, batchsize=batchsize)))
		#tc = Flux.mean(map(x->aeloss(x), Flux.Data.DataLoader(val_x, batchsize=batchsize)))
		
		println("Epoch $(epoch)/$(nepochs), validation loss: AE = $ael, C = $cl")
		for i in 1:nl
			z = encode(model, gval_x, i)
			push!(zs[i], cpu(z))
		end
		push!(hist, :aeloss, epoch, aeloss)
		push!(hist, :closs, epoch, closs)
		push!(rdata, cpu(reconstruct(model, gval_x)))
	end
	
	return model, hist, rdata, zs
end

