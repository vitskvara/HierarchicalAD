struct VLAE{E,D,G,F,XD,ZD,V,XDIST<:Val} <: AbstractVLAE
    e::E
    d::D
    g::G # extracts latent variables
    f::F # concatenates latent vars with the rest
    xdim::XD # (h,w,c)
    zdim::ZD # scalar
    var::V # dense or conv last layer
    xdist::XDIST # gaussian or bernoulli
end
VLAE(e,d,g,f,xdim,zdim,var,xdist::Symbol=:gaussian) = 
    VLAE(e,d,g,f,xdim,zdim,var,Val(xdist))
Flux.@functor VLAE
(m::VLAE)(x) = reconstruct(m, x)

"""
	VLAE(zdim::Int, ks, ncs, stride, datasize; layer_depth=1, var=:dense, activation="relu",
        xdist=:gaussian)

Basic VLAE constructor.
"""
function VLAE(zdim::Int, ks, ncs, strd, datasize; var=:dense, xdist=:gaussian, kwargs...)
    (xdist in [:gaussian, :bernoulli]) ? nothing : 
        error("xdist must be either :gaussian or :bernoulli")

	# get encoder, decoder and latent extractors and reshapers
	e,d,g,f = basic_model_constructor(zdim, ks, ncs, strd, datasize; var=var, xdist=xdist, kwargs...)

    return VLAE(e,d,g,f,datasize[1:3],zdim,var,xdist)
end

"""
	train_vlae(zdim, batchsize, ks, ncs, stride, nepochs, data, val_x, tst_x; 
    	λ=0.0f0, epochsize = size(data,4), layer_depth=1, lr=0.001f0, var=:dense, 
    	activation=activation)

"""
function train_vlae(zdim, batchsize, ks, ncs, strd, nepochs, data, val_x, tst_x; 
    λ=0.0f0, epochsize = size(data,4), layer_depth=1, lr=0.001f0, var=:dense, 
    activation=activation, xdist=:gaussian)
    gval_x = gpu(val_x[:,:,:,1:min(1000, size(val_x,4))])
    gtst_x = gpu(tst_x)
    
    model = gpu(VLAE(zdim, ks, ncs, strd, size(data), layer_depth=layer_depth, var=var, 
        activation=activation, xdist=xdist))
    nl = length(model.e)
    
    ps = Flux.params(model)
    loss(x) = -elbo(model,gpu(x)) + λ*sum(l2, ps)
    opt = ADAM(lr)
    rdata = []
    hist = MVHistory()
    zs = [[] for _ in 1:nl]
    
    # train
    println("Training in progress...")
    for epoch in 1:nepochs
        data_itr = Flux.Data.DataLoader(data[:,:,:,sample(1:size(data,4), epochsize)], batchsize=batchsize)
        Flux.train!(loss, ps, data_itr, opt)
        l = Flux.mean(map(x->loss(x), Flux.Data.DataLoader(val_x, batchsize=batchsize)))
        println("Epoch $(epoch)/$(nepochs), validation loss = $l")
        for i in 1:nl
            z = encode(model, gval_x, i)
            push!(zs[i], cpu(z))
        end
        push!(hist, :loss, epoch, l)
        push!(rdata, cpu(reconstruct(model, gval_x)))
    end
    
    return model, hist, rdata, zs
end

function logpdf(m::VLAE{E,D,G,F,X,Z,V,XD}, x, zs...) where {E,D,G,F,X,Z,V,XD<:Val{:gaussian}}
    μx, σx = _decoded_mu_var(m, zs...)
    _x = (m.var == :dense) ? vectorize(x) : x      
    return  gauss_logpdf(_x, μx, σx)
end
function logpdf(m::VLAE{E,D,G,F,X,Z,V,XD}, x, zs...) where {E,D,G,F,X,Z,V,XD<:Val{:bernoulli}}
    _x = _decoder_out(m, zs...)
    return  bernoulli_logpdf(_x, x)
end