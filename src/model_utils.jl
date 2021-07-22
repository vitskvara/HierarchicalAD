softplus_safe(x) = softplus(x) .+ 0.00001f0 
rptrick(μ::Flux.CUDA.CuArray, σ2) = μ .+ gpu(randn(Float32,size(μ))) .* σ2
rptrick(μ, σ2) = μ .+ randn(Float32,size(μ)) .* σ2

kld(μ::AbstractArray{T,N}, σ2::AbstractArray{T,N}) where T where N = 
    T(1/2)*sum(σ2 + μ.^2 - log.(σ2) .- T(1.0), dims = 1)
kld(μ::AbstractArray{T,N}, σ2::T) where T where N = 
    T(1/2)*sum(σ2 .+ μ.^2 .- log(σ2) .- T(1.0), dims = 1)

# gaussians
gauss_logpdf(x::AbstractArray{T,N}, μ::AbstractArray{T,N}, σ2::AbstractArray{T,N}) where T where N = 
    - vec(sum(((x - μ).^2) ./ σ2 .+ log.(σ2), dims=1)) .+ size(x,1)*log(T(2π)) / T(2)
gauss_logpdf(x::AbstractArray{T,N}, μ::AbstractArray{T,N}, σ2::T) where T where N = 
    - vec(sum(((x - μ).^2) ./ σ2 .+ log(σ2), dims=1)) .+ size(x,1)*log(T(2π)) / T(2)
gauss_logpdf(x::AbstractArray{T,4}, μ::AbstractArray{T,4}, σ2::AbstractArray{T,4}) where T = 
    - vec(sum((x - μ).^2 ./ σ2 .+ log.(σ2), dims = (1,2,3))) .+ prod(size(x)[1:3])*log(T(2π)) / T(2)

function mu_var(x)
    N = floor(Int, size(x,1)/2)
    μ = x[1:N,:]
    σ = softplus_safe.(x[N+1:end,:])
    return μ, σ
end

function mu_var1(x::AbstractArray{T,2}) where T
    μ = x[1:end-1,:]
    σ = softplus_safe.(x[end:end,:])
    return μ, σ
end
function mu_var1(x::AbstractArray{T,4}) where T
    μ = x[:,:,1:end-1,:]
    σ = softplus_safe.(Flux.mean(x[:,:,end:end,:], dims=(1,2,3)))
    return μ, σ
end

# logit prob.
bernoulli_logpdf(xh::AbstractArray{T,4}, x::AbstractArray{T,4}) where T = 
    -Flux.binarycrossentropy(xh, x, agg=x->vec(sum(x, dims=(1,2,3))))
bernoulli_logpdf(xh::AbstractArray{T,2}, x::AbstractArray{T,2}) where T = 
    -Flux.binarycrossentropy(xh, x, agg=x->vec(sum(x, dims=(1))))

l2(x) = sum(x.^2)

function param_update!(loss, ps, x, opt)
    gs = gradient(ps) do
        loss(x)
    end
    Flux.Optimise.update!(opt, ps, gs)
end

# constructors
function basic_model_constructor(zdim, hdims, indim; activation=relu, layer_depth=1,
        kwargs...)
    # activation function
    af = (typeof(activation) <: Function) ? activation : eval(Meta.parse(activation))

    # check that hdimensions are even
    all((hdims .% 2 ).== 0) ? nothing : error("hdims have to be even")

    # encoder
    e_h_in = vcat([indim], Int.([hdims[1:end-1]...]./2))
    e_h_out = [hdims...]
    e_afs = vcat([af for _ in 1:length(hdims)], [identity])
    e = []
    for (h_in, h_out, f) in zip(e_h_in, e_h_out, e_afs)
        hs = vcat(repeat([h_in], layer_depth), [h_out])
        fs = (f == identity) ? vcat([af for _ in 1:layer_depth-1], [f]) : [f for _ in 1:layer_depth]
        layer =  Chain([Dense(hs[i], hs[i+1], fs[i]) for i in 1:layer_depth]...)
        push!(e, layer)
    end
    e = Tuple(e)

    # decoder
    d_h_in = reverse([hdims...])
    d_h_out = vcat(Int.([reverse(hdims)[2:end]...]./2), [indim*2])
    d_afs = e_afs[2:end]
    d = []
    for (h_in, h_out, f) in zip(d_h_in, d_h_out, d_afs)
        hs = vcat([h_in], repeat([h_out], layer_depth))
        fs = (f == identity) ? vcat([af for _ in 1:layer_depth-1], [f]) : [f for _ in 1:layer_depth]
        layer =  Chain([Dense(hs[i], hs[i+1], fs[i]) for i in 1:layer_depth]...)
        push!(d, layer) 
    end
    d = Tuple(d)

    # f network
    f_h_in = Int.([hdims...]./2)
    f_h_out = [zdim*2 for _ in 1:length(hdims)]
    f = Tuple([Dense(h_in, h_out) for (h_in, h_out) in zip(f_h_in, f_h_out)])
    
    # g network
    g_h_in = [zdim for _ in 1:length(hdims)]
    g_h_out = Int.([reverse(hdims)...]./2)
    g_h_out[1] *= 2
    g = Tuple([Dense(h_in, h_out, af) for (h_in, h_out) in zip(g_h_in, g_h_out)])
    
    e,d,f,g
end
function basic_model_constructor(zdim::Int, ks, ncs, strd, datasize; layer_depth=1, 
    var=:dense, activation="relu", xdist=:gaussian, pad=0, kwargs...)
    nl = length(ncs) # no layers

    # number of channels for encoder/decoder
    indim = prod(datasize[1:3])
    rks = reverse(ks)
    ncs_in_e = vcat([datasize[3]], [floor(Int,n/2) for n in ncs[1:end-1]])
    ncs_in_d = reverse(ncs)
    ncs_out_d = vcat([floor(Int,n/2) for n in ncs_in_d[2:end]], [datasize[3]])
    ncs_out_f = vcat([ncs[end]], ncs_out_d[1:end-1])
    
    # activation function
    af = (typeof(activation) <: Function) ? activation : eval(Meta.parse(activation))

    # encoder/decoder
    if strd in [1,2]
        e, d = ae_constructor(indim, strd, ks, rks, ncs_in_e, ncs, ncs_in_d, 
                ncs_out_d, af, nl, xdist, var, datasize, pad) 
    else
        error("Requested stride length $strd not implemented.") 
    end

    # now capture the dimensions after each convolution
    outs = datasize
    _sout = []
    for (l,nc) in zip(e,ncs_in_e)
        cindim = (outs[1:2]..., nc, outs[end])
        outs = outdims(l, cindim)
        push!(_sout, [outs[1:2]...])
    end
    sout_e = (_sout...,) # this needs to be here for gpu compatibility 

    # now capture the dimensions after each decoder step
     sc=2^(strd-1) # this has to be here for proper size scaling
    outs = (sout_e[end].*sc..., datasize[3:4]...)
    _sout = [[outs[1], outs[2]]]
   for (l,nc) in zip(d[1:end],ncs_in_d[1:end-1])
        cindim = (outs[1:2]..., nc, outs[end])
        outs = outdims(l, cindim)
        push!(_sout, [outs[1:2]...])
    end
    sout_d = (_sout...,) # this needs to be here for gpu compatibility 

    # this is the vec. dimension after each convolution
    ddim_e = map(i->floor(Int,prod(sout_e[i])*ncs[i]/2), 1:length(ncs))
    ddim_d = map(i->floor(Int,prod(sout_d[i])*ncs_out_f[i]), 1:length(ncs))
    
    # latent extractor
    f = Tuple([Chain(x->reshape(x, :, size(x,4)), Dense(ddim_e[i], zdim*2)) for i in 1:nl])
    
    # latent reshaper
    g = Tuple([Chain(Dense(zdim, ddim_d[i], af), 
            x->reshape(x, sout_d[i]..., ncs_out_f[i], size(x,2))) for i in 1:nl])

    return e, d, f, g
end

function ae_constructor(indim, strd, ks, rks, ncs_in_e, ncs_out_e, ncs_in_d, ncs_out_d, af, 
    nl, xdist, var, datasize, pad=0)

    # encoder/decoder
    # negative striding does not work on GPU
    e = Tuple([Conv(ks[i], ncs_in_e[i]=>ncs_out_e[i], af, stride=strd) for i in 1:nl])
    if xdist == :bernoulli
        d = Tuple([[ConvTranspose(rks[i], ncs_in_d[i]=>ncs_out_d[i], af, stride=strd, 
                        pad=pad) for i in 1:nl-1]...,
                    Chain(
                        ConvTranspose(rks[end], ncs_in_d[end]=>ncs_out_d[end], af, stride=strd,
                            pad=pad),
                        AdaptiveMeanPool((datasize[1:2]...,)),
                        Conv((1,1), datasize[3]=>datasize[3], σ)
                        )]
                )
    elseif var == :dense
        d = Tuple([[ConvTranspose(rks[i], ncs_in_d[i]=>ncs_out_d[i], af, stride=strd,
                        pad=pad) for i in 1:nl-1]..., 
                Chain(
                    ConvTranspose(rks[end], ncs_in_d[end]=>ncs_out_d[end], af, stride=strd,
                        pad=pad),
                    AdaptiveMeanPool((datasize[1:2]...,)),
                    x->reshape(x, :, size(x,4)),
                    Dense(indim, indim+1)
                )])
    elseif var == :conv
        ncs_out_d[end] += 1
        d = Tuple([[ConvTranspose(rks[i], ncs_in_d[i]=>ncs_out_d[i], af, stride=strd,
                        pad=pad) for i in 1:nl-1]...,
                    Chain(
                        ConvTranspose(rks[end], ncs_in_d[end]=>ncs_out_d[end], af, stride=strd,
                            pad=pad),
                        AdaptiveMeanPool((datasize[1:2]...,)),
                        Conv((1,1), datasize[3]=>datasize[3])
                        )
                    ])
    else
        error("Decoder var=$var not implemented! Try one of `[:dense, :conv]`.")
    end    

    return e, d
end

function discriminator_constructor(zdim::Int, hdim::Int, nlayers::Int; activation="relu", kwargs...)
    (nlayers >= 2) ? nothing : error("`nlayers` must be at least 2.")
    af = (typeof(activation) <: Function) ? activation : eval(Meta.parse(activation))
    Chain(
        Dense(zdim, hdim, af), 
        [Dense(hdim, hdim, af) for _ in 1:nlayers-2]...,
        Dense(hdim, 1, σ) 
        )
end
