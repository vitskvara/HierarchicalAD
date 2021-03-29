softplus_safe(x) = softplus(x) .+ 0.00001f0 
rptrick(μ::Flux.CUDA.CuArray, σ2) = μ .+ gpu(randn(Float32,size(μ))) .* σ2
rptrick(μ, σ2) = μ .+ randn(Float32,size(μ)) .* σ2

kld(μ::AbstractArray{T,N}, σ2::AbstractArray{T,N}) where T where N = 
    T(1/2)*sum(σ2 + μ.^2 - log.(σ2) .- T(1.0), dims = 1)
kld(μ::AbstractArray{T,N}, σ2::T) where T where N = 
    T(1/2)*sum(σ2 .+ μ.^2 .- log(σ2) .- T(1.0), dims = 1)

# gaussians
logpdf(x::AbstractArray{T,N}, μ::AbstractArray{T,N}, σ2::AbstractArray{T,N}) where T where N = 
    - vec(sum(((x - μ).^2) ./ σ2 .+ log.(σ2), dims=1)) .+ size(x,1)*log(T(2π)) / T(2)
logpdf(x::AbstractArray{T,N}, μ::AbstractArray{T,N}, σ2::T) where T where N = 
    - vec(sum(((x - μ).^2) ./ σ2 .+ log(σ2), dims=1)) .+ size(x,1)*log(T(2π)) / T(2)
logpdf(x::AbstractArray{T,4}, μ::AbstractArray{T,4}, σ2::AbstractArray{T,4}) where T = 
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
bernoulli_prob(xh::AbstractArray{T,4}, x::AbstractArray{T,4}) where T = 
    map(i->Flux.binarycrossentropy(xh[:,:,:,i], x[:,:,:,i]), 1:size(x,4))
bernoulli_prob(xh::AbstractArray{T,2}, x::AbstractArray{T,2}) where T = 
    map(i->Flux.binarycrossentropy(xh[:,i], x[:,i]), 1:size(x,2))

l2(x) = sum(x.^2)

function param_update!(loss, ps, x, opt)
    gs = gradient(ps) do
        loss(x)
    end
    Flux.Optimise.update!(opt, ps, gs)
end

# constructors
function basic_model_constructor(zdim::Int, ks, ncs, strd, datasize; layer_depth=1, 
    var=:dense, activation="relu", xdist=:gaussian, kwargs...)
    nl = length(ncs) # no layers
    # this captures the dimensions after each convolution
    sout = Tuple(map(j -> datasize[1:2] .- [sum(map(k->k[i]-1, ks[1:j])) for i in 1:2], 1:length(ncs))) 
    # this is the vec. dimension after each convolution
    ddim = map(i->floor(Int,prod(sout[i])*ncs[i]/2), 1:length(ncs))
    ddim_d = copy(ddim)
    ddim_d[end] = ddim_d[end]*2
    indim = prod(datasize[1:3])
    rks = reverse(ks)
    rsout = reverse(sout)

    # number of channels for encoder/decoder
    ncs_in_e = vcat([datasize[3]], [floor(Int,n/2) for n in ncs[1:end-1]])
    ncs_in_d = reverse(ncs)
    ncs_out_d = vcat([floor(Int,n/2) for n in ncs_in_d[2:end]], [datasize[3]])
    ncs_out_f = vcat([floor(Int,n/2) for n in ncs[1:end-1]], [ncs[end]])
    
    # activation function
    af = (typeof(activation) <: Function) ? activation : eval(Meta.parse(activation))

    # encoder/decoder
    e = Tuple([Conv(ks[i], ncs_in_e[i]=>ncs[i], af, stride=strd) for i in 1:nl])
    if xdist == :bernoulli
        d = Tuple([[ConvTranspose(rks[i], ncs_in_d[i]=>ncs_out_d[i], af, stride=strd) for i in 1:nl-1]...,
                    ConvTranspose(rks[end], ncs_in_d[end]=>ncs_out_d[end], σ, stride=strd)]
                )
    elseif var == :dense
        d = Tuple([[ConvTranspose(rks[i], ncs_in_d[i]=>ncs_out_d[i], af, stride=strd) for i in 1:nl-1]...,
                Chain(
                    ConvTranspose(rks[end], ncs_in_d[end]=>ncs_out_d[end], af, stride=strd),
                    x->reshape(x, :, size(x,4)),
                    Dense(indim, indim+1)
                )]
            )
    elseif var == :conv
        ncs_out_d[end] += 1
        d = Tuple([[ConvTranspose(rks[i], ncs_in_d[i]=>ncs_out_d[i], af, stride=strd) for i in 1:nl-1]...,
                    ConvTranspose(rks[end], ncs_in_d[end]=>ncs_out_d[end], stride=strd)]
                )
    else
        error("Decoder var=$var not implemented! Try one of `[:dense, :conv]`.")
    end    
    
    # latent extractor
    g = Tuple([Chain(x->reshape(x, :, size(x,4)), Dense(ddim[i], zdim*2)) for i in 1:nl])
    
    # latent reshaper
    f = Tuple([Chain(Dense(zdim, ddim_d[i], af), 
            x->reshape(x, sout[i]..., ncs_out_f[i], size(x,2))) for i in nl:-1:1])

    return e, d, g, f
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
