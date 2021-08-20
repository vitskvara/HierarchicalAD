abstract type AbstractVLAE end

function elbo(m::AbstractVLAE, x::AbstractArray{T,4}) where T
    # encoder pass - KL divergence
    μzs_σzs = _encoded_mu_vars(m, x)
    zs = map(y->rptrick(y...), μzs_σzs)
    kldl = sum(map(y->Flux.mean(kld(y...)), μzs_σzs))
        
    # decoder pass - logpdf
    lpdf = Flux.mean(logpdf(m, x, zs...))

    -kldl + lpdf
end

function _encoded_mu_vars(m::AbstractVLAE, x)
    nl = length(m.e)

    h = x
    mu_vars = map(1:nl) do i
        h = m.e[i](h)
        h, μz, σz = _extract_mu_var(m.f, h, i)
        (μz, σz)
    end
    mu_vars
end

function _extract_mu_var(f, h::AbstractArray{T,4}, i) where T
    nch = floor(Int,size(h,3)/2)
    μz, σz = mu_var(f[i](h[:,:,(nch+1):end,:]))
    h = h[:,:,1:nch,:]
    h, μz, σz
end

function _extract_mu_var(f, h::AbstractArray{T,2}, i) where T
    nch = floor(Int,size(h,1)/2)
    μz, σz = mu_var(f[i](h[(nch+1):end,:]))
    h = h[1:nch,:]
    h, μz, σz
end

function _decoder_out(m::AbstractVLAE, zs...)
    nl = length(m.d)
    @assert length(zs) == nl
    
    h = m.g[1](zs[end])
    # now, propagate through the decoder
    for i in 1:nl-1
        h1 = m.d[i](h)
        h2 = m.g[i+1](zs[end-i])
        h = _cat_arrays(h1, h2)
    end
    h = m.d[end](h)
end
function _decoded_mu_var(m::AbstractVLAE, zs...)
    h = _decoder_out(m, zs...)
    μx, σx = mu_var(h)
end
function _decoded_mu_var1(m::AbstractVLAE, zs...)
    h = _decoder_out(m, zs...)
    μx, σx = mu_var1(h)
end
_cat_arrays(h1::AbstractArray{T,4}, h2::AbstractArray{T,4}) where T = cat(h1, h2, dims=3)
_cat_arrays(h1::AbstractArray{T,2}, h2::AbstractArray{T,2}) where T = cat(h1, h2, dims=1)

# encoding functions
function _batched_encs(f, m, x, batchsize, args...; kwargs...)
    local encs
    @suppress begin
        encs = map(y->f(gpu(m), gpu(y), args...; kwargs...), 
            Flux.Data.DataLoader(x, batchsize=batchsize))
    end
    [cat([y[i] for y in encs]..., dims=2) for i in 1:length(encs[1])]
end
function _batched_encs(f, m, x::AbstractArray{T,2}, batchsize, args...; kwargs...) where T
    local encs
    @suppress begin
        encs = map(y->cpu(f(m, y, args...; kwargs...)), 
            Flux.Data.DataLoader(x, batchsize=batchsize))
    end
    [cat([y[i] for y in encs]..., dims=2) for i in 1:length(encs[1])]
end
function _encode_ith(m::AbstractVLAE, x, i::Int; mean=false)
    μzs_σzs = _encoded_mu_vars(m, x)
    if mean
        return [μzs_σzs[i][1]]
    else 
        return [rptrick(μzs_σzs[i]...)]
    end
end
_encode_ith_batched(m::AbstractVLAE, x, i::Int; mean=false, batchsize=128) = 
    _batched_encs(_encode_ith, m, x, batchsize, i; mean=mean)[1]
function _encode_all(m::AbstractVLAE, x; mean=false)
    if mean
        return Tuple(map(y->y[1], _encoded_mu_vars(m, x)))
    else
        return Tuple(map(y->rptrick(y...), _encoded_mu_vars(m, x)))
    end
end

"""
    encode(m::AbstractVLAE, x, [i::Int]; mean=false, batchsize=128)

Encoding in the i-th latent layer. If i is not given, all encodings are returned instead. 
"""
encode(m::AbstractVLAE, x, i::Int; batchsize::Int=128, mean=false) = 
    _encode_ith_batched(m , x, i; batchsize=batchsize, mean=mean)
encode(m::AbstractVLAE, x; batchsize::Int=128, mean=false) =
    _batched_encs(_encode_all, m, x, batchsize; mean=mean)
encode_mean(args...; kwargs...) = encode(args...; mean=true, kwargs...)
encode_all(m::AbstractVLAE, x; kwargs...) =
     encode(m, x; kwargs...)

# decoding functions
function decode(m::AbstractVLAE, zs...)
    # this is for 2D data
    if length(m.xdim) == 1
        μx, σx = _decoded_mu_var(m, zs...)
        return rptrick(μx, σx)
    end
    # this is for images
    if m.xdist == Val(:gaussian)
        μx, σx = _decoded_mu_var1(m, zs...)
        return devectorize(rptrick(μx, σx), m.xdim...)
    else # bernoulli
        return _decoder_out(m, zs...)
    end
end

reconstruct(m::AbstractVLAE, x) = decode(m, encode_all(m, x)...)

function reconstruction_probability(m::AbstractVLAE, x::AbstractArray{T,2}) where T  
    zs = map(y->rptrick(y...), _encoded_mu_vars(m, x))
    return -logpdf(m, x, zs...)
end
function reconstruction_probability(m::AbstractVLAE, x::AbstractArray{T,4}) where T 
    gx = gpu(x)
    zs = map(y->rptrick(y...), _encoded_mu_vars(m, gx))
    return -logpdf(m, gx, zs...)
end
reconstruction_probability(m::AbstractVLAE, x, L::Int) = mean([reconstruction_probability(m,x) for _ in 1:L])
function reconstruction_probability(m::AbstractVLAE, x, L::Int, batchsize::Int)
    if size(x, ndims(x)) == 0
        return Float32[]
    else
        return vcat(map(b->cpu(reconstruction_probability(m, b, L)), Flux.Data.DataLoader(x, batchsize=batchsize))...)
    end
end

function generate(m::AbstractVLAE, n::Int)
    nl = length(m.e)
    zs = [randn(Float32, m.zdim, n) for _ in 1:nl]
    
    if is_gpu(m)
        zs = gpu(zs)
    end
    
    cpu(decode(m, zs...))
end

is_gpu(m::AbstractVLAE) = typeof(m.f[1][2].W) <: Flux.CUDA.CuArray
