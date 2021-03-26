abstract type AbstractVLAE end

# improved elbo
function elbo(m::AbstractVLAE, x::AbstractArray{T,4}) where T
    # encoder pass
    μzs_σzs = _encoded_mu_vars(m, x)
    zs = map(y->rptrick(y...), μzs_σzs)
    kldl = sum(map(y->Flux.mean(kld(y...)), μzs_σzs))
        
    # decoder pass
    μx, σx = _decoded_mu_var(m, zs...)
    _x = (m.var == :dense) ? vectorize(x) : x
        
    -kldl + Flux.mean(logpdf(_x, μx, σx))
end

function _encoded_mu_vars(m::AbstractVLAE, x)
    nl = length(m.e)

    h = x
    mu_vars = map(1:nl) do i
        h = m.e[i](h)
        nch = floor(Int,size(h,3)/2)
        μz, σz = mu_var(m.g[i](h[:,:,(nch+1):end,:]))
        h = h[:,:,1:nch,:]
        (μz, σz)
    end
    mu_vars
end

function _decoded_mu_var(m::AbstractVLAE, zs...)
    nl = length(m.d)
    @assert length(zs) == nl
    
    h = m.f[1](zs[end])
    # now, propagate through the decoder
    for i in 1:nl-1
        h1 = m.d[i](h)
        h2 = m.f[i+1](zs[end-i])
        h = cat(h1, h2, dims=3)
    end
    h = m.d[end](h)
    μx, σx = mu_var1(h)
end

function encode(m::AbstractVLAE, x, i::Int)
    μzs_σzs = _encoded_mu_vars(m, x)
    rptrick(μzs_σzs[i]...)
end
encode(m::AbstractVLAE, x) = encode(m, x, length(m.g))
encode_all(m::AbstractVLAE, x) = Tuple(map(y->rptrick(y...), _encoded_mu_vars(m, x)))
function encode_all(m::AbstractVLAE, x, batchsize::Int)
    encs = map(y->cpu(encode_all(gpu(m), gpu(y))), Flux.Data.DataLoader(x, batchsize=batchsize))
    [cat([y[i] for y in encs]..., dims=2) for i in 1:length(encs[1])]
end

function decode(m::AbstractVLAE, zs...) 
    μx, σx = _decoded_mu_var(m, zs...)
    devectorize(rptrick(μx, σx), m.xdim...)
end

reconstruct(m::AbstractVLAE, x) = decode(m, encode_all(m, x)...)

function reconstruction_probability(m::AbstractVLAE, x)  
    x = gpu(x)
    zs = map(y->rptrick(y...), _encoded_mu_vars(m, x))
    μx, σx = _decoded_mu_var(m, zs...)
    _x = (m.var == :dense) ? vectorize(x) : x
    -logpdf(_x, μx, σx)
end
reconstruction_probability(m::AbstractVLAE, x, L::Int) = mean([reconstruction_probability(m,x) for _ in 1:L])
function reconstruction_probability(m::AbstractVLAE, x, L::Int, batchsize::Int)
    vcat(map(b->cpu(reconstruction_probability(m, b, L)), Flux.Data.DataLoader(x, batchsize=batchsize))...)
end

function generate(m::AbstractVLAE, n::Int)
    nl = length(m.e)
    zs = [randn(Float32, m.zdim, n) for _ in 1:nl]
    
    if is_gpu(m)
        zs = gpu(zs)
    end
    
    cpu(decode(m, zs...))
end

is_gpu(m::AbstractVLAE) = typeof(m.g[1][2].W) <: Flux.CUDA.CuArray
