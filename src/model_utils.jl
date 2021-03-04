softplus_safe(x) = softplus(x) .+ 0.00001f0 
rptrick(μ::Flux.CUDA.CuArray, σ2) = μ .+ gpu(randn(Float32,size(μ))) .* σ2
rptrick(μ, σ2) = μ .+ randn(Float32,size(μ)) .* σ2

kld(μ::AbstractArray{T,N}, σ2::AbstractArray{T,N}) where T where N = 
    T(1/2)*sum(σ2 + μ.^2 - log.(σ2) .- T(1.0), dims = 1)
kld(μ::AbstractArray{T,N}, σ2::T) where T where N = 
    T(1/2)*sum(σ2 .+ μ.^2 .- log(σ2) .- T(1.0), dims = 1)

logpdf(x::AbstractArray{T,N}, μ::AbstractArray{T,N}, σ2::AbstractArray{T,N}) where T where N = 
    - vec(sum(((x - μ).^2) ./ σ2 .+ log.(σ2), dims=1)) .+ size(x,1)*log(T(2π)) / T(2)
logpdf(x::AbstractArray{T,N}, μ::AbstractArray{T,N}, σ2::T) where T where N = 
    - vec(sum(((x - μ).^2) ./ σ2 .+ log(σ2), dims=1)) .+ size(x,1)*log(T(2π)) / T(2)

function mu_var(x)
    N = floor(Int, size(x,1)/2)
    μ = x[1:N,:]
    σ = softplus_safe.(x[N+1:end,:])
    return μ, σ
end

function mu_var1(x)
    μ = x[1:end-1,:]
    σ = softplus_safe.(x[end:end,:])
    return μ, σ
end

l2(x) = sum(x.^2)
