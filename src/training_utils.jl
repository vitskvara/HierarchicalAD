function timetag()
    n = string(now())
    reduce(replace, ["-" => "", ":" => "", "T" => "", "." => ""]; init=n)
end

function save_model(res, modelpath, val_labels)
    id = timetag()
    f = joinpath(modelpath, "$(id)_layers-$(length(res[1].e)).bson")
    save(f, 
        Dict(:model => cpu(res[1]), :hist => res[2], 
        :rdata => res[3], :zs => res[4], :val_labels => val_labels))
    f
end

function load_model(id, modelpath)
    f = filter(x->occursin("$id", x), readdir(modelpath))[1]
    modeldata = load(joinpath(modelpath,f))
    return gpu(modeldata[:model]), modeldata[:hist], modeldata[:rdata], modeldata[:zs], modeldata[:val_labels]
end
function load_model_gpu(id, modelpath)
    res = load_model(id, modelpath)
    return gpu(res[1]), res[2], res[3], res[4], res[5]
end

sample_tensor(x::AbstractArray{T,4}, n::Int; kwargs...) where T = x[:,:,:,sample(1:size(x,4), n; kwargs...)]
sample_tensor(x::AbstractArray{T,2}, n::Int; kwargs...) where T = x[:,sample(1:size(x,2), n; kwargs...)]

function batched_loss(lf, x, batchsize)
    @suppress begin # suppress the batchsize warning
       return Flux.mean(map(y->cpu(lf(y)), Flux.Data.DataLoader(x, batchsize=batchsize)))
    end
end

"""
    restart_weights!(model, init = Flux.glorot_uniform) 

Restarts the weights of the model using given initialization method.
"""
function restart_weights!(model, init = Flux.glorot_uniform) 
    ps = params(model)
    for p in ps
        p .= init(size(p)...)
    end
    model
end

"""
    normalization_coefficients(x::Matrix)

Row-wise normalization coefficients.
"""
normalization_coefficients(x::AbstractArray{T,2}) where T = maximum(x, dims=2), minimum(x, dims=2)

"""
    normalize(x::Matrix[, maxs, mins])

Row-wise normalization of x. If maxima/minima are not supported, they are returned as well.
"""
normalize(x::AbstractArray{T,2}, maxs, mins) where T = (x .- mins) ./ (maxs - mins)
function normalize(x::AbstractArray{T,2}) where T
    maxs, mins = normalization_coeffs(x)
    return normalize(x, maxs, mins), maxs, mins
end
sqnorm(x) = sum(abs2, x)