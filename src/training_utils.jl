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

batched_loss(lf, x, batchsize) = 
    Flux.mean(map(y->cpu(lf(y)), Flux.Data.DataLoader(x, batchsize=batchsize)))