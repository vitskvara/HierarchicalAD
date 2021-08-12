# this is for the metric computation
"""
    MajorityVoteClassifier(x, y)

Construct it using training inputs x and outputs y.
"""
struct MajorityVoteClassifier
    x::Vector
    y::Vector
end

function predict(m::MajorityVoteClassifier, x::Number) 
    yx = m.y[m.x .== x]
    if length(yx) == 0
        return sample(unique(m.y), 1)[1]
    end
    cm = countmap(yx)
    collect(keys(cm))[argmax(collect(values(cm)))]
end
predict(m::MajorityVoteClassifier, x::Vector) = map(_x->predict(m,_x), x)
error_rate(y_true::Vector, y_pred::Vector) = mean(y_pred .!= y_true)
precision(y_true::Vector, y_pred::Vector) = mean(y_pred .== y_true)

function split_pairs(all_pairs, train_ratio=0.5; seed=nothing)
    tr_inds, val_inds, tst_inds = 
        HierarchicalAD.train_val_test_inds(1:length(all_pairs), (train_ratio,0,1-train_ratio); seed=seed)
    tr_x = [_x[1] for _x in all_pairs[tr_inds]]
    tr_y = [_x[2] for _x in all_pairs[tr_inds]]
    tst_x = [_x[1] for _x in all_pairs[tst_inds]]
    tst_y = [_x[2] for _x in all_pairs[tst_inds]]
    (tr_x, tr_y), (tst_x, tst_y)
end

# this is for disentanglement per dim
function get_least_varied_dim(x::AbstractArray{T,2}, sds) where T
    x = x ./ sds
    v = var(x, dims=2)
    argmin(v)[1], v
end

function disentanglement_per_dim(z::AbstractArray{T,2}, y, factors; max_samples=size(z,2)) where T
    # first compute sd per dimension
    sds = map(std, eachrow(z))
    
    # now loop over factors and latents
    all_pairs = []
    for k in 1:length(factors)
        factor = factors[k]
        fvals = unique(y[!,factor])

        k_pairs = []
        for fk in fvals
            zk = z[:,y[!,factor] .== fk]
            ik, vs = get_least_varied_dim(zk, sds)
            push!(k_pairs, (ik, k))
        end
        push!(all_pairs, k_pairs)
    end
    # now this contains the pairs with (x=least varied latent index, y=factor index)
    all_pairs = vcat(all_pairs...)
    
    # here, train and predict using majority voting classifier
    (tr_x, tr_y), (tst_x, tst_y) = split_pairs(all_pairs, 0.5)
    mvc = MajorityVoteClassifier(tr_x, tr_y);
    y_pred = predict(mvc, tst_x)
    precision(y_pred, tst_y)
end
disentanglement_per_dim(z::AbstractArray{T,2}, y, factors, M::Int; kwargs...) where T = 
	mean(map(_->disentanglement_per_dim(z, y, factors; kwargs...), 1:M))
disentanglement_per_dim(z::AbstractArray{T,2}, y::Nothing, args...; kwargs...) where T = nothing

# this is for disentanglement per latent
function get_least_varied_latent(zs::Vector, sds_per_latent) where T
    dcovs = []
    for latenti in 1:length(zs)
        z = zs[latenti]
        nz = z ./ sds_per_latent[latenti]
        push!(dcovs, det(cov(nz, dims=2)))
    end
    argmin(dcovs)[1], dcovs
end

function disentanglement_per_latent(zs::Vector, y, factors; max_samples=size(zs[1],2))
    # first compute sd per dimension
    sds_per_latent = map(z->map(std, eachrow(z)),zs)
    
    # now loop over factors and latents
    all_pairs = []
    for k in 1:length(factors)
        factor = factors[k]
        fvals = unique(y[!,factor])

        k_pairs = []
        for fk in fvals
            zks = []
            for latenti in 1:length(zs)
                push!(zks, zs[latenti][:,y[!,factor] .== fk])
            end
            ik, dcovs = get_least_varied_latent(zks, sds_per_latent)
            push!(k_pairs, (ik, k))
        end
        push!(all_pairs, k_pairs)
    end
    # now this contains the pairs with (x=least varied latent index, y=factor index)
    all_pairs = vcat(all_pairs...)
    
    # here, train and predict using majority voting classifier
    (tr_x, tr_y), (tst_x, tst_y) = split_pairs(all_pairs, 0.5)
    mvc = MajorityVoteClassifier(tr_x, tr_y);
    y_pred = predict(mvc, tst_x)
    precision(y_pred, tst_y)
end
disentanglement_per_latent(zs::Vector, y, factors, M::Int; kwargs...) = 
	mean(map(_->disentanglement_per_latent(zs, y, factors; kwargs...), 1:M))
disentanglement_per_latent(zs::Vector, y::Nothing, args...; kwargs...) = nothing
