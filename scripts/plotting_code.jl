using DrWatson
@quickactivate
using HierarchicalAD
using Plots
using IPMeasures
using HierarchicalAD: draw, reconstruct, decode
using StatsBase
using Measures

function check_reconstructions(model, x, s=(28,28))
    println("Original samples")
    display(draw(x))
    
    println("Reconstructed samples")
    rx = cpu(reconstruct(model, gpu(x)))
    if length(size(rx)) == 2
        display(draw(rx,s))
    else
        display(draw(rx))
    end
end
check_reconstructions(res::Tuple, args...) = check_reconstructions(res[1], args...)

function animate_latent(zs...; fps::Int=5, gf="gifs/zs.gif", dims=(1,2))
    gr(size=(300*length(zs),300))
    nz = length(zs)
    ne = length(zs[1])
    
    anim = @animate for ie in 1:ne
        ps = []
        for iz in 1:nz
            p = scatter(zs[iz][ie][dims[1],:], zs[iz][ie][dims[2],:], legend=false, markersize=4, α=1,
                xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])], epoch $ie")
            push!(ps, p)
        end
        plot(ps...,layout=(1,nz))
    end
    gif(anim, gf, fps=fps)
end
animate_latent(res::Tuple, args...; kwargs...) = animate_latent(res[4], args...; kwargs...)

function animate_latent(cat_vals, cat_ind::Int, val_labels, zs...; fps::Int=5, gf="gifs/zs.gif", dims=(1,2))
    cat_inds = map(d->[l[cat_ind] == d for l in val_labels], cat_vals)
    
    gr(size=(300*length(zs),300))
    nz = length(zs)
    ne = length(zs[1])
    
    anim = @animate for ie in 1:ne
        pls = []
        for iz in 1:nz
            p = scatter(zs[iz][ie][dims[1],cat_inds[1]], zs[iz][ie][dims[2],cat_inds[1]], 
                label="$(cat_vals[1])", markersize=4, α=0.7,
                xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])], epoch $ie")
            for (i,ci) in enumerate(cat_inds[2:end])
                scatter!(zs[iz][ie][dims[1],ci], zs[iz][ie][dims[2],ci], label="$(cat_vals[i+1])", 
                    markersize=4, α=0.7,
                    xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])], epoch $ie")
            end
            push!(pls, p)
        end
        plot(pls...,layout=(1,nz))
    end
    gif(anim, gf, fps=fps)
end
function plot_latent(cat_vals, cat_ind::Int, val_labels, zs...; dims=(1,2))
    cat_inds = map(d->[l[cat_ind] == d for l in val_labels], cat_vals)
    
    gr(size=(200*length(zs),300))
    nz = length(zs)
    ne = length(zs[1])
    
    pls = []
    for iz in 1:nz
        p = scatter(zs[iz][ne][dims[1],cat_inds[1]], zs[iz][ne][dims[2],cat_inds[1]], 
            label="$(cat_vals[1])", markersize=4, α=0.7,
            xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])], epoch $ne")
        for (i,ci) in enumerate(cat_inds[2:end])
            scatter!(zs[iz][ne][dims[1],ci], zs[iz][ne][dims[2],ci], label="$(cat_vals[i+1])", 
                markersize=4, α=0.7,
                xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])], epoch $ne")
        end
        push!(pls, p)
    end
    plot(pls...,layout=(1,nz))
end
function plot_latent(cat_vals, cat_ind::Int, val_labels, k::IPMeasures.AbstractKernel, zs...; dims=(1,2))
    cat_inds = map(d->[l[cat_ind] == d for l in val_labels], cat_vals)
    length(cat_inds) == 2 ? nothing : error("MMD computation not supported for more than 2 classes")
    
    gr(size=(200*length(zs),300))
    nz = length(zs)
    
    pls = []
    for iz in 1:nz
        z1 = zs[iz][:,cat_inds[1]]
        p = scatter(z1[dims[1],:], z1[dims[2],:], 
            label="$(cat_vals[1])", markersize=4, α=0.7,
            xlims=(-3,3),ylims=(-3,3))
        for (i,ci) in enumerate(cat_inds[2:end])
            z2 = zs[iz][:,ci]
            mmd_val = mmd(k, z1, z2)
            scatter!(z2[dims[1],:], z2[dims[2],:], label="$(cat_vals[i+1])", 
                markersize=4, α=0.7,
                xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])],\n MMD=$(round(mmd_val, digits=5))")
        end
        push!(pls, p)
    end
    plot(pls...,layout=(1,nz))
end
function plot_latent(z1s, z2s, labels, k::IPMeasures.AbstractKernel; dims=(1,2))
    gr(size=(200*length(z1s),300))
    nz = length(z1s)
    
    pls = []
    mmds = []
    for iz in 1:nz
        z1 = z1s[iz]
        p = scatter(z1[dims[1],:], z1[dims[2],:], 
            label=labels[1], markersize=4, α=0.7,
            xlims=(-3,3),ylims=(-3,3))
        z2 = z2s[iz]
        mmd_val = mmd(k, z1, z2)
        scatter!(z2[dims[1],:], z2[dims[2],:], label=labels[2], 
                markersize=4, α=0.7,
                xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])],\n MMD=$(round(mmd_val, digits=5))")
        push!(pls, p)
    end
    plot(pls...,layout=(1,nz))
end
# newer version
function plot_latent(cat_vals, category::Symbol, labels::DataFrame, zs...; dims=(1,2))
    cat_inds = map(d->labels[!,category] .== d, cat_vals)
        
    gr(size=(300*length(zs),300))
    nz = length(zs)
    
    pls = []
    for iz in 1:nz
        p = scatter(zs[iz][dims[1],cat_inds[1]], zs[iz][dims[2],cat_inds[1]], 
            label="$(cat_vals[1])", markersize=4, α=0.5,
            xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])]")
        for (i,ci) in enumerate(cat_inds[2:end])
            scatter!(zs[iz][dims[1],ci], zs[iz][dims[2],ci], label="$(cat_vals[i+1])", 
                markersize=4, α=0.5,
                xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])]")
        end
        push!(pls, p)
    end
    plot(pls...,layout=(1,nz))
end
function plot_latent(cat_vals, category::Symbol, labels::DataFrame, k::IPMeasures.AbstractKernel, 
        zs...; dims=(1,2))
    cat_inds = map(d->labels[!,category] .== d, cat_vals)
    
    gr(size=(300*length(zs),300))
    nz = length(zs)
    pmmds = map(i->pairwise_mmd(k, cat_vals, category, labels, zs[i]), 1:nz)
    
    pls = []
    for iz in 1:nz
        p = scatter(zs[iz][dims[1],cat_inds[1]], zs[iz][dims[2],cat_inds[1]], 
            label="$(cat_vals[1])", markersize=4, α=0.5,
            xlims=(-3,3),ylims=(-3,3),
            title="z$(iz)[$(dims[1]), $(dims[2])], $(category),\n mean MMD=$(round(mean(pmmds[iz]),digits=3))",
            topmargin = 5mm)
        for (i,ci) in enumerate(cat_inds[2:end])
            scatter!(zs[iz][dims[1],ci], zs[iz][dims[2],ci], label="$(cat_vals[i+1])", 
                markersize=4, α=0.5,
                xlims=(-3,3),ylims=(-3,3))
        end
        push!(pls, p)
    end
    plot(pls...,layout=(1,nz)), pmmds
end

function plot_latent_per_dim(z1s, z2s, labels, k::IPMeasures.AbstractKernel)
    gr(size=(200*length(z1s),500))
    nz = length(z1s)
    nd = size(z1s[1],1)
    
    pls = []
    for iz in 1:nz
        for id in 1:nd
            z1 = z1s[iz][id:id,:]
            p = histogram(vec(z1), label=labels[1], α=0.7, normalize = true)
            z2 = z2s[iz][id:id,:]
            mmd_val = mmd(k, z1, z2)
            histogram!(vec(z2), label=labels[2], α=0.7, normalize = true,
                title="z[$(iz),$(id)],\n MMD=$(round(mmd_val, digits=5))")
            push!(pls, p)
        end
    end
    plot(pls...,layout=(nz,2))
end

function animate_reconstructions(rdata, val_x, inds; fps=5, gf="gifs/reconstructions.gif")
    @assert length(inds) == 10
    gr(size=(800,300))
    anim = @animate for i in 1:length(rdata)
        rd = rdata[i][:,:,1,inds]
        vx = val_x[:,:,1,inds]
        p=heatmap(vcat(
                hcat([vx[end:-1:1,:,i] for i in 1:5]...),
                hcat([rd[end:-1:1,:,i] for i in 1:5]...),
                hcat([vx[end:-1:1,:,i] for i in 6:10]...),
                hcat([rd[end:-1:1,:,i] for i in 6:10]...)),
            title = "epoch $i"
        )
        plot(p)
    end
    gif(anim, gf, fps=fps)
end
animate_reconstructions(res::Tuple, args...; kwargs...) = animate_reconstructions(res[2], args...; kwargs...)

function plot_training(hist)
    gr(size=(400,200))
    plot(hist, xlabel="epochs", ylabel="loss", label="$(hist[end])")
end
plot_training(res::Tuple) = plot_training(res[2])

function explore_latent(m::VLAE, i)
    nl = length(m.e)
    rows = []
    for xx in -3:0.5:3.0
        row = []
        for yy in 3:-0.5:-3.0
            zs = map(1:nl) do j
                if j == i
                    z = gpu(vcat([xx;yy], randn(m.zdim-2,1)))
                else
                    z = gpu(randn(m.zdim,1))
                end
                z
            end
            push!(row, cpu(decode(m, zs...)))
        end
        push!(rows, hcat(row...))
    end
    rows
end

function explore_one_sample(ind, ws, test_labels, test_data, zns, test_zs; kwargs...)
    wzs = map(w->w[ind], ws)
    println(wzs)
    println(test_labels[ind])
    draw(test_data[:,:,:,ind])
    
    ps = []
    for i in 1:length(zns)
        p=scatter(zns[i][1,:],zns[i][2,:], title="$(wzs[i])"; kwargs...)
        scatter!(test_zs[i][1,ind:ind], test_zs[i][2,ind:ind]; kwargs...)
        push!(ps, p)
    end
    plot(ps..., layout=(1,length(zns)))
end

function pairwise_mmd(k, groups...)
    ngroups = length(groups)
    dist_mat = zeros(Float32, ngroups, ngroups)
    for i in 1:ngroups-1
        for j in i+1:ngroups
            dist_mat[i,j] = dist_mat[j,i] = mmd(k, groups[i], groups[j])
        end
    end
    dist_mat
end
function pairwise_mmd(k, cat_vals, category, labels::DataFrame, data)
    cat_inds = map(d->labels[!,category] .== d, cat_vals)
    groups = map(is->data[:,is], cat_inds)
    pairwise_mmd(k, groups...)
end
function discretize_category(labels, category, nbins)
    limits = collect(range(minimum(labels[!,category]), maximum(labels[!,category])+ 1e-5, length = nbins+1))
    bins = [(limits[i],limits[i+1]) for i in 1:nbins]
    bininds = map(i->findfirst(map(b->b[1] <= labels[i,category] < b[2], bins)), 1:length(labels[!,category]));
    temp_labels = DataFrame()
    temp_labels[!,category] = bininds
    temp_labels
end
StatsBase.sample(x::Array{T,4}, n::Int; kwargs...) where T = x[:,:,:,sample(1:size(x,4), n; kwargs...)]
StatsBase.sample(x::Array{T,2}, n::Int; kwargs...) where T = x[:,sample(1:size(x,2), n; kwargs...)]