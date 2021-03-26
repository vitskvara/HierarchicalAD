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

function compare_reconstructions(model, x; plotsize=(1000,300))
    gr(size=plotsize)
    N = size(x,4)
    p1 = heatmap(cat(map(i->x[end:-1:1,:,1,i],1:N)..., dims=2), title="original digits")
    r_x = cpu(reconstruct(model, gpu(x)))
    p2 = heatmap(cat(map(i->r_x[end:-1:1,:,1,i],1:N)..., dims=2), title="reconstructed digits")
    g_x = HierarchicalAD.generate(model, N)
    p3 = heatmap(cat(map(i->g_x[end:-1:1,:,1,i],1:N)..., dims=2), title="generated digits")
    plot(p1, p2, p3, layout = (3,1))
end

function plot_latents(model)
    ps = []
    nl = length(model.e)
    for i in 1:nl
        x = explore_latent(model, i)
        x = cat([z[end:-1:1,:,1,1] for z in x]..., dims=1)
        push!(ps, heatmap(x,title = "latent $i", color=:gist_gray))
    end
    plot(ps..., layout=(nl,1),size=(1500,600*nl))
end

function plot_anomalies_digit(tr_y, a_y, tr_scores, val_scores, a_scores)
    n_digits = sort(unique(tr_y[!,:digit]))
    a_digits = filter(x->!(x in n_digits), sort(unique(a_y[!,:digit])));
    a_inds = [l in a_digits for l in a_y[!,:digit]]
    ps = []
    p=plot(tr_scores,seriestype=:stephist, title="digit_anomalies", normalize=true, lw=3, label="normal - train")
    plot!(val_scores,seriestype=:stephist, normalize=true, lw=3, label="normal - validation")
    plot!(a_scores[a_inds],seriestype=:stephist, normalize=true, lw=3, label="anomalous")
    push!(ps, p)
    for d in a_digits
        a_inds_d = [l == d for l in a_y[!,:digit]]
        p=plot(tr_scores,seriestype=:stephist, title="digit_anomalies", normalize=true, lw=3, label="normal")
        plot!(a_scores[a_inds],seriestype=:stephist, normalize=true, lw=3, label="anomalous", alpha=0.3)
        plot!(a_scores[a_inds_d],seriestype=:stephist, normalize=true, lw=3, label="anomalous - $d")
        push!(ps, p)
    end
    plot(ps..., layout = (length(a_digits)+1,1), size=(500,200*(length(a_digits)+1)))
end

function plot_anomalies_other(non_default_filters, filter_dict, a_y, tr_scores, a_scores)
    ps = []
    for factor in filter(f->f!="digit", non_default_filters)
        labels = a_y[!,Symbol(factor)]
        f = filter_dict[factor]
        ff(x) = eval(Meta.parse("!($x $(f[1]) $(f[2]))"))
        a_inds = map(ff,  labels)
        p=plot(tr_scores,seriestype=:stephist, title="$factor anomalies", normalize=true, lw=3, label="normal ($(f[1]) $(f[2]))")
        plot!(a_scores[a_inds],seriestype=:stephist, normalize=true, lw=3, label="anomalous")
        push!(ps, p)
    end
    nl = length(non_default_filters)-1
    plot(ps..., layout=(nl,1), size=(500,300*nl))
end

function print_mmd_overview(category, vals, mmds)
    println("\n")
    println("Training latent encodings split by $category ($(vals)) identity:")
    for i in 1:length(mmds)
        mmd = mmds[i]
        println("  Mean MMD[$i] = $(mean(mmd))")
    end
end

function print_mmd_overview_anomalies(mmd_dict)
    println("\n")
    println("Normal and anomalous encodings by category:")
    for (k,v) in pairs(mmd_dict)
        println("  $k")
        for (i,val) in enumerate(v)
            println("    Mean MMD[$i] = $(val)")
        end
    end
end

function mmd_overview_other(nbins, non_default_filters, tr_y, tr_encodings, k, model)
    ps = []
    mmd_dict = Dict()
    nbins = 5
    for category in filter(f->f!="digit", non_default_filters)
        temp_labels, bins = discretize_category(tr_y, category, nbins)
        bins = map(x->round.(x, digits=4), bins)
        cat_vals = sort(unique(temp_labels[!,category]))
        p,mmds=plot_latent(cat_vals, Symbol(category), temp_labels, k, tr_encodings..., dims=[1,2])
        print_mmd_overview(category, bins, mmds)
        push!(ps,p)
        mmd_dict[category] = mmds
    end
    nl = length(model.e)
    nf = length(non_default_filters)-1
    p = plot(ps..., layout=(nf,1), size=(300*nl, 400*nf))
    p, mmd_dict
end

function plot_latent_anomalies(n_data, a_data, category, k)
    ps = []
    mmds = map(x->mmd(k,x[1],x[2]), zip(n_data,a_data))
    nl = length(a_data)
    for (mmd, n_z, a_z) in zip(mmds, n_data, a_data)
        p=scatter(n_z[1,:], n_z[2,:], alpha=1, markersize=2, markerstrokewidth=0,
                xlims=(-3,3),ylims=(-3,3),
                title="$(category), MMD=$(round(mean(mmd),digits=3))",
                topmargin = 5mm, label="normal")
        scatter!(a_z[1,:], a_z[2,:], alpha=1, markersize=2, markerstrokewidth=0,
                xlims=(-3,3),ylims=(-3,3), label="anomalous")
        push!(ps, p)
    end
    p=plot(ps..., layout=(1,nl), size=(500*nl,500))
    p, mmds
end

function mmd_overview_anomalies(tr_y, a_y, tr_encodings, a_encodings, non_default_filters, 
    k, filter_dict)
    ps = []
    mmd_dict = Dict()

    # digit
    category = :digit
    n_digits = sort(unique(tr_y[!,:digit]))
    a_digits = filter(x->!(x in n_digits), sort(unique(a_y[!,:digit])));
    a_inds = [l in a_digits for l in a_y[!,:digit]]
    n_data = tr_encodings
    a_data = map(x->x[:,a_inds],a_encodings)
    p,mmd=plot_latent_anomalies(n_data, a_data, category, k)
    push!(ps, p)
    mmd_dict["$(category)_anomalies"] = mmd

    # other factors
    for factor in filter(f->f!="digit", non_default_filters)
        f = filter_dict[factor]
        ff(x) = eval(Meta.parse("!($x $(f[1]) $(f[2]))"))
        a_inds = map(ff, a_y[!,Symbol(factor)])
        n_data = tr_encodings
        a_data = map(x->x[:,a_inds],a_encodings)

        p, mmds = plot_latent_anomalies(n_data, a_data, "$factor $(f[1]) $(f[2])", k)
        push!(ps, p)
        mmd_dict["$(factor)_anomalies"] = mmds
    end

    # plot
    plot(ps..., layout=(length(ps),1), size=(500*length(n_data),500*length(ps))), mmd_dict
end









######### OLD STUFF
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
            label="$(cat_vals[1])", markersize=2, markerstrokewidth=0, α=0.5,
            xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])]")
        for (i,ci) in enumerate(cat_inds[2:end])
            scatter!(zs[iz][dims[1],ci], zs[iz][dims[2],ci], label="$(cat_vals[i+1])", 
                markersize=2, α=0.5, markerstrokewidth=0,
                xlims=(-3,3),ylims=(-3,3),title="z$(iz)[$(dims[1]), $(dims[2])]")
        end
        push!(pls, p)
    end
    plot(pls...,layout=(1,nz))
end
function plot_latent(cat_vals, category::Symbol, labels::DataFrame, k::IPMeasures.AbstractKernel, 
        zs...; dims=(1,2))
    cat_inds = map(d->labels[!,category] .== d, cat_vals)
    
    nz = length(zs)
    pmmds = map(i->pairwise_mmd(k, cat_vals, category, labels, zs[i]), 1:nz)
    
    pls = []
    for iz in 1:nz
        p = scatter(zs[iz][dims[1],cat_inds[1]], zs[iz][dims[2],cat_inds[1]], 
            label="$(cat_vals[1])", markersize=2, α=0.5, markerstrokewidth=0,
            xlims=(-3,3),ylims=(-3,3),
            title="z$(iz)[$(dims[1]), $(dims[2])], $(category),\n mean MMD=$(round(mean(pmmds[iz]),digits=3))",
            topmargin = 5mm)
        for (i,ci) in enumerate(cat_inds[2:end])
            scatter!(zs[iz][dims[1],ci], zs[iz][dims[2],ci], label="$(cat_vals[i+1])", 
                markersize=2, α=0.5, markerstrokewidth=0,
                xlims=(-3,3),ylims=(-3,3))
        end
        push!(pls, p)
    end
    plot(pls...,layout=(1,nz),size=(500*length(zs),500)), pmmds
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

function plot_training(hist; plotsize=(400,200))
    gr(size=plotsize)
    ks = collect(keys(hist))
    nk = length(ks)
    ps = []
    for (ik,k) in enumerate(ks)
        vals = get(hist, k)
        push!(ps, plot(vals..., xlabel="epochs", ylabel=k, label="$k = $(vals[2][end])"))
    end
    plot(ps..., layout=(nk,1), size=(400, 300*nk))
end
plot_training(res::Tuple) = plot_training(res[2])

function explore_latent(m::AbstractVLAE, i)
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
    temp_labels, bins
end
StatsBase.sample(x::Array{T,4}, n::Int; kwargs...) where T = x[:,:,:,sample(1:size(x,4), n; kwargs...)]
StatsBase.sample(x::Array{T,2}, n::Int; kwargs...) where T = x[:,sample(1:size(x,2), n; kwargs...)]