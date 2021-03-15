struct VLAE
    e
    d
    g # extracts latent variables
    f # concatenates latent vars with the rest
    xdim # (h,w,c)
    zdim # scalar
end

Flux.@functor VLAE
(m::VLAE)(x) = reconstruct(m, x)

function VLAE(zdim::Int, ks, ncs, stride, datasize)
    nl = length(ncs) # no layers
    # this captures the dimensions after each convolution
    sout = Tuple(map(j -> datasize[1:2] .- [sum(map(k->k[i]-1, ks[1:j])) for i in 1:2], 1:length(ncs))) 
    # this is the vec. dimension after each convolution
    #ddim = map(i->prod(sout[i])*ncs[i], 1:length(ncs))
    #ddim_d = copy(ddim)
    #ddim_d[1:end-1] .= ddim_d[1:end-1]/2
    ddim = map(i->floor(Int,prod(sout[i])*ncs[i]/2), 1:length(ncs))
    ddim_d = copy(ddim)
    ddim_d[end] = ddim_d[end]*2
    indim = prod(datasize[1:3])
    rks = reverse(ks)
    rsout = reverse(sout)

    # number of channels for encoder/decoder
#    ncs_in_e = vcat([datasize[3]], [n for n in ncs[1:end-1]])
    ncs_in_e = vcat([datasize[3]], [floor(Int,n/2) for n in ncs[1:end-1]])
    ncs_in_d = reverse(ncs)
    ncs_out_d = vcat([floor(Int,n/2) for n in ncs_in_d[2:end]], [datasize[3]])
    ncs_out_f = vcat([floor(Int,n/2) for n in ncs[1:end-1]], [ncs[end]])
    
    # encoder/decoder
    e = Chain([Conv(ks[i], ncs_in_e[i]=>ncs[i], relu, stride=stride) for i in 1:nl]...)
    d = Chain([ConvTranspose(rks[i], ncs_in_d[i]=>ncs_out_d[i], relu, stride=stride) for i in 1:nl]...,
     x->reshape(x, :, size(x,4)),
     Dense(indim, indim+1)
    )
    
    # latent extractor
    g = Tuple([Chain(x->reshape(x, :, size(x,4)), Dense(ddim[i], zdim*2)) for i in 1:nl])
    
    # latent reshaper
    f = Tuple([Chain(Dense(zdim, ddim_d[i], relu), 
            x->reshape(x, sout[i]..., ncs_out_f[i], size(x,2))) for i in nl:-1:1])

    return VLAE(e,d,g,f,datasize[1:3],zdim)
end

# improved elbo
function elbo(m::VLAE, x::AbstractArray{T,4}) where T
    # encoder pass
    μzs_σzs = _encoded_mu_vars(m, x)
    zs = map(y->rptrick(y...), μzs_σzs)
    kldl = sum(map(y->Flux.mean(kld(y...)), μzs_σzs))
        
    # decoder pass
    μx, σx = _decoded_mu_var(m, zs...)
    vx = vectorize(x)
        
    -kldl + Flux.mean(logpdf(vx, μx, σx))
end

function _encoded_mu_vars(m::VLAE, x)
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

function _decoded_mu_var(m::VLAE, zs...)
    nl = length(m.d) - 2
    @assert length(zs) == nl
    
    h = m.f[1](zs[end])
    # now, propagate through the decoder
    for i in 1:nl-1
        h1 = m.d[i](h)
        h2 = m.f[i+1](zs[end-i])
        h = cat(h1, h2, dims=3)
    end
    h = m.d[nl:end](h)
    μx, σx = mu_var1(h)
end

function train_vlae(zdim, batchsize, ks, ncs, stride, nepochs, data, val_x, tst_x; λ=0.0f0, epochsize = size(data,4))
    gval_x = gpu(val_x[:,:,:,1:min(1000, size(val_x,4))])
    gtst_x = gpu(tst_x)
    
    model = gpu(VLAE(zdim, ks, ncs, stride, size(data)))
    nl = length(model.e)
    
    ps = Flux.params(model)
    loss(x) = -elbo(model,gpu(x)) + λ*sum(l2, ps)
    opt = ADAM()
    rdata = []
    hist = []
    zs = [[] for _ in 1:nl]
    
    # train
    println("Training in progress...")
    for epoch in 1:nepochs
        data_itr = Flux.Data.DataLoader(data[:,:,:,sample(1:size(data,4), epochsize)], batchsize=batchsize)
        Flux.train!(loss, ps, data_itr, opt)
        l = Flux.mean(map(x->loss(x), Flux.Data.DataLoader(val_x, batchsize=batchsize))
        println("Epoch $(epoch)/$(nepochs), validation loss = $l")
        for i in 1:nl
            z = encode(model, gval_x, i)
            push!(zs[i], cpu(z))
        end
        push!(hist, l)
        push!(rdata, cpu(reconstruct(model, gval_x)))
    end
    
    return model, hist, rdata, zs
end

function encode(m::VLAE, x, i::Int)
    μzs_σzs = _encoded_mu_vars(m, x)
    rptrick(μzs_σzs[i]...)
end
encode(m::VLAE, x) = encode(m, x, length(m.g))
encode_all(m::VLAE, x) = map(y->rptrick(y...), _encoded_mu_vars(m, x))

function decode(m::VLAE, zs...) 
    μx, σx = _decoded_mu_var(m, zs...)
    devectorize(rptrick(μx, σx), m.xdim...)
end

reconstruct(m::VLAE, x) = decode(m, encode_all(m, x)...)
