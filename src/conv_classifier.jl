function create_input_data(X, y, labels, nntr::Int, nnval::Int, natr::Int, naval::Int, ntst::Int; 
		seed=nothing)
    # create anomalous and normal splits
    nX = X[:,:,:,labels]
    aX = X[:,:,:,.!labels];
    labels = Int.(labels);
    
    # fix the seed
    isnothing(seed) ? nothing : Random.seed!(seed)

    # sample normal data
    nn = size(nX,4)
    randinds = sample(1:nn, nn, replace=false);
    tr_nX = nX[:,:,:,randinds[1:nntr]]
    val_nX = nX[:,:,:,randinds[nntr+1:nntr+nnval]]
    tst_nX = nX[:,:,:,randinds[nntr+nnval+1:end]];
    
    # sample anomalous data
    an = size(aX, 4)
    randinds = sample(1:an, an, replace=false);
    tr_aX = aX[:,:,:,randinds[1:natr]]
    val_aX = aX[:,:,:,randinds[natr+1:natr+naval]]
    tst_aX = aX[:,:,:,randinds[natr+naval+1:end]];
    
    # mixed data
    tr_X = cat(tr_nX, tr_aX, dims=4)
    tr_y = cat(zeros(nntr), ones(natr), dims=1)
    val_X = cat(val_nX, val_aX, dims=4)
    val_y = cat(zeros(nnval), ones(naval), dims=1)
    tst_X = cat(sample_tensor(tst_nX, ntst), sample_tensor(tst_aX, ntst), dims=4)
    tst_y = cat(zeros(ntst), ones(ntst), dims=1)

    # restart seed
    Random.seed!()
    
    return tr_nX, (tr_X, tr_y), (val_X, val_y), (tst_X, tst_y)
end

function construct_conv_classifier(ks, activation, strd, ncs, datasize)
	af = eval(Meta.parse(activation))
    inncs = vcat([datasize[3]], ncs[1:end-1])
    classifier_conv_part = Chain([Conv(k, inc => outc, af) for (k, inc, outc) in zip(ks, inncs, ncs)]...)
    odims = outdims(classifier_conv_part, datasize)
    odim = reduce(*, odims[1:3])
    gpu(Chain(classifier_conv_part..., x->reshape(x, odim, size(x,4)),  Dense(odim, 2)))
end

function train_conv_classifier!(classifier, tr_X, tr_y, val_X, val_y; 
        lr = 0.001f0, 
        λ = 0f0,
        batchsize = 64,
        nepochs = 1000,
        patience = 50,
        verb = true
    )
    
    tr_classifier = deepcopy(classifier)
    opt = ADAM(lr)
    ps = params(tr_classifier)
    loss(x,y) = Flux.Losses.logitcrossentropy(tr_classifier(gpu(x)), gpu(y)) + λ*sum(sqnorm, ps)
    loss(x) = loss(x...)
    oh_tr_y = Flux.onehotbatch(Bool.(tr_y), 0:1)
    oh_val_y = Flux.onehotbatch(Bool.(val_y), 0:1)

    local data_itr
    @suppress begin
        data_itr = Flux.Data.DataLoader((tr_X, oh_tr_y), batchsize=batchsize)
    end
    
    _val_auc = 0
    _patience = 1
    history = MVHistory()
    @info "Starting convolutional classifier training..."
    for epoch in 1:nepochs
        for (x, y) in data_itr
            param_update!(loss, ps, (x, y), opt)
        end
        local tr_l, val_l, tr_auc, val_auc
        @suppress begin
			tr_l = mean(map(x->cpu(loss(x)), Flux.Data.DataLoader((tr_X, oh_tr_y), batchsize=batchsize)))
	        val_l = mean(map(x->cpu(loss(x)), Flux.Data.DataLoader((val_X, oh_val_y), batchsize=batchsize)))
            tr_auc = auc_val(tr_y, cpu(classifier_score(tr_classifier, tr_X, batchsize)))
	        val_auc = auc_val(val_y, cpu(classifier_score(tr_classifier, val_X, batchsize)))
	    end
        if verb
            println("Epoch $epoch: loss = $(tr_l), tr AUC = $(tr_auc), val AUC = $(val_auc)")
        end
        push!(history, :tr_loss, epoch, tr_l)
        push!(history, :val_loss, epoch, val_l)
        push!(history, :tr_auc, epoch, tr_auc)
        push!(history, :val_auc, epoch, val_auc)

        if (val_auc <= _val_auc)
            if _patience >= patience
                if verb
                    @info "Stopping early after $epoch epochs."
                end
                return classifier, history, opt
            else
                _patience += 1
            end
        else
        	classifier = deepcopy(tr_classifier)
            _val_auc = val_auc
        end
    end
    @info "Finished."
    
    return classifier, history, opt
end