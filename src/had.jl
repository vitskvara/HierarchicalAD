AbstractAE = Union{AbstractVLAE,} # add more here
Base.Dict(nt::NamedTuple) = Dict(pairs(nt))
# for show functions
function trim(s::String)
    if length(s) > 70
        return "$(s[1:67])..."
    else
        return s
    end
end

### represents the outputs of training ###
mutable struct TrainingOutputs{AE<:Dict, D<:Dict, C<:Dict}
    autoencoder::AE
    detectors::D
    classifier::C
end
function Base.show(io::IO, to::TrainingOutputs)
    msg = """autoencoder = 
        """
    ae_outs = to.autoencoder
    for (k, v) in pairs(ae_outs)
        v = trim(repr(v))
        msg = msg*"""    $k = $v
        """ 
    end

    msg = msg*"""detectors = 
        """
    d_outs = to.detectors
    for (k, v) in pairs(d_outs)
        v = trim(repr(v))
        msg = msg*"""    $k = $v
        """ 
    end

    msg = msg*"""classifier = 
        """
    c_outs = to.classifier
    for (k, v) in pairs(c_outs)
        v = trim(repr(v))
        msg = msg*"""    $k = $v
        """ 
    end

    print(io, msg)
end

### represents the input parameters ###
mutable struct TrainingParameters{AE<:Dict, D<:Dict, C<:Dict}
    autoencoder::AE
    detectors::D
    classifier::C
end
function Base.show(io::IO, tp::TrainingParameters)
    msg = """autoencoder = 
        """
    ae_outs = tp.autoencoder
    for (k, v) in pairs(ae_outs)
       msg = msg*"""    $k = $v
        """ 
    end
    
    msg = msg*"""detectors = 
        """
    d_outs = tp.detectors
    for (k, v) in pairs(d_outs)
       msg = msg*"""    $k = $v
        """ 
    end
    
    msg = msg*"""classifier = 
        """
    c_outs = tp.classifier
    for (k, v) in pairs(c_outs)
       msg = msg*"""    $k = $v
        """ 
    end

    print(io, msg)
end

TrainingOutputs() = TrainingOutputs(Dict(), Dict(), Dict())
TrainingParameters() = TrainingParameters(Dict(), Dict(), Dict())

### Hierarchical Anomaly Detector ###

mutable struct HAD{AE<:AbstractAE,DS<:Vector,CL,PS<:TrainingParameters,TO<:TrainingOutputs}
    autoencoder::AE
    detectors::DS
    classifier::CL
    parameters::PS
    training_output::TO
end
function Base.show(io::IO, m::HAD)
    nl = length(m.detectors)
    ae = repr(m.autoencoder)
    cl = repr(m.classifier)
    d = repr(m.detectors[1])
    outs = repr(m.training_output)
    
    msg = """Hierarchical Anomaly Detector:
    
     autoencoder        = 
        $(ae)
     detectors          =  
        $(nl) x $d
     classifier          =  
        $(cl)
     
     training_output    = 
    $outs
    """
    
    # now parse the output
    print(io, msg)
end

"""
    HAD(
        n_latents::Int, 
        autoencoder_params, 
        autoencoder_constructor, 
        detector_constructor, 
        detector_params,
        classifier_params,
        [classifier]
        )

Params must a a Dict/NamedTuple. Default classifier is just a single Dense layer.
"""
HAD(nl::Int, ap, ac, dp, dc, cp, c) = HAD(
        ac(;ap...), 
        [dc(;dp...) for _ in 1:nl],
        c,
        TrainingParameters(Dict(ap), Dict(dp), Dict(cp)),
        TrainingOutputs()
        )
HAD(nl::Int, ap, ac, dp, dc, cp) = HAD(nl, ap, ac, dp, dc, cp, Dense(nl+1, 2))

# fvlae parameters
"""
zdim, 
hdims, 
batchsize, 
discriminator_hdim, 
nepochs,
λ=0.0f0, γ=1.0f0, epochsize = size(tr_x,2), layer_depth=1, lr=0.001f0,
activation="relu", discriminator_nlayers=3,
initial_convergence_threshold=0, initial_convergence_epochs=10,
max_retrain_tries=10
"""
fvlae_constructor(;zdim::Int=1, hdims=(1,), discriminator_hdim=1, datasize=(1,1), kwargs...) =
    FVLAE(zdim, hdims, discriminator_hdim, datasize; kwargs...)

# knn parameters
"""
k::Int=1 - number of neighbors
v::Symbol=:kappa - distance computation method, one of [:kappa, :gamma, :delta]
t=:BruteTree - tree that encodes training data, one of [:BruteTree, :KDTree, :BallTree]
"""

### FIT METHODS ###
"""
	fit_autoencoder!(model::HAD, tr_x::AbstractArray, val_x::AbstractArray)
"""
function fit_autoencoder!(model::HAD, tr_x::AbstractArray, val_x::AbstractArray)
    @info "############################# \nStarting autoencoder training"
    
    # train fvlae
    ae, hist, rdata, zs, aeopt, copt = train_fvlae(
        model.parameters.autoencoder[:zdim],
        model.parameters.autoencoder[:hdims],
        model.parameters.autoencoder[:batchsize],
        model.parameters.autoencoder[:discriminator_hdim],
        model.parameters.autoencoder[:nepochs],
        tr_x,
        val_x;
        model.parameters.autoencoder...
    )
    println("Training values at the end of training:")
	for key in keys(hist)
	    v = get(hist, key)[2][end]
	    println("    $key = $v")
	end

    # now assign the right fields
    @info "Autoencoder training finished. \n#############################\n"
    model.autoencoder = ae
    model.training_output.autoencoder = Dict((history = hist, reconstructions = rdata, latents = zs)) # opts?
end

"""
	fit_detectors!(model::HAD, tr_x::AbstractArray)
"""
function fit_detectors!(model::HAD, tr_x::AbstractArray)
    @info "############################# \nStarting detector training"
    
    # encode the data
    tr_e = encode_all(model.autoencoder, tr_x; batchsize=model.parameters.autoencoder[:batchsize], mean=true)
    
    # fit the detectors
    map(x->fit!(x[1], x[2]), zip(model.detectors, tr_e))
    
    # now assign the right fields
    @info "Detector training finished. \n#############################\n"
end

"""
	train_classifier!(classifier, tr_x, tr_y, val_x, val_y; 
        verb = true, λ=0f0, batchsize=128, lr=0.001f0, nepochs=500, 
        patience = 10, val_ratio=0.2)
"""
function train_classifier!(classifier, tr_x, tr_y, val_x, val_y; 
        verb = true, λ=0f0, batchsize=128, lr=0.001f0, nepochs=500, 
        patience = 10, val_ratio=0.2, kwargs...)
    opt = ADAM(lr)
    ps = params(classifier)
    loss(x,y) = Flux.Losses.logitcrossentropy(classifier(x), y) + λ*sum(sqnorm, ps)
    loss(x) = loss(x...)
    
    local data_itr
    @suppress begin
        data_itr = Flux.Data.DataLoader((tr_x, tr_y), batchsize=batchsize)
    end
    
    _val_auc = 0
    _patience = 1
    history = MVHistory()
    for epoch in 1:nepochs
        for (x, y) in data_itr
            param_update!(loss, ps, (x, y), opt)
        end
        tr_l = loss(tr_x, tr_y)
        val_l = loss(val_x, val_y)
        tr_auc = auc_val(Flux.onecold(tr_y) .-1, classifier_score(classifier, tr_x))
        val_auc = auc_val(Flux.onecold(val_y) .-1, classifier_score(classifier, val_x))
        if verb
            println("Epoch $epoch: loss = $(tr_l), tr AUC = $(tr_auc), val AUC = $(val_auc)")
        end
        push!(history, :tr_loss, epoch, tr_l)
        push!(history, :val_loss, epoch, val_l)
        push!(history, :tr_auc, epoch, tr_auc)
        push!(history, :val_auc, epoch, val_auc)
            
        if (val_auc < _val_auc)
            if _patience >= patience
                if verb
                    @info "Stopping early after $epoch epochs."
                end
                return history
            else
                _patience += 1
            end
        else
            _val_auc = val_auc
            #_patience = 0
        end
    end
        
    return history
end

"""
	fit_classifier!(model::HAD, val_x::AbstractArray, val_y::Vector; n_candidates = 10)
"""
function fit_classifier!(model::HAD, val_x::AbstractArray, val_y::Vector)
    @info "############################# \nStarting classifier training"
    
    # create and scale inputs
    vr = model.parameters.classifier[:val_ratio]
    _, (_tr_x, _tr_y), (_val_x, _val_y) = train_val_test_split(val_x[:,val_y.==0], val_x[:,val_y.==1],
        (0, 1-vr, vr))
    cl_tr_x, cl_tr_y = classifier_inputs(model, _tr_x, _tr_y)
    cl_val_x, cl_val_y = classifier_inputs(model, _val_x, _val_y)
    norm_coeffs = normalization_coefficients(hcat(cl_tr_x, cl_val_x))
    cl_tr_x = normalize(cl_tr_x, norm_coeffs...)
    cl_val_x = normalize(cl_val_x, norm_coeffs...)
    model.training_output.classifier[:normalization_coefficients] = norm_coeffs
    
    # train it
    # try out more classifiers
    n_candidates = model.parameters.classifier[:n_candidates]
    candidates = [deepcopy(restart_weights!(model.classifier)) for _ in 1:n_candidates]
    histories = []
    for (i,candidate) in enumerate(candidates)
        history = train_classifier!(candidate, cl_tr_x, cl_tr_y, cl_val_x, cl_val_y; 
            verb = false, model.parameters.classifier...)
        push!(histories, history)
        val_auc = get(history[:val_auc])[2][end]
        @info "Trained classifier $i/$(n_candidates), val AUC = $(val_auc)"
    end

    # now assign the right fields
    # select the best one
    best_candidate_ind = argmax([get(history[:val_auc])[2][end] for history in histories])
    model.classifier = candidates[best_candidate_ind]
    model.training_output.classifier[:history] = histories[best_candidate_ind]
    
    @info "Selected classifier $(best_candidate_ind)"
    
    @info "Classifier training finished. \n#############################\n"
end

"""
	StatsBase.fit!(model::HAD, tr_x::AbstractArray, val_x::AbstractArray)
"""
function StatsBase.fit!(model::HAD, tr_x::AbstractArray, val_x::AbstractArray, val_y::Vector)
    fit_autoencoder!(model, tr_x, val_x)
    fit_detectors!(model, tr_x)
    fit_classifier!(model, val_x, val_y)
end
"""
	StatsBase.predict(model::HAD, x::AbstractArray; L=10)
"""
function StatsBase.predict(model::HAD, x::AbstractArray; L=10)
    cl_x = classifier_inputs(model, x; L=L)
    cl_x = normalize(cl_x, model.training_output.classifier[:normalization_coefficients]...)
    classifier_score(model.classifier, cl_x)
end

### UTILS ###
"""
	ensemble_scores(detectors, encodings)

Scores of the ensemble of detectors.
"""
function ensemble_scores(detectors, x)
    scores = map(_x->predict(_x[1], _x[2]), zip(detectors, x))
end

"""
	all_scores(model::HAD, x; L::Int=10)

Detector + AE rec. probability, L is number of samples in the computation of rec. probability.
"""
function all_scores(model::HAD, x; L::Int=10)
    rec_score = reconstruction_probability(model.autoencoder, x, 10)
    detector_scores = ensemble_scores(
        model.detectors, 
        encode_all(model.autoencoder, x; batchsize=model.parameters.autoencoder[:batchsize], mean=true)
        )
    cat(detector_scores, [rec_score], dims=1)
end

"""
	classifier_inputs(model::HAD, x[, y]; L=10)

Transforms x (and y) into inputs for the classifier - combination of all scores.
"""
classifier_inputs(model::HAD, x; L=10) = Array(transpose(cat(all_scores(model, x; L=L)..., dims = 2)))
function classifier_inputs(model::HAD, x, y; L=10)
    cl_x = classifier_inputs(model::HAD, x; L=L)
    cl_y = Flux.onehotbatch(Bool.(y), 0:1);
    cl_x, cl_y
end

"""
	classifier_score(classifier, x)

Extracts the probability that a sample is an anomaly.
"""
classifier_score(c, x) = softmax(c(x))[2,:]
