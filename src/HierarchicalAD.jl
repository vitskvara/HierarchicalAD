module HierarchicalAD

using DrWatson
@quickactivate
using Luxor
using Augmentor
using ImageIO, FileIO
using StatsBase
using Images
using BSON
using IPMeasures
using Flux
using CSV
using DataFrames
using MLDatasets
using Random
using Dates
using ValueHistories
using DataDeps
using PyCall
using UCI
using NearestNeighbors
using LinearAlgebra
using Statistics
using EvalMetrics
using Suppressor

include("digits.jl")
include("model_utils.jl")
include("disentanglement.jl")
include("training_utils.jl")
include("experiments.jl")
include("shapes2d.jl")
include("data.jl")
include("evaluation.jl")

include("knn.jl")
include("abstract_vlae.jl")
include("vlae.jl")
include("fvlae.jl")
include("had.jl")

export VLAE, FVLAE, AbstractVLAE

end