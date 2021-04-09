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

include("digits.jl")
include("abstract_vlae.jl")
include("vlae.jl")
include("fvlae.jl")
include("model_utils.jl")
include("training_utils.jl")
include("experiments.jl")
include("data.jl")

export VLAE, FVLAE, AbstractVLAE

end