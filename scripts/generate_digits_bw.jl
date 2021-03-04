using DrWatson
@quickactivate
using HierarchicalAD
using FileIO, BSON
using StatsBase
using ProgressMeter

N_samples = 100000

font_size = 24
img_size = (28, 28)
fonts = ["Georgia", "", "Arial", "Ubuntu Mono", "Free Mono", "Eufm10", "Lobster Two", "Purisa", "Manjari Thin", "Chilanka", "Uroob", "MathJax_Typewriter"]
accents = ["", " Bold"]
digs = 0:9
rots = [-20, -10, -5, -3, 0, 3, 5, 10, 20]
shearxs = -20:20
shearys = -20:20
zooms =0.9:0.1:1.2

sample_input() = map(x->sample(x,1)[1], (fonts, accents, digs, rots, shearxs, shearys, zooms))

inputs = map(i->sample_input(), 1:N_samples)
inputf = datadir("digits_bw/labels.bson")
save(inputf, Dict(:inputs => inputs))

@info "Generating digits..."
outputs = @showprogress map(i->HierarchicalAD.create_digit_bw(font_size, img_size, i...), inputs)
outputf = datadir("digits_bw/digits.bson")
save(outputf, Dict(:outputs=> outputs))
