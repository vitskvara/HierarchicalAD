using DrWatson
@quickactivate
using HierarchicalAD
using FileIO, BSON
using StatsBase
using ProgressMeter
using Colors

N_samples = 100000

font_size = 24
img_size = (28, 28)
fonts = ["Georgia", "", "Arial", "Ubuntu Mono", "Free Mono", "Eufm10", "Lobster Two", "Purisa", "Manjari Thin", "Chilanka", "Uroob", "MathJax_Typewriter"]
accents = ["", " Bold"]
digs = 0:9
cols = collect(keys(Colors.color_names))
rots = [-20, -10, -5, -3, 0, 3, 5, 10, 20]
shearxs = -20:20
shearys = -20:20
zooms =0.9:0.01:1.2

sample_input() = map(x->sample(x,1)[1], (fonts, accents, digs, cols, rots, shearxs, shearys, zooms))

inputs = map(i->sample_input(), 1:N_samples)
inputf = datadir("digits_rgb/labels.bson")
save(inputf, Dict(:labels => inputs))

@info "Generating digits..."
outputs = @showprogress map(i->HierarchicalAD.create_digit_rgb(font_size, img_size, i...), inputs)
outputf = datadir("digits_rgb/digits.bson")
save(outputf, Dict(:digits=> outputs))
