using DrWatson
@quickactivate
using HierarchicalAD
using MLDatasets
using FileIO, BSON
using CSV
using DataFrames

#
y_tr = MLDatasets.MNIST.trainlabels();
y_tst = MLDatasets.MNIST.testlabels();
inpath = datadir("morpho_mnist/original")
f_tr = joinpath(inpath, "train-morpho.csv")
f_tst = joinpath(inpath, "t10k-morpho.csv")
label_df = vcat(CSV.read(f_tr, DataFrame), CSV.read(f_tst, DataFrame))
label_df[!,:digit] = vcat(y_tr, y_tst)
CSV.write(datadir("morpho_mnist/labels.csv"), label_df)

#
bwl = load(datadir("digits_bw/labels.bson"))[:labels]
bw_df = DataFrame(
		:digit => [l[3] for l in bwl],
		:font => [l[1] for l in bwl],
		:boldness => [l[2] for l in bwl],
		:rotation => [l[4] for l in bwl],
		:xshear => [l[5] for l in bwl],
		:yshear => [l[6] for l in bwl],
		:scale => [l[7] for l in bwl]
	)
CSV.write(datadir("digits_bw/labels.csv"), bw_df)

# RGB
rgbl = load(datadir("digits_rgb/labels.bson"))[:labels]
rgb_df = DataFrame(
		:digit => [l[3] for l in rgbl],
		:color => [l[4] for l in rgbl],
		:font => [l[1] for l in rgbl],
		:boldness => [l[2] for l in rgbl],
		:rotation => [l[5] for l in rgbl],
		:xshear => [l[6] for l in rgbl],
		:yshear => [l[7] for l in rgbl],
		:scale => [l[8] for l in rgbl]
	)
CSV.write(datadir("digits_rgb/labels.csv"), rgb_df)
