function __init__()
	register(
		DataDep(
			"shapes2D",
			"""
			Dataset: 2D shapes
			Authors: Loic Matthey and Irina Higgins and Demis Hassabis and Alexander Lerchner
			Website: https://github.com/deepmind/dsprites-dataset
			""",
			[
				"https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
			],
			"857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2"
		))
end

"""
	symmetric_angle(shape, angle)

Returns a normalized rotation angle that is dependent on the input shape and the number
of its symmetry axes.
"""
function symmetric_angle(shape, angle)
    if shape == 1 # square
        nsym = 4
    elseif shape == 2 # oval
        nsym = 2
    elseif shape == 3 # heart
        nsym = 1
    else
        error("Unknown shape code $shape.")
    end
    symfac = 360/nsym
    return (angle%symfac)/symfac
end
function _preprocess_shapes2D()
	@info "Preprocessing 2D shapes data..."
	f = joinpath(datadep"shapes2D", "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
	np = pyimport("numpy")
	data = np.load(f, allow_pickle=true, encoding="latin1")

	# image data
	x = get(data, "imgs")
	n = size(x,1)
	imax = 400000
	X1 = similar(x, 64, 64, 1, imax)
	X2 = similar(x, 64, 64, 1, n-imax)
	for i in 1:imax
		X1[:,:,1,i] .= x[i,:,:]
	end
	for i in imax+1:n
		X2[:,:,1,(i-imax)] .= x[i,:,:]
	end
	outf1 = datadir("shapes2D/shapes_part1.bson")
	outf2 = datadir("shapes2D/shapes_part2.bson")
	save(outf1, Dict(:imgs=>X1))
	save(outf2, Dict(:imgs=>X2))

	# labels
	factor_names = ["color", "shape", "scale", "orientation", "posX", "posY"]
	y = get(data, "latents_values")
	y[:,4] .= y[:,4]/(2*pi)*360 #rads to degrees
	df = DataFrame(y)
	map(x->rename!(df, x[1]=>Symbol(x[2])), zip(names(df), factor_names))
	# now add rotation factor that is different for each shape based on its symmetry
	df[!,:normalized_orientation] = symmetric_angle.(df[!,:shape], df[!,:orientation])
	outf = datadir("shapes2D/labels.csv")
	CSV.write(outf, df)
	@info "Done."
end
function load_shapes2D()
	# load X
	infs = map(x->datadir("shapes2D/shapes_part$x.bson"), [1,2])
	if !all(isfile.(infs))
		_preprocess_shapes2D()
	end
	Xs = map(load, infs)
	X = Float32.(cat([X[:imgs] for X in Xs]..., dims=4))

	# load labels
	f = datadir("shapes2D/labels.csv")
	if !isfile(f)
		_preprocess_shapes2D()
	end
	y = CSV.read(f, DataFrame)

	return X,y
end

