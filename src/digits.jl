using Luxor
using Augmentor
using ImageIO, FileIO
using StatsBase
using Images
using BSON

function bw_digit(font::String, accent::String, digit, font_size::Int, img_size::Tuple)
    Drawing(img_size...)
    background("black")
    sethue("white")
    p = Point(6, 28)
    setfont(font*accent, font_size)
    settext("$digit", p)
    img = image_as_matrix()
    finish()
    img
end

function rgb_digit(font::String, accent::String, digit, font_size::Int, img_size::Tuple, col)
    Drawing(img_size...)
    background("black")
    sethue(col)
    p = Point(6, 28)
    setfont(font*accent, font_size)
    settext("$d", p)
    img = image_as_matrix()
    finish()
    img
end

img_to_array_bw(img) = reshape(Float32.(channelview(Gray.(img))), size(img)...,1)
img_to_array_rgb(img) = permutedims(Float32.(channelview(RGB.(img))), (2,3,1))
array_to_img_bw(arr) = Gray.(reshape(permutedims(arr, (3,1,2)), size(arr)[1:2]...))
array_to_img_rgb(arr) = RGB.(arr[:,:,1], arr[:,:,2], arr[:,:,3])

function transform(img, rot, shearx, sheary, zoom)
    pl = Rotate([rot]) |>
            ShearX([shearx]) |>
            ShearY([sheary]) |>
            CropSize(28, 28) |>
            Zoom([zoom])
   augment(img, pl)
end

function create_digit_bw(font_size, img_size, font, accent, dig, rot, shearx, sheary, zoom)
	orig = bw_digit(font, accent, dig, font_size, img_size)
	new = transform(orig, rot, shearx, sheary, zoom)
	img_to_array_bw(new)
end

function create_digit_rgb(font_size, img_size, font, accent, dig, col, rot, shearx, sheary, zoom)
    orig = rgb_digit(font, accent, dig, font_size, img_size, col)
    new = transform(orig, rot, shearx, sheary, zoom)
    img_to_array_rgb(new)
end

vectorize(x::AbstractArray{T,4}) where T = reshape(x, :, size(x,4))
devectorize(x::AbstractArray{T,2}, w, h) where T = reshape(x, w, h, 1, :)
devectorize(x::AbstractArray{T,2}, w, h, c) where T = reshape(x, w, h, c, :)

function draw(x::AbstractArray{T,4}) where T
    return hcat(array_to_img_bw.([x[:,:,:,i] for i in 1:size(x,4)])...)
end
function draw(x::AbstractArray{T,3}) where T
    return array_to_img_bw(x)
end
function draw(x::AbstractArray{T,2}, s=(28,28)) where T
    draw(devectorize(x, s...))
end
