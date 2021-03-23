using DrWatson
@quickactivate
using HierarchicalAD
HAD = HierarchicalAD
using Flux

X = randn(Float32,28,28,3,10);
zdim = 2
ks = ((3,3), (3,3))
ncs = (2,2)
str = 1
datasize = size(X)
var = :dense

model1 = VLAE(zdim, ks, ncs, str, datasize; var=var)
zs = HAD.encode_all(model, X)
XH = HAD.decode(model, zs...);

var = :conv
model = VLAE(zdim, ks, ncs, str, datasize; var=var)
zs = HAD.encode_all(model, X)
XH = HAD.decode(model, zs...);
