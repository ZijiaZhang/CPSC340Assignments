using DelimitedFiles

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

# Standardize columns
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)

# Plot matrix as image
using PyPlot
figure(1)
clf()
imshow(X)

include("PCA.jl")
model = PCA(X,13)
compressed = model.compress(X);
# Show scatterplot of 2 random features
# j1 = rand(1:d)
# j2 = rand(1:d)
figure(2)
clf()
plot(compressed[:,1],compressed[:,2],".")
for i in rand(1:n,10)
    annotate(dataTable[i+1,1],
	xy=[compressed[i,1],compressed[i,2]],
	xycoords="data")
end
