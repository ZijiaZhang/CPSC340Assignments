# Load data
using JLD
X = load("clusterData2.jld","X")
include("kMeans.jl")
# K-means clustering
ak = zeros(10,1)
ae = zeros(10,1)
for k in 1:10
    global ak
    global ae
    goodModel = 0
    minError = Inf
    @show k
    for i in 1:50
        model = kMeans(X,k,doPlot=false)
        y = model.predict(X)
        err = kMeansError(X,y,model.W);
        if (err < minError)
            minError = err;
            goodModel = model
        end
    ak[k] = k
    ae[k] = minError
    end
end

using PyPlot
plot(ak,ae,linestyle="-")

# y = goodModel.predict(X)
# include("clustering2Dplot.jl")
# clustering2Dplot(X,y,goodModel.W)
