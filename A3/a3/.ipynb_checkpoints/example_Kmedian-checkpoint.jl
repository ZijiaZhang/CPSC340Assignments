# Load data
using JLD
X = load("clusterData2.jld","X")

# K-means clustering
goodModel = 0
minError = Inf
k = 3
for i in 1:50
    global minError;
    global goodModel;
    include("kMedian.jl")
    model = kMedians(X,k,doPlot=false)
    y = model.predict(X)
    err = kMediansError(X,y,model.W);
    if (err < minError)
        minError = err;
        goodModel = model
    end
end

y = goodModel.predict(X)
include("clustering2Dplot.jl")
clustering2Dplot(X,y,goodModel.W)
