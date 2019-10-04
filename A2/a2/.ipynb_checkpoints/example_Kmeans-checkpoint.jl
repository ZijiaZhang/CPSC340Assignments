# Load data
using JLD
X = load("clusterData.jld","X")

# K-means clustering
k = 4
include("kMeans.jl")
goodmodel = 0
goodErrors = Inf


for i in 1:50
    @show i
    model = kMeans(X,k,doPlot=false);
    y = model.predict(X);
    thisError = kMeansError(X,y,model.W);
    if (goodErrors - thisError > 0)
        goodErrors = thisError
        goodmodel = model
    end
end

y = goodmodel.predict(X) 
include("clustering2Dplot.jl")
clustering2Dplot(X,y,goodmodel.W)