using Printf

# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

ValX = X[1:floor(Int,n/2),:]
ValY = y[1:floor(Int,n/2),:]

ValSize = size(TrainX,1)

TrainX = X[floor(Int,n/2)+1:end,:]
TrainY = y[floor(Int,n/2)+1:end,:]

TrainSIze = size(ValX,1)

# Train a depth-2 decision tree
maxdepth = 15
depthArray = zeros(maxdepth)
resultTrain = zeros(maxdepth)
resultTest = zeros(maxdepth)
resultVal = zeros(maxdepth)

for depth in 1:15
    include("decisionTree_infoGain.jl")
    model = decisionTree_infoGain(TrainX,TrainY,depth)

    # Evaluate the training error
    yhat = model.predict(TrainX)
    trainError = sum(yhat .!= TrainY)/TrainSize
    @printf("Train error with depth-%d decision tree: %.3f\n",depth,trainError)
    resultTrain[depth] = trainError
    
    
    
    #Evaluate Validation error
    yhat = model.predict(ValX)
    valError = sum(yhat .!= ValY)/ValSize
    @printf("Validation error with depth-%d decision tree: %.3f\n",depth,trainError)
    resultVal[depth] = valError
    
    # Evaluate the test error
    Xtest = load("citiesSmall.jld","Xtest")
    ytest = load("citiesSmall.jld","ytest")
    t = size(Xtest,1)
    yhat = model.predict(Xtest)
    testError = sum(yhat .!= ytest)/t
    @printf("Test error with depth-%d decision tree: %.3f\n",depth,testError)
    resultTest[depth] = testError
    depthArray[depth] = depth
end

minVal = minimum(resultVal)
DesiredDepth = depthArray[findall(x->x==minVal,resultVal)]

using PyPlot

annotate("Minimum Validation Error",
	xy=[depthArray[convert(Int64,maximum(DesiredDepth))]+0.005;resultVal[convert(Int64,maximum(DesiredDepth))]+ 0.005],
	xytext=[depthArray[convert(Int64,maximum(DesiredDepth))]+0.1;resultVal[convert(Int64,maximum(DesiredDepth))]+0.1],
	xycoords="data",
	arrowprops=Dict("facecolor"=>"black"))


plot(depthArray,resultTest, label="Test Error")
plot(depthArray,resultTrain, label="Training Error")
plot(depthArray,resultVal, label="Validation Error")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
