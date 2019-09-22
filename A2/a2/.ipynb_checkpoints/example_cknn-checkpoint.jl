using Printf
using Statistics
#using TickTock

# Load X and y variable
using JLD
dataName = "citiesBig2.jld"
X = load(dataName,"X")
y = load(dataName,"y")
Xtest = load(dataName,"Xtest")
ytest = load(dataName,"ytest")

# Fit a KNN classifier
for k in 1:2
    include("knn.jl")
    model = knn(X,y,k)

    #tick()
    # Evaluate training error
    yhat = @time model.predict(X)
    trainError = mean(yhat .!= y)
    @printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)
    #tock()
    
     #tick()
    # Evaluate test error
    yhat = @time model.predict(Xtest)
    testError = mean(yhat .!= ytest)
    @printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)
    #tock()
    
    model = cknn(X,y,k)

    #tick()
    # Evaluate training error
    yhat = @time model.predict(X)
    trainError = mean(yhat .!= y)
    @printf("Train Error cknn with %d-nearest neighbours: %.3f\n",k,trainError)
    #tock()
    
    #tick()
    # Evaluate test error
    yhat = @time model.predict(Xtest)
    testError = mean(yhat .!= ytest)
    @printf("Test Error cknn with %d-nearest neighbours: %.3f\n",k,testError)
    #tock()
end
