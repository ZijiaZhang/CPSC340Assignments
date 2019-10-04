using Printf
using Statistics

# Load data
using JLD
fileName = "vowel.jld"
X = load(fileName,"X")
y = load(fileName,"y")
Xtest = load(fileName,"Xtest")
ytest = load(fileName,"ytest")

# Fit a decision tree classifier
include("decisionTree_infoGain.jl")
depth = 7
model = decisionTree_infoGain(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d decision tree: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d decision tree: %.3f\n",depth,testError)

# Fit a random tree classifier
include("randomTree.jl")
depth = 5
model = randomTree(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d random tree: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d random tree: %.3f\n",depth,testError)

function randomForest(X,y,depth,ntrees)
    myForest = Array{GenericModel}(undef,ntrees)
    for i in 1:ntrees
        myForest[i] = randomTree(X,y,depth)
    end
    function predict(Xhat)
        (t,d) = size(Xhat)
        predictResult = zeros(t,ntrees)
        for i in 1:ntrees
            predictResult[:,i] = myForest[i].predict(Xhat)
        end
        yhat = zeros(t)
        for i in 1:t
            yhat[i] = mode(predictResult[i,:])
        end
        return yhat
    end
    return GenericModel(predict)
    
end


# Fit a random forest classifier
depth = 7
model = randomForest(X,y,depth,50)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d random forest: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d random forest: %.3f\n",depth,testError)