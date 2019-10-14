using Printf
using Statistics

# Load X and y variable
using JLD
data = load("outliersData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("weightedLeastSquares.jl")
v = ones(500,1)
for i in 401:500
    v[i] = 0.1
end
model = weightedLeastSquares(X,y,v)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with: %.3f\n",trainError)

    # Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with: %.3f\n",testError)
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.1:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,"g")
