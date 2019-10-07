using Printf
using Statistics

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquaresBiases.jl")
n=100
Xa = zeros(n,1)
ya = zeros(n,1)
mint = Inf
minr = Inf
for i in 1:n
    global Xa
    global ya
    global mint, minr
    model = leastSquares(X,y,i)

    # Evaluate training error
    yhat = model.predict(X)
    trainError = mean((yhat - y).^2)
    #@printf("Squared train Error with p = %d: %.3f\n",i,trainError)

    # Evaluate test error
    yhat = model.predict(Xtest)
    testError = mean((yhat - ytest).^2)
    @printf("Squared test Error with p = %d: %.3f\n",i,testError)
    Xa[i] = i
    ya[i] = testError
    if testError < mint
        mint = testError
        minr = i
    end
    # Plot model
#     using PyPlot
#     figure()
#     plot(X,y,"b.")
#     Xhat = minimum(X):.1:maximum(X)
#     yhat = model.predict(Xhat)
#     plot(Xhat,yhat,"g")
end

using PyPlot
figure()
plot(Xa,ya)
yscale("log")

@show mint
@show minr
