using Printf
using JLD
using PyPlot
using Statistics
include("misc.jl")
# Load X and y variable
data = load("uspsData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

(n,d) = size(X)
t = size(Xtest,1)
@show(size(X))
# Standardize columns and add bias variable to input layer
(X,mu,sigma) = standardizeCols(X)
#X = [ones(n,1) X]
d += 1

include("PCA.jl")
PCAfit = PCA(X, 200)
Z = PCAfit.compress(X);
Z = [ones(n,1) Z]
(n,d) = size(Z)

# Apply the same transformation to test data
Xtest = standardizeCols(Xtest,mu=mu,sigma=sigma)
#Xtest = [ones(t,1) Xtest]



Ztest = PCAfit.compress(Xtest);
Ztest = [ones(t,1) Ztest]

# Let 'k' be the number of classes, and 'Y' be a matrix of binary labels
k = maximum(y)
@show k
Y = zeros(n,k)
for i in 1:n
	Y[i,y[i]] = 1
end

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [200]
@show nHidden
nParams = NeuralNetMulti_nParams(d,k,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
maxIter = 50000
stepSize = 1e-3
for iter in 1:maxIter

	# The stochastic gradient update:
	i = rand(1:n)
    if (iter == 25000)
        global stepSize
        stepSize = 1e-4
    elseif iter == 50000
        global stepSize
        stepSize = 1e-6
    end
	(f,g) = NeuralNetMulti_backprop(w,Z[i,:],Y[i,:],k,nHidden)
	global w = w - stepSize*g

	# Every few iterations, plot the data/model:
	if (mod(iter,round(maxIter/50)) == 0)
		yhat = NeuralNet_predict(w,Ztest,k,nHidden)
		@printf("Training iteration = %d, test error = %f, stepSize = %f\n",iter,sum(yhat .!= ytest)/t, stepSize)
	end
end
