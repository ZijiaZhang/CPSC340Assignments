using LinearAlgebra
include("misc.jl")

function leastSquares(X,y)

	# Find regression weights minimizing squared error
	w = (X'*X)\(X'*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return LinearModel(predict,w)
end

function leastSquaresBiasL2(X,y,lambda)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X X.*X X.*X.*X]

	# Find regression weights minimizing squared error
 	v = (Z'*Z + lambda*I)\(Z'*y)
    #v = Z'*(Z*Z' + lambda * I)^-1*y
    #@show(v1, v)
	# Make linear prediction function
    @show (Z*Z')[1:1]
	predict(Xhat) = [ones(size(Xhat,1),1) Xhat Xhat.*Xhat Xhat.*Xhat.*Xhat]*v

	# Return model
	return LinearModel(predict,v)
end

function leastSquaresBasis(x,y,p)
	Z = polyBasis(x,p)

	v = (Z'*Z)\(Z'*y)

	predict(xhat) = polyBasis(xhat,p)*v

	return LinearModel(predict,v)
end

function polyBasis(x,p)
	n = length(x)
	Z = zeros(n,p+1)
	for i in 0:p
		Z[:,i+1] = x.^i
	end
	return Z
end

function weightedLeastSquares(X,y,v)
	V = diagm(v)
	w = (X'*V*X)\(X'*V*y)
	predict(Xhat) = Xhat*w
	return LinearModel(predict,w)
end

function binaryLeastSquares(X,y)
	w = (X'X)\(X'y)

	predict(Xhat) = sign.(Xhat*w)

	return LinearModel(predict,w)
end


function leastSquaresRBF(X,y,sigma)
	(n,d) = size(X)

	Z = rbf(X,X,sigma)

	v = (Z'*Z)\(Z'*y)

	predict(Xhat) = rbf(Xhat,X,sigma)*v

	return LinearModel(predict,v)
end

function rbf(Xhat,X,sigma)
	(t,d) = size(Xhat)
	n = size(X,1)
	D = distancesSquared(Xhat,X)
	return (1/sqrt(2pi*sigma^2))exp.(-D/(2sigma^2))
end


function leastSquaresKernalBias(X,y,lambda, p)

	# Find regression weights minimizing squared error
 	k = polyKernal(X,X, p)
	# Make linear prediction function
	predict(Xhat) = polyKernal(Xhat, X, p) * ((k + lambda * I) \ y)

	# Return model
	return GenericModel(predict)
end

function polyKernal(X1, X2, p)
    t = size(X1,1)
    n = size(X2,1)
    k = ones(t,n)
    for i = 1:t
        for j = 1:n
            k[i,j] = (1 + X1[i,:]'*X2[j,:])^p
        end
    end
    return k
end

function RBFKernalBias(X,y,lambda, sigma)

	# Find regression weights minimizing squared error
 	k = RBFKernal(X,X, sigma)
	# Make linear prediction function
	predict(Xhat) = RBFKernal(Xhat, X, sigma) * ((k + lambda * I) \ y)

	# Return model
	return GenericModel(predict)
end

function RBFKernal(X1, X2, sigma)
    t = size(X1,1)
    n = size(X2,1)
    k = ones(t,n)
    for i = 1:t
        for j = 1:n
            k[i,j] = exp(- norm(X1[i,:] - X2[j,:])^2/(2*sigma^2))
        end
    end
    return k
end