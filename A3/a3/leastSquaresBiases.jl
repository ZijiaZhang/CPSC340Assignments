include("misc.jl")

function leastSquares(X,y,d)
    n = size(X,1)
    v = ones(n,1)
    R = pow(X,d)
    R = [v R]
	# Find regression weights minimizing squared error
	w = (R'R)\(R'y)

	# Make linear prediction function
	function predict(Xhat)
        n1 = size(Xhat,1)
        v = ones(n1,1)
        R = pow(Xhat,d)
        R = [v R]
        return R*w
    end
	# Return model
	return GenericModel(predict)
end

function pow(x,d)
    r = x[:,1]
    for i in 2:d
        g =r.^i
        x = [x g]
    end
    return x
end
