include("misc.jl")

function leastSquares(X,y,d,s)
    n = size(X,1)
    R = pow(X,d,s)
	# Find regression weights minimizing squared error
	w = (R'R)\(R'y)

	# Make linear prediction function
	function predict(Xhat)
        n1 = size(Xhat,1)
        R = pow(Xhat,d,s)
        return R*w
    end
	# Return model
	return GenericModel(predict)
end

function pow(x,d,s)
    r = x[:,1]
    ans = ones(size(x))
    for i in 1:d
        g =r.^i
        p = sin.(g/(s/30))
        #q = cos.(g)
        ans = [ans g p]
    end
    return ans
end
