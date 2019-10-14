include("misc.jl")

function leastSquares(X,y,d,s)
    n = size(X,1)
    v = ones(n,1)
    R = pow(X,5)
    R2 = cos.(R/s*0.1)
    R3 = sin.(R/d*0.1)
#     R4 = cos.(R/s*2).^2
#     R5 = sin.(R/d*2).^2
    R = [v R R2 R3]
	# Find regression weights minimizing squared error
	w = (R'R)\(R'y)

	# Make linear prediction function
	function predict(Xhat)
        n1 = size(Xhat,1)
        v = ones(n1,1)
        R = pow(Xhat,5)
        R2 = cos.(R/s*0.1)
        R3 = sin.(R/d*0.1)
#         R4 = cos.(R/s*2).^2
#         R5 = sin.(R/d*2).^2
        R = [v R R2 R3]
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
