using Printf
include("misc.jl")
include("findMin.jl")

function robustRegression(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = robustRegressionObj(w,X,y)

	# This is how you compute the function and gradient:
	(f,g) = funObj(w)
    @show g
	# Derivative check that the gradient code is correct:
	g2 = numGrad(funObj,w)

	if maximum(abs.(g-g2)) > 1e-4
		@printf("User and numerical derivatives differ:\n")
		@show([g g2])
	else
		@printf("User and numerical derivatives agree\n")
	end

	# Solve least squares problem
	w = findMin(funObj,w)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function robustRegressionObj(w,X,y)
    n = size(X)[1]

    f=0
    g = zeros(size(w))
	for i in 1:n
        r = dot(X[i,:],w) - y[i]
        if abs(r) > 1
            f = f + (abs(r) - 0.5)
        else
            f = f + 0.5*r*r
        end
        g = g + sign(r) *min(abs(r),1)* transpose(X[i,:])
    end
	
	return (f,g)
end

