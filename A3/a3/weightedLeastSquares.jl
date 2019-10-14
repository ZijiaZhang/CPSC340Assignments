include("misc.jl")

function weightedLeastSquares(X,y,v)
    X = X.*v
	# Find regression weights minimizing squared error
	w = (X'X)\(X'y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end
