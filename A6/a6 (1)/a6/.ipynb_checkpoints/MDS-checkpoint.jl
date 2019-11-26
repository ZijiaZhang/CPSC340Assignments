include("misc.jl")
include("PCA.jl")
include("findMin.jl")

function MDS(X)
    (n,d) = size(X)

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stress(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2

            # Gradient
            df = s
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
        end
    end
    return (f,G[:])
end

function ISOMAP(X)
    n,d = size(X);
    G = zeros(n,n);
    D = distancesSquared(X,X);
    D = sqrt.(abs.(D));
    for i in 1:n
        G[i,:] = fourMin(D[i,:])
        G[i,i] = Inf
    end
    largest = 0;
    for i in 1:n
        for j in 1:n
            if G[i,j]!= Inf && G[i,j] > largest
                largest = G[i,j]
            end
        end
    end
    
    for i in 1:n
        for j in 1:n
            if G[i,j] != Inf
                G[j,i] = G[i,j]
            else
                G[i,j] = largest
            end
        end
    end
    #@show G
    r = zeros(n,n)
    for i in 1:n
        for j in 1:i
            r[i,j] = dijkstra(G,i,j)
            #@show r[i,j]
            r[j,i] = r[i,j]
        end
    end
    #@show r
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,r)

    Z[:] = findMin(funObj,Z[:])

    return Z
    
end

function fourMin(V)
    n = size(V,1)
    min1Index = 1
    min2Index = 1
    min3Index = 1
    min4Index = 1
    for i =1:n
        if V[i] < V[min1Index]
            min4Index = min3Index
            min3Index = min2Index
            min2Index = min1Index
            min1Index = i
        elseif V[i] < V[min2Index]
            min4Index = min3Index
            min3Index = min2Index
            min2Index = i
        elseif V[i] < V[min3Index]
            min4Index = min3Index
            max3Index = i
        elseif V[i] < V[min4Index]
            min4Index = i
        end
    end
    r = zeros(size(V))
    for i = 1:n
        r[i] = Inf
    end
    r[min1Index] = V[min1Index]
    r[min2Index] = V[min2Index]
    r[min3Index] = V[min3Index]
    #r[min4Index] = V[min4Index]
    return r
end
        