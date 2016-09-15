module DynamicProgramming

using Interpolations

export dp_loop

# index into a grid defined by a tuple of FloatRange
function index_into(grid::Tuple, index::CartesianIndex)
    #for (j, k) in enumerate(index.I)
    #    println(ugrid[j])
    #end
    return convert(Array{Float64, 1}, [grid[j][k] for (j, k) in enumerate(index.I)])
end

# compute the optimal value function at x 
function step_u(k::Int64, x::Array{Float64, 1}, ugrid::Tuple, theta::Array{Float64, 1}, V, f::Function, g::Function)
    J_star = -Inf
    ugrid_length_tuple = ([length(l) for l in ugrid]...)
    for i in CartesianRange(ugrid_length_tuple)
        u = index_into(ugrid, i)
        J_u = g(x, u, theta) + V[f(k, x, u, theta)...]
        if J_u > J_star
            J_star = J_u
        end
    end
    return J_star
end

# perform dynamic programming algorithm using a non-parallel loop 
function dp_loop(f::Function, phi::Function, ugrid::Tuple, xgrid::Tuple,
            theta0::Array{Float64, 1}, N::Int64)   
    xgrid_length_tuple = ([length(l) for l in xgrid]...)
    J = SharedArray(Float64, xgrid_length_tuple..., N)
    for k = N-1:-1:1
        V = interpolate(xgrid, reshape(slicedim(J, length(xgrid_length_tuple)+1, k+1), xgrid_length_tuple), Gridded(Linear()))            
        for i in CartesianRange(xgrid_length_tuple)
            x_i = index_into(xgrid, i)
            J[i.I..., k] = step_u(k, x_i, ugrid, theta0, V, f, phi)
        end
    end            
    return J
end

end # Module