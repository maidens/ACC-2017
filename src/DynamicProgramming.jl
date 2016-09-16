module DynamicProgramming

using Interpolations

export dp_loop, dp_rollout 

# index into a grid defined by a tuple of FloatRange
function index_into(grid::Tuple, index::CartesianIndex)
    return convert(Array{Float64, 1}, [grid[j][k] for (j, k) in enumerate(index.I)])
end

# compute the optimal value function at x 
function step(k::Int64, x::Array{Float64, 1}, ugrid::Tuple, theta::Array{Float64, 1}, V, f::Function, g::Function)
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

# compute optimal value function with dynamic programming algorithm using a non-parallel loop 
function dp_loop(f::Function, phi::Function, ugrid::Tuple, xgrid::Tuple,
            theta0::Array{Float64, 1}, N::Int64)   
    xgrid_length_tuple = ([length(l) for l in xgrid]...)
    J = SharedArray(Float64, xgrid_length_tuple..., N)
    for k = N-1:-1:1
        V = interpolate(xgrid, reshape(slicedim(J, length(xgrid_length_tuple)+1, k+1), xgrid_length_tuple), Gridded(Linear()))            
        for i in CartesianRange(xgrid_length_tuple)
            x_i = index_into(xgrid, i)
            J[i.I..., k] = step(k, x_i, ugrid, theta0, V, f, phi)
        end
    end            
    return J
end

# compute optimal input at x 
function step_u(k::Int64, x::Array{Float64, 1}, ugrid::Tuple, theta::Array{Float64, 1}, V, f::Function, g::Function)
    J_star = -Inf
    ugrid_length_tuple = ([length(l) for l in ugrid]...)
    u_star = zeros(length(ugrid_length_tuple))
    for i in CartesianRange(ugrid_length_tuple)
        u = index_into(ugrid, i)
        J_u = g(x, u, theta) + V[f(k, x, u, theta)...]
        if J_u > J_star
            J_star = J_u
            u_star = u
        end
    end
    return u_star
end

# rollout a trajectory starting at x0 using the value function J 
function dp_rollout(J, x0::Array{Float64, 1}, f::Function, phi::Function, 
                    ugrid::Tuple, xgrid::Tuple, theta0::Array{Float64, 1}, N::Int64)
    x = zeros((length(x0), N))
    u = zeros((length(ugrid), N-1))
    x[:, 1] = x0
    xgrid_length_tuple = ([length(l) for l in xgrid]...)
    for k=1:N-1
        V = interpolate(xgrid, reshape(slicedim(J, length(xgrid_length_tuple)+1, k), xgrid_length_tuple), Gridded(Linear()))
        u[:, k] = step_u(k, x[:, k], ugrid, theta0, V, f, phi)
        x[:, k+1] = f(k, x[:, k], u[:, k], theta0)
    end
    return x, u
end

end # Module