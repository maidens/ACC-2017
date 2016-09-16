module DynamicProgramming

using Interpolations

export dp_loop, dp_rollout 

# index into a grid defined by a tuple of FloatRange
function index_into(grid::Tuple, grid_size::Tuple, index::Int64)
    return convert(Array{Float64, 1}, [grid[j][k] for (j, k) in enumerate(ind2sub(grid_size, index))])
end

# compute grid size as a tuple = (size x1, size x2, ..., size xn) 
#             as well as an int = size x1 * size x2 * ... * size xn
function grid_size(grid::Tuple)
    tuple = ([length(l) for l in grid]...)
    int = 1
    for t in tuple
        int *= t
    end
    return tuple, int
end

# compute the optimal value function at x 
function step(k::Int64, x::Array{Float64, 1}, ugrid::Tuple, theta::Array{Float64, 1}, V, f::Function, g::Function)
    J_star = -Inf
    ugrid_length_tuple, ugrid_length_total = grid_size(ugrid)
    for i=1:ugrid_length_total
        u = index_into(ugrid, ugrid_length_tuple, i)
        J_u = g(x, u, theta) + V[f(k, x, u, theta)...]
        if J_u > J_star
            J_star = J_u
        end
    end
    return J_star
end

# compute optimal value function with dynamic programming algorithm using a parallel loop 
function dp_loop(f::Function, phi::Function, ugrid::Tuple, xgrid::Tuple,
            theta0::Array{Float64, 1}, N::Int64)   
    xgrid_length_tuple, xgrid_length_total = grid_size(xgrid)
    println("====== Constructing array J ======")
    println("J is size ", xgrid_length_tuple, " by ", N)
    J = SharedArray(Float64, xgrid_length_tuple..., N)
    println("====== Array J constructed  ======")
    for k = N-1:-1:1
        println("Step k = ", k)
        V = interpolate(xgrid, reshape(slicedim(J, length(xgrid_length_tuple)+1, k+1), xgrid_length_tuple), Gridded(Linear()))            
        @sync @parallel for i=1:xgrid_length_total
            x_i = index_into(xgrid, xgrid_length_tuple, i)
            J[ind2sub(xgrid_length_tuple, i)..., k] = step(k, x_i, ugrid, theta0, V, f, phi)
        end
    end            
    return J
end

# compute optimal input at x 
function step_u(k::Int64, x::Array{Float64, 1}, ugrid::Tuple, theta::Array{Float64, 1}, V, f::Function, g::Function)
    J_star = -Inf
    ugrid_length_tuple, ugrid_length_total = grid_size(ugrid)
    u_star = zeros(length(ugrid_length_tuple))
    for i=1:ugrid_length_total
        u = index_into(ugrid, ugrid_length_tuple, i)
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