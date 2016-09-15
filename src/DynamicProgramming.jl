module DynamicProgramming

using Interpolations

export dp_map, dp_loop, dp_forward

# function implementing parallel map with data grouping
function pmap_group(f, lst, group_size::Int)
    np = nprocs()  # determine the number of processes available
    n = length(lst)
    results = Array(Float64, n)
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=group_size; idx)
    @sync begin
        for p=1:np
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                    results[idx:idx+group_size-1] = remotecall_fetch(p, map, f, lst[idx:idx+group_size-1])
                    end
                end
            end
        end
    end
    results
end

# compute value function and optimal input 
function step_u(ts::Int64, x::Array{Float64, 1},
              ugrid::FloatRange{Float64}, theta::Array{Float64, 1}, V, f::Function, phi::Function)
    V_max = -Inf        # maximum value
    u_max = ugrid[1]    # maximizing input
    for u in ugrid
        V_u = phi(x, u, theta) + V[f(ts, x, u, theta)...]
        if V_u > V_max
            V_max = V_u
            u_max = u
        end
    end
    return (V_max, u_max)
end

function step(ts::Int64, x::Array{Float64, 1},
              ugrid::FloatRange{Float64}, theta::Array{Float64, 1}, V, f::Function, phi::Function)
    return step_u(ts, x, ugrid, theta, V, f, phi)[1]
end

# dynamic programming function
function dp_map(n, f::Function, phi::Function, ugrid::FloatRange{Float64}, xgrid::Tuple,
            theta0::Array{Float64, 1}, N, group_size::Int64)
    
    J = zeros([length(gridvals) for gridvals in xgrid]..., N)

    for t = N-1:-1:1
        if n == 2
            V = interpolate(xgrid, J[:, :, t+1], Gridded(Linear()))
            X = [[i, j] for i in xgrid[1], j in xgrid[2]]
            J[:, :, t] = pmap_group(x->step(t, x, ugrid, theta0, V, f, phi), X, group_size)
        elseif n == 4
            println(t)
            V = interpolate(xgrid, J[:, :, :, :, t+1], Gridded(Linear()))
            X = [[i, j, k, l] for i in xgrid[1], j in xgrid[2], k in xgrid[3], l in xgrid[4]]
            J[:, :, :, :, t] = pmap_group(x->step(t, x, ugrid, theta0, V, f, phi), X, group_size)
        else 
            error("Dynamic programming function has not been defined for this value of n") 
        end
    end
    return J
end

function dp_loop(n, f::Function, phi::Function, ugrid::FloatRange{Float64}, xgrid::Tuple,
            theta0::Array{Float64, 1}, N::Int64)
    
    J = SharedArray(Float64, [length(gridvals) for gridvals in xgrid]..., N)
    u = SharedArray(Float64, [length(gridvals) for gridvals in xgrid]..., N-1)

    if n == 4
        for t = N-1:-1:1
            println(t)
            V = interpolate(xgrid, J[:, :, :, :, t+1], Gridded(Linear()))

            @sync @parallel for i = 1:length(xgrid[1])
                for j = 1:length(xgrid[2])
                    for k= 1:length(xgrid[3])
                        for l = 1:length(xgrid[4])
                            (J[i, j, k, l, t], u[i, j, k, l, t]) = step_u(t, 
                                            [xgrid[1][i], xgrid[2][j], xgrid[3][k], xgrid[4][l]], 
                                            ugrid, theta0, V, f, phi)
                        end
                    end
                end
            end
        end
    elseif n == 2
        for t = N-1:-1:1
            V = interpolate(xgrid, J[:, :, t+1], Gridded(Linear()))

            @sync @parallel for i = 1:length(xgrid[1])
                for j = 1:length(xgrid[2])
                    (J[i, j, t], u[i, j, t]) = step_u(t, 
                                            [xgrid[1][i], xgrid[2][j]], 
                                            ugrid, theta0, V, f, phi)
                end
            end
        end
    else 
        error("Dynamic programming function has not been defined for this value of n") 
    end
    return (J, u)
end

function dp_forward(n, Jvals, uvals, f, x0, xgrid, theta0, N)
    u_opt = zeros(N-1)
    x = zeros(n, N)
    x[:, 1] = x0
    for t=1:N-1
        if n == 4
            u = interpolate(xgrid, uvals[:, :, :, :, t], Gridded(Linear())) 
        elseif n == 2
            u = interpolate(xgrid, uvals[:, :, t], Gridded(Linear())) 
        else
            error("Dynamic programming function has not been defined for this value of n") 
        end
        u_opt[t] = u[x[:, t]...]
        x[:, t+1] = f(t, x[:, t], u_opt[t], theta0)
    end
    
    return (x, u_opt) 
end


end # Module