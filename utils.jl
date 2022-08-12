using LinearAlgebra
using Random
using Compat

using JuMP
using MathOptInterface
using Gurobi
using Mosek
using MosekTools

@doc raw"""
This function takes the matrix H and returns a new matrix H + δ_matrix that is positive definite, by some margin.
"""
function make_matrix_posdef(H; tol = 1e-3, kind = "simple")
    if !(size(H, 1) == size(H, 2))
        error("""
        Dimension mismatch.
        Input matrix H must have size (n, n).
        """)
    end

    (n, n) = size(H)

    model = Model(Mosek.Optimizer)
    if kind == "tight"
        @variable(model, δ[1:n] ≥ 0)
        # @SDconstraint(model, H + Matrix(Diagonal(δ)) ≥ zeros(n, n))
        @constraint(model, H + Matrix(Diagonal(δ)) in PSDCone())
        @objective(model, Min, sum(δ))
        optimize!(model)
        δ = JuMP.value.(δ)
        H_new = H + Matrix(Diagonal(JuMP.value.(δ))) + tol * I
    elseif kind == "loose"
        @variable(model, δ ≥ 0)
        # @SDconstraint(model, H + (δ * Matrix(1.0 * I, n, n)) ≥ zeros(n, n))
        @constraint(model, H + (δ * Matrix(1.0 * I, n, n)) in PSDCone())        
        @objective(model, Min, δ)
        optimize!(model)
        δ = JuMP.value.(δ)
        H_new =  H + JuMP.value.(δ) * I + tol * I
    elseif kind == "simple"
        δ = LinearAlgebra.eigvals(H)[1]
        H_new = H - δ * I + tol * I
    else
        error("Argument kind must be 'simple', 'loose', or 'tight'!")
    end

    return Dict(
        "δ" => δ, 
        "H_new" => H_new,
    )
end

function generate_masked_bitmatrix(
    dim1::Int,
    dim2::Int,
    sparsity::Int,
    seed::Int,
)
    Random.seed!(seed)
    while true
        index_pairs = randperm(dim1 * dim2)[1:sparsity]
        index_vec = zeros(dim1 * dim2)
        index_vec[index_pairs] .= 1.0
        indices = reshape(index_vec, (dim1, dim2))
        if (
            minimum(sum(indices, dims=1)) ≥ 0.5
            && minimum(sum(indices, dims=2)) ≥ 0.5
        )
            return indices
        end
    end
end

function generate_matrixcomp_data( # FIXME! make matrices seeds different
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    noise::Bool = false,
    ϵ::Float64 = 0.01,
)
    if n_indices < (n + m) * k
        error("""
        System is under-determined. 
        n_indices must be at least (n + m) * k.
        """)
    end
    if n_indices > n * m
        error("""
        Cannot generate random indices of length more than the size of matrix A.
        """)
    end
    if (n > 1000) || (m > 1000)
        error("""
        Currently does not support n > 1000, m > 1000. 
        Supplied n = $n, m = $m.
        """)
    end
    # 4 sources of randomness
    seeds = abs.(rand(MersenneTwister(seed), Int, 4)) 
    A_left = randn(MersenneTwister(seeds[1]), Float64, (1000, k))[1:n, :]
    A_right = randn(MersenneTwister(seeds[2]), Float64, (k, 1000))[:, 1:m]
    A = A_left * A_right
    if noise
        A_noise = randn(MersenneTwister(seeds[3]), Float64, (1000, 1000))[1:n, 1:m]
        A = A + ϵ * A_noise
    end
    indices = generate_masked_bitmatrix(n, m, n_indices, seeds[4])
    return A, indices
end