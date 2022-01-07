using Test
using Random

include("matrix_completion.jl")
include("utils.jl")

function generate_masked_bitmatrix(
    dim1::Int,
    dim2::Int,
    sparsity::Int,
)
    index_pairs = randperm(dim1 * dim2)[1:sparsity]
    index_vec = zeros(dim1 * dim2)
    index_vec[index_pairs] .= 1.0
    return reshape(index_vec, (dim1, dim2))
end

function test_SDP_relax_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    U_lower::Array{Float64,2},
    U_upper::Array{Float64,2},
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 1.0,
    solver_output::Int = 1,
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
    Random.seed!(seed)
    A = randn(Float64, (n, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    return SDP_relax_frob_matrixcomp(
        U_lower,
        U_upper,
        A,
        indices,
        γ,
        λ,
        ;
        solver_output = solver_output,
    )
end

include("matrix_completion.jl");

# example with solution infeasible for original
result = test_SDP_relax_frob_matrixcomp(
    1,3,4,8,0,
    -ones(4,1), ones(4,1),
    ; 
    solver_output=0,
);
@test(!master_problem_frob_matrixcomp_feasible(
    result["Y"],
    result["U"],
    result["t"],
    result["X"],
    result["Θ"],
))

# example with solution feasible for original
result = test_SDP_relax_frob_matrixcomp(
    1,3,4,8,0,
    fill(0.5,(4,1)), ones(4,1),
    ; 
    solver_output=0,
)
@test(master_problem_frob_matrixcomp_feasible(
    result["Y"],
    result["U"],
    result["t"],
    result["X"],
    result["Θ"],
))

result = test_SDP_relax_frob_matrixcomp(
    2,5,7,25,0,
    -ones(7,2), ones(7,2),
    ; 
    solver_output=0,
)
@test(!master_problem_frob_matrixcomp_feasible(
    result["Y"],
    result["U"],
    result["t"],
    result["X"],
    result["Θ"],
))

function test_SOCP_relax_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    U_lower::Array{Float64,2},
    U_upper::Array{Float64,2},
    ;
    γ::Float64 = 10.0,
    λ::Float64 = 1.0,
    solver_output::Int = 1,
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
    Random.seed!(seed)
    A = randn(Float64, (n, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    return SOCP_relax_frob_matrixcomp(
        U_lower,
        U_upper,
        A,
        indices,
        γ,
        λ,
        ;
        solver_output = solver_output,
    )
end

function test_alternating_minimization(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 1.0,
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
    Random.seed!(seed)
    A = randn(Float64, (n, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    return alternating_minimization(
        A, k, indices, γ, λ,
    )
end

U, V = test_alternating_minimization(1,3,4,8,1)
@test(rank(U * V) == 1)

function test_branchandbound_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 1.0,
    relaxation::String = "SDP",
    max_steps::Int = 10000,
    time_limit::Int = 3600,
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
    Random.seed!(seed)
    A = randn(Float64, (n, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    return branchandbound_frob_matrixcomp(
        k,
        A,
        indices,
        γ,
        λ,
        ;
        relaxation = relaxation,
        max_steps = max_steps,
        time_limit = time_limit,
    )
end

include("matrix_completion.jl")

result = test_branchandbound_frob_matrixcomp(1,3,4,8,1, time_limit = 120, max_steps = 100000000)


result = test_branchandbound_frob_matrixcomp(1,3,5,9,0, max_steps = 10000, time_limit = 600)

result = test_branchandbound_frob_matrixcomp(1,3,4,8,0)

result = test_branchandbound_frob_matrixcomp(1,3,4,8,0, time_limit = 120, max_steps = 100000000)

result = test_branchandbound_frob_matrixcomp(1,5,6,24,0, max_steps = 200000, time_limit = 300)
result = test_branchandbound_frob_matrixcomp(1,5,6,24,1, max_steps = 200000, time_limit = 300)

result = test_branchandbound_frob_matrixcomp(2,5,6,24,0, max_steps = 200000, time_limit = 300)
result = test_branchandbound_frob_matrixcomp(2,5,6,24,1, max_steps = 200000, time_limit = 300)

