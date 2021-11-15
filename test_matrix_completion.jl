using Test

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

function test_compute_f_Y_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
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
    U = qr!(randn(Float64, (n, k))).Q[:,1:k]
    Y = U * U'

    indices = generate_masked_bitmatrix(n, m, n_indices)

    result = compute_f_Y_frob_matrixcomp(Y, A, indices, γ; solver_output=solver_output)
    return result
end

result = test_compute_f_Y_frob_matrixcomp(1,3,4,8,0; solver_output=0)
@test (
    termination_status(result["model"]) == MOI.OPTIMAL
)

H = result["H"]
@test(
    all(
        abs.(
            make_matrix_posdef(H, kind="simple")["H_new"] 
            .- make_matrix_posdef(H, kind="loose")["H_new"]
        ) .< 1e-9
    )
)

function test_naive_master_problem_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 1.0,
    solver_output::Int = 1,
    n_cuts::Int = 5
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
    U = qr!(randn(Float64, (n, k))).Q[:,1:k]
    Y = U * U'

    indices = generate_masked_bitmatrix(n, m, n_indices)

    return naive_master_problem_frob_matrixcomp(
        Y, k, A, indices, γ, λ;
        solver_output=solver_output,
        n_cuts=n_cuts
    )
end

# fast!
result = test_naive_master_problem_frob_matrixcomp(
    1, # rank
    3, # m (ncols(A))
    4, # n (dim of Y == nrow(A))
    8, # number of indices provided
    0  # seed
)
println(result)

# # slow! 
# result = test_naive_master_problem_frob_matrixcomp(2,7,5,26,0; n_cuts=2)
# println(result)


function test_trustregion_master_problem_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 1.0,
    solver_output::Int = 1,
    n_cuts::Int = 5
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
    U = qr!(randn(Float64, (n, k))).Q[:,1:k]
    Y = U * U'

    indices = generate_masked_bitmatrix(n, m, n_indices)

    return trustregion_master_problem_frob_matrixcomp(
        U, k, A, indices, γ, λ;
        solver_output=solver_output,
        n_cuts=n_cuts
    )
end

# fast! but infeasible
result = test_trustregion_master_problem_frob_matrixcomp(
    1, # rank
    3, # m (ncols(A))
    4, # n (dim of Y == nrow(A))
    8, # number of indices provided
    0  # seed
)
println(result)

function test_branchandbound_SDP_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
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
    U_lower = -ones(n, k)
    U_upper = ones(n, k)
    A = randn(Float64, (n, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    return branchandbound_SDP_frob_matrixcomp(
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

result = test_branchandbound_SDP_frob_matrixcomp(1,3,4,8,0; solver_output=0)
println(result)

function test_branchandbound_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
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
    U_lower = -ones(n, k)
    U_upper = ones(n, k)
    A = randn(Float64, (n, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    return branchandbound_frob_matrixcomp(
        k,
        A,
        indices,
        γ,
        λ,
        ;
        solver_output = solver_output,
    )
end

include("matrix_completion.jl")
test_branchandbound_frob_matrixcomp(1,3,4,8,0, solver_output = 0)