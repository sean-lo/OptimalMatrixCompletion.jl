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
    seed::Int;
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