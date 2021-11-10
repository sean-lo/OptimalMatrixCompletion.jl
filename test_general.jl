using Test
include("general.jl")

function test_compute_f_Y_frob(
    k::Int, # rank of Y, ncol of U
    l::Int, # nrow of A, nrow of B
    m::Int, # ncol of B, ncol of C
    n::Int, # ncol of A, nrow of C, size of Y
    seed::Int;
    γ::Float64 = 1.0, # regularization parameter for problem
    solver_output::Int = 1,
)
    Random.seed!(seed)
    A = randn(Float64, (l, n))
    B = randn(Float64, (l, m))
    C = randn(Float64, (n, m))
    U = qr!(randn(Float64, (n, k))).Q[:,1:k]
    Y = U * U'
    result = compute_f_Y_frob(Y, A, B, C, γ; solver_output=solver_output)
    return result
end

result = test_compute_f_Y_frob(3,1,5,6,2; solver_output=0)
@test (
    termination_status(result["model"]) == MOI.OPTIMAL 
)

result = test_compute_f_Y_frob(3,2,5,6,2; solver_output=0)
@test (
    termination_status(result["model"]) == MOI.OPTIMAL 
)
