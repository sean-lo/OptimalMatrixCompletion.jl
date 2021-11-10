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
        @SDconstraint(model, H + Matrix(Diagonal(δ)) ≥ zeros(n, n))
        @objective(model, Min, sum(δ))
        optimize!(model)
        δ = JuMP.value.(δ)
        H_new = H + Matrix(Diagonal(JuMP.value.(δ))) + tol * I
    elseif kind == "loose"
        @variable(model, δ ≥ 0)
        @SDconstraint(model, H + (δ * Matrix(1.0 * I, n, n)) ≥ zeros(n, n))
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