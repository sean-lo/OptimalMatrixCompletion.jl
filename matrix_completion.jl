using LinearAlgebra
using Random
using Compat

using JuMP
using MathOptInterface
using Gurobi
using Mosek
using MosekTools

function compute_f_Y_frob_matrixcomp(
    Y::Array{Float64,2},
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64;
    solver_output = 0,
)

    if !(
        size(A, 1) == size(Y, 1) == size(Y, 2)
    )
        error("""
        Dimension mismatch. 
        Input matrix Y must have size (n, n); 
        input matrix A must have size (n, m).
        """)
    end

    (n, n) = size(Y)
    (n, m) = size(A)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, α[1:n, 1:m])
    @constraint(
        model,
        α .* (ones(n, m) - indices) .== 0.0
    )

    @objective(
        model,
        Max,
        - (γ / 2) * sum(
            Y[i, j] * sum(
                α[i, k] * α[j, k] 
                for k = 1:m
            )
            for i = 1:n, j = 1:n
        )
        - (1 / 2) * sum(
            (α[i,j] - A[i,j])^2 * indices[i,j]
            for i = 1:n, j = 1:m
        )
        + (1 / 2) * sum(
            A[i,j]^2 * indices[i,j]
            for i = 1:n, j = 1:m
        )
    )

    optimize!(model)

    return Dict(
        "model" => model,
        "objective" => objective_value(model),
        "α" => value.(α) .* indices,
        "H" => -(value.(α) * value.(α)') .* (γ / 2),
    )
end