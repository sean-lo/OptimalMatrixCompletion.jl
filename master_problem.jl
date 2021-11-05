using LinearAlgebra
using Random

using JuMP
using Gurobi
using Mosek
using MosekTools

@doc raw"""
This function computes $f(\bm{Y})$, solving the inner maximization problem in α. It returns the best α and optimal value, if it exists.
"""
function compute_f_Y_frob(
    Y::Array{Float64,2},
    A::Array{Float64,2},
    B::Array{Float64,2},
    C::Array{Float64,2},
    γ::Float64;
    solver_output = 0,
)
    if !(
        size(C, 1) == size(Y, 1) == size(Y, 2) == size(A, 2) &&
        size(A, 1) == size(B, 1) &&
        size(B, 2) == size(C, 2)
    )
        error("""
        Dimension mismatch. 
        Input matrix Y must have size (n, n); 
        input matrix A must have size (l, n); 
        input matrix B must have size (l, m); 
        input matrix C must have size (n, m).
        """)
    end

    (n, n) = size(Y)
    (l, n) = size(A)
    (l, m) = size(B)
    (n, m) = size(C)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, Π[1:l, 1:m])
    @variable(model, α[1:n, 1:m])

    @constraint(model, vec(C - α - A' * Π) in MOI.Nonnegatives(n * m))

    @objective(
        model,
        Max,
        sum(B .* Π)
        # - γ/2 * ⟨α, Yα⟩ = -γ/2 * tr(α^⊤ Y α) = -γ/2 * tr(Y α α^⊤) 
        -
        (γ / 2) *
        sum(Y[i, j] * sum(α[i, k] * α[j, k] for k = 1:m) for i = 1:n, j = 1:n)
    )

    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return Dict(
            "model" => model,
            "objective" => objective_value(model),
            "α" => value.(α),
            "Π" => value.(Π),
            "H" => -(value.(α) * value.(α)') .* (γ / 2),
        )
    else
        return Dict(
            "model" => model,
            "objective" => Inf,
            # TODO: decide how to return some choice of α and Π, producing cut
        )
    end
end


@doc raw"""
This function takes the matrix H and returns a new matrix H + δ_matrix that is positive definite, by some margin.
"""
function make_matrix_posdef(H; tol = 1e-3, kind = "loose")
    # TODO: for the "loose" case, what about just subtracting I * (the most negative eigenvalue)?
    if !(size(H, 1) == size(H, 2))
        error("""
        Dimension mismatch.
        Input matrix H must have size (n, n).
        """)
    end

    model = Model(Mosek.Optimizer)
    if kind == "tight"
        @variable(model, δ[1:n] ≥ 0)
        @SDconstraint(model, H + Matrix(Diagonal(δ)) ≥ zeros(n, n))
        @objective(model, Min, sum(δ))
    elseif kind == "loose"
        @variable(model, δ ≥ 0)
        @SDconstraint(model, H + (δ * Matrix(1.0 * I, n, n)) ≥ zeros(n, n))
        @objective(model, Min, δ)
    else
        error("Argument kind must be 'loose' or 'tight'!")
    end

    @suppress optimize!(model)

    if kind == "tight"
        δ_matrix = Matrix(Diagonal(JuMP.value.(δ)))
    elseif kind == "loose"
        δ_matrix = JuMP.value.(δ) * I
    end

    return Dict("δ" => JuMP.value.(δ), "H_new" => H + δ_matrix + tol * I)
end

function master_problem_frob(
    Y_initial::Array{Float64,2},
    A::Array{Float64,2},
    B::Array{Float64,2},
    C::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    solver_output = 0,
)
    if !(
        size(C, 1) == size(Y_initial, 1) == size(Y_initial, 2) == size(A, 2) &&
        size(A, 1) == size(B, 1) &&
        size(B, 2) == size(C, 2)
    )
        error("""
        Dimension mismatch. 
        Input matrix Y_initial must have size (n, n); 
        input matrix A must have size (l, n); 
        input matrix B must have size (l, m); 
        input matrix C must have size (n, m).
        """)
    end

    (n, n) = size(Y_initial)
    (l, n) = size(A)
    (l, m) = size(B)
    (n, m) = size(C)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", solver_output)
    set_optimizer_attribute(model, "NonConvex", 2)

    @variable(model, theta ≥ -1e5)
    @variable(model, U[1:n, 1:k])

    # Orthogonality of U;
    # If i == j, then dot(U_i, U_j) should be 1
    # If not, then dot(U_i, U_j) should be 0
    for i = 1:k
        for j = 1:k
            delta = 1.0 * (i == j)
            @constraint(model, U[:, i]' * U[:, j] ≤ delta + 1e-6)
            @constraint(model, U[:, i]' * U[:, j] ≥ delta - 1e-6)
        end
    end

    inner_result = compute_f_Y_frob(Y_initial, A, B, C, γ; solver_output = 0)
    f_Y = inner_result["objective"]
    if f_Y != Inf
        H = inner_result["H"]
        h = f_Y - sum(H .* Y_initial)
        L = cholesky(H + 1e-3 * I).U'
        # form initial lazy constraint here
        Û = [Compat.dot(L[:, i], U[:, j]) for i = 1:n, j = 1:k]
        @constraint(model, theta + 1e-3 * k - h ≥ Compat.dot(Û, Û))
    else
        error("Initial f_Y unbounded!")
    end

    @objective(
        model,
        Min,
        theta + λ * k # theta + λ * trace(U' U)
    )

    function outer_approx(cb_data)
        println("Started callback!")
        U0 = zeros(n, k)
        for i = 1:n
            for j = 1:k
                U0[i, j] = callback_value(cb_data, U[i, j])
            end
        end
        # U = callback_value(cb_data, U)
        Y = U0 * U0'
        inner_result = compute_f_Y_frob(Y, A, B, C, γ; solver_output = 0)
        f_Y = inner_result["objective"]
        if f_Y != Inf
            H = inner_result["H"]
            h = f_Y - sum(H .* Y)
            L = cholesky(H + 1e-3 * I).U'
            # form new lazy constraint here
            Û = [Compat.dot(L[:, i], U[:, j]) for i = 1:n, j = 1:k]
            con = @build_constraint(theta + 1e-3 * k - h ≥ Compat.dot(Û, Û))
            println("Added constraint: theta + 1e-3 * k - h ≥ ||L' * U||²₂")
            println("k = $k")
            println("h = $h")
            println("L' = $(L')")
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), outer_approx)

    println("Starting solution...")
    optimize!(model)
    U_opt = JuMP.value.(U)
    Y_opt = U_opt * U_opt'

    println("Solved!")
    println("Optimal Y: $Y_opt")
    println("Optimal objective: $(objective_value(model))")

end

# Other constraint forms:
# 1: pure summations
# con = @build_constraint(
#     sum(
#         sum(
#             L[ind,i] * U[ind,j]
#             for ind in 1:n
#         )^2
#         for i in 1:n, j in 1:k
#     ) ≤ theta + 1e-3 * k - h 
# )
# 2: sum of dot products
# con = @build_constraint(
#     theta + 1e-3 * k - h ≥ sum(
#         Compat.dot(L[:,i], U[:,j])^2
#         for i in 1:n, j in 1:k
#     )
# )
# 3: dot product of matrix products
# con = @build_constraint(
#     theta + 1e-3 * k - h 
#     ≥ Compat.dot(
#         L' * U, 
#         L' * U
#     )
# )
# 4: dot product of transformed matrix of variables
# Û = [Compat.dot(L[:,i], U[:,j]) for i in 1:n, j in 1:k]
# con = @build_constraint(
# theta + 1e-3 * k - h ≥ 
# Compat.dot(Û, Û)
# )
# 5: SecondOrderCone formulation => not exactly right (because should it be theta^2?)
# con = @build_constraint(
#     [
#         theta + 1e-3 * k - h; 
#         vec(L[:,:]' * U[:,:])
#     ] in SecondOrderCone()
# )