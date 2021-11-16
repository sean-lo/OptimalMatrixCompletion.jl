using LinearAlgebra
using Random
using Compat

using JuMP
using MathOptInterface
using Gurobi
using Mosek
using MosekTools

include("utils.jl")

function compute_f_Y_frob_matrixcomp(
    Y::Array{Float64,2},
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64,
    ;
    solver_output::Int = 0,
)

    if !(size(A, 1) == size(Y, 1) == size(Y, 2))
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
    @constraint(model, α .* (ones(n, m) - indices) .== 0.0)

    @objective(
        model,
        Max,
        -(γ / 2) *
        sum(Y[i, j] * sum(α[i, k] * α[j, k] for k = 1:m) for i = 1:n, j = 1:n) -
        (1 / 2) *
        sum((α[i, j] - A[i, j])^2 * indices[i, j] for i = 1:n, j = 1:m) +
        (1 / 2) * sum(A[i, j]^2 * indices[i, j] for i = 1:n, j = 1:m)
    )

    optimize!(model)

    return Dict(
        "model" => model,
        "objective" => objective_value(model),
        "α" => value.(α) .* indices,
        "H" => -(value.(α) * value.(α)') .* (γ / 2),
    )
end

function naive_master_problem_frob_matrixcomp(
    Y_initial::Array{Float64,2},
    k::Int,
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    n_cuts::Int = 5,
    solver_output::Int = 0,
)
    if !(size(A, 1) == size(Y_initial, 1) == size(Y_initial, 2))
        error("""
        Dimension mismatch. 
        Input matrix Y_initial must have size (n, n); 
        input matrix A must have size (n, m).
        """)
    end

    (n, n) = size(Y_initial)
    (n, m) = size(A)

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

    @objective(
        model,
        Min,
        theta + λ * k # theta + λ * trace(U' U)
    )

    Y = Y_initial

    for epoch = 1:n_cuts
        println("Epoch $epoch")
        inner_result =
            compute_f_Y_frob_matrixcomp(Y, A, indices, γ; solver_output = 0)
        f_Y = inner_result["objective"]
        H = inner_result["H"]

        h = f_Y - sum(H .* Y_initial)

        matrix_posdef_result = make_matrix_posdef(H)
        H_new = matrix_posdef_result["H_new"]
        δ = matrix_posdef_result["δ"]
        L = cholesky(H_new).U'
        Û = [Compat.dot(L[:, i], U[:, j]) for i = 1:n, j = 1:k]
        @constraint(model, theta + δ * k - h ≥ Compat.dot(Û, Û))

        optimize!(model)
        U = JuMP.value.(U)
        Y = U * U'
    end

    U_opt = U
    Y_opt = Y

    println("Solved!")
    println("Optimal Y: $Y_opt")
    println("Optimal objective: $(objective_value(model))")

    return Dict(
        "model" => model,
        "objective" => objective_value(model),
        "U" => U_opt,
        "Y" => Y_opt,
    )
end

function trustregion_master_problem_frob_matrixcomp(
    U_initial::Array{Float64,2},
    k::Int,
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    n_cuts::Int = 5,
    tol::Float64 = 1e-5,
    solver_output::Int = 0,
)
    if !(size(A, 1) == size(U_initial, 1))
        error("""
        Dimension mismatch. 
        Input matrix Y_initial must have size (n, n); 
        input matrix A must have size (n, m).
        """)
    end

    Y_initial = U_initial * U_initial'
    (n, n) = size(Y_initial)
    (n, m) = size(A)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, theta ≥ -1e5)
    @variable(model, U[1:n, 1:k])

    @objective(model, Min, theta + λ * k,)

    Y = Y_initial

    for epoch = 1:n_cuts
        println("Epoch $epoch")
        inner_result =
            compute_f_Y_frob_matrixcomp(Y, A, indices, γ; solver_output = 0)
        f_Y = inner_result["objective"]
        H = inner_result["H"]

        h = f_Y - sum(H .* Y_initial)

        matrix_posdef_result = make_matrix_posdef(H)
        H_new = matrix_posdef_result["H_new"]
        δ = matrix_posdef_result["δ"]
        L = cholesky(H_new).U'
        Û = [Compat.dot(L[:, i], U[:, j]) for i = 1:n, j = 1:k]
        @constraint(model, theta + δ * k - h ≥ Compat.dot(Û, Û))
        U_prev_1 = U_initial
        U_prev_2 = zeros(n, k)
        i = 0
        while sum(abs.(U_prev_2 .- U_prev_1)) ≥ 1e-5
            print(i)
            i += 1
            println(sum(U_prev_1))
            println(sum(U_prev_2))
            @constraint(model, orthogonality_1, U * U_prev_1' - I .<= tol)
            @constraint(model, orthogonality_2, I - U * U_prev_1' .<= tol)
            optimize!(model)
            U_prev_2 = U_prev_1
            U_prev_1 = JuMP.value.(U)
            delete(model, orthogonality_1)
            unregister(model, :orthogonality_1)
            delete(model, orthogonality_2)
            unregister(model, :orthogonality_2)
        end
        U = JuMP.value.(U)
        Y = U * U'
    end

    U_opt = U_prev_1
    Y_opt = Y

    println("Solved!")
    println("Optimal Y: $Y_opt")
    println("Optimal objective: $(objective_value(model))")

    return Dict(
        "model" => model,
        "objective" => objective_value(model),
        "U" => U_opt,
        "Y" => Y_opt,
    )
end

function branchandbound_frob_matrixcomp(
    k::Int,
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    max_steps::Int = 10000,
    solver_output::Int = 0,
)

    if !(size(A) == size(indices))
        error("""
        Dimension mismatch. 
        Input matrix A must have size (n, m);
        Input matrix indices must have size (n, m);
        """)
    end

    (n, m) = size(A)

    U_lower_initial = -ones(n, k)
    U_upper_initial = ones(n, k)

    nodes = [(U_lower_initial, U_upper_initial)]
    best_solution = Dict(
        "objective" => 1e10, # some really high number
        "Y" => zeros(n, n),
        "U" => zeros(n, k),
        "X" => zeros(n, m),
    )

    counter = 0

    while true
        (U_lower, U_upper) = popfirst!(nodes)
        # println()
        # println("New node:")
        # println("U_lower: ", U_lower)
        # println("U_upper: ", U_upper)

        # solve SDP relaxation of master problem
        SDP_result = SDP_relax_frob_matrixcomp(U_lower, U_upper, A, indices, γ, λ)
        objective_SDP = SDP_result["objective"]
        Y_SDP = SDP_result["Y"]
        U_SDP = SDP_result["U"]
        t_SDP = SDP_result["t"]
        X_SDP = SDP_result["X"]
        Θ_SDP = SDP_result["Θ"]

        # if solution for SDP_result has higher objective than best found so far: prune the node
        if objective_SDP ≥ best_solution["objective"]
            # println("Pruning dominated node")
            continue
        end

        # if solution for SDP_result is feasible for original problem:
        # prune this node;
        # if it is the best found so far, update best_solution
        if master_problem_frob_matrixcomp_feasible(
            Y_SDP,
            U_SDP,
            t_SDP,
            X_SDP,
            Θ_SDP,
        )
            println("SDP solution feasible for original problem!")
            # if best found so far, update best_solution
            if objective_SDP ≤ best_solution["objective"]
                best_solution["objective"] = objective_SDP
                best_solution["Y"] = copy(Y_SDP)
                best_solution["U"] = copy(U_SDP)
                best_solution["X"] = copy(X_SDP)
                println(best_solution)
                println(counter)
            end
            continue
        end

        # branch on variable
        # for now: branch on biggest element-wise difference between U_lower and U_upper
        (diff, index) = findmax(U_upper - U_lower)
        mid = U_lower[index] + diff / 2
        U_lower_new = copy(U_lower)
        U_lower_new[index] = mid
        U_upper_new = copy(U_upper)
        U_upper_new[index] = mid
        append!(nodes, [(U_lower, U_upper_new), (U_lower_new, U_upper)])
        # println(nodes)

        counter += 1
        if counter ≥ max_steps
            break
        end
    end

    return best_solution

end

function master_problem_frob_matrixcomp_feasible(Y, U, t, X, Θ)
    if !(all(abs.(U' * U - I) .≤ 1e-5))
        return false
    end
    if !(eigvals(Symmetric(Y - U * U'), 1:1)[1] ≥ 0)
        return false
    end
    if !(eigvals(Symmetric([Y X; X' Θ]), 1:1)[1] ≥ 0)
        return false
    end
    return true
end

function SDP_relax_frob_matrixcomp(
    U_lower::Array{Float64,2},
    U_upper::Array{Float64,2},
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    solver_output::Int = 0,
)
    if !(
        size(U_lower) == size(U_upper) &&
        size(U_lower, 1) ==
        size(U_upper, 1) ==
        size(A, 1) ==
        size(indices, 1) &&
        size(A) == size(indices)
    )
        error("""
        Dimension mismatch. 
        Input matrix U_lower must have size (n, k); 
        Input matrix U_upper must have size (n, k); 
        Input matrix A must have size (n, m);
        Input matrix indices must have size (n, m).
        """)
    end

    (n, k) = size(U_lower)
    (n, m) = size(A)

    model = Model(Mosek.Optimizer)
    if solver_output == 0
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
    end
    # set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, X[1:n, 1:m])
    @variable(model, Y[1:n, 1:n])
    @variable(model, Θ[1:m, 1:m])
    @variable(model, U[1:n, 1:k])
    @variable(model, t[1:n, 1:k, 1:k])

    @constraint(model, LinearAlgebra.Symmetric([Y X; X' Θ]) in PSDCone())
    @constraint(model, LinearAlgebra.Symmetric([Y U; U' I]) in PSDCone())

    @constraint(model, I - Y in PSDCone())

    # McCormick inequalities at U_lower and U_upper here
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = 1:k],
        t[i, j1, j2] ≥ (
            U_lower[i, j2] * U[i, j1] + U_lower[i, j1] * U[i, j2] -
            U_lower[i, j1] * U_lower[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = 1:k],
        t[i, j1, j2] ≥ (
            U_upper[i, j2] * U[i, j1] + U_upper[i, j1] * U[i, j2] -
            U_upper[i, j1] * U_upper[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = 1:k],
        t[i, j1, j2] ≤ (
            U_upper[i, j2] * U[i, j1] + U_lower[i, j1] * U[i, j2] -
            U_lower[i, j1] * U_upper[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = 1:k],
        t[i, j1, j2] ≤ (
            U_lower[i, j2] * U[i, j1] + U_upper[i, j1] * U[i, j2] -
            U_upper[i, j1] * U_lower[i, j2]
        )
    )

    # Orthogonality constraints U'U = I using new variables
    for j1 = 1:k, j2 = 1:k
        delta = 1.0 * (j1 == j2)
        @constraint(
            model,
            [j1 = 1:k, j2 = 1:k],
            sum(t[i, j1, j2] for i = 1:n) ≤ delta + 1e-6
        )
        @constraint(
            model,
            [j1 = 1:k, j2 = 1:k],
            sum(t[i, j1, j2] for i = 1:n) ≥ delta - 1e-6
        )
    end

    @objective(
        model,
        Min,
        (1 / 2) * sum(
            (X[i, j] - A[i, j])^2 * indices[i, j] 
            for i = 1:n, j = 1:k
        ) 
        + (1 / (2 * γ)) * sum(Θ[i, i] for i = 1:m) 
        + λ * sum(Y[i, i] for i = 1:n)
    )

    optimize!(model)

    return Dict(
        "model" => model,
        "objective" => objective_value(model),
        "Y" => value.(Y),
        "U" => value.(U),
        "t" => value.(t),
        "X" => value.(X),
        "Θ" => value.(Θ),
    )
end