using LinearAlgebra
using Random
using Compat

using Printf
using Dates

using JuMP
using MathOptInterface
using Gurobi
using Mosek
using MosekTools

function branchandbound_frob_matrixcomp(
    k::Int,
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    gap::Float64 = 1e-6, # optimality gap for algorithm (proportion)
    max_steps::Int = 1000000,
    time_limit::Int = 3600, # time limit in seconds
    with_log::Bool = true,
)

    function print_output(outfile::String, printlist, with_log::Bool)
        if with_log
            open(outfile, "a+") do f
                for note in printlist
                    print(f, note)
                    print(note)
                end
            end
        else
            for note in printlist
                print(note)
            end
        end
    end

    function add_update(node_id, counter, lower, upper, start_time, outfile, with_log)
        if (lower == -1e10 || upper == 1e10)
            return
        end
        printlist = [
            Printf.@sprintf(
                "| %10d | %10d | %10f | %10f | %10f | %10.3f  s  |\n",
                node_id,
                counter,
                lower,
                upper,
                abs((upper - lower) / (lower + 1e-10)),
                time() - start_time,
            ),
        ]
        print_output(outfile, printlist, with_log)
    end

    if !(size(A) == size(indices))
        error("""
        Dimension mismatch. 
        Input matrix A must have size (n, m);
        Input matrix indices must have size (n, m);
        """)
    end

    if with_log
        log_time = Dates.now()
        time_string = Dates.format(log_time, "yyyymmdd_HHMMSS")
        outfile = "logs/" * time_string * ".txt"
    end

    (n, m) = size(A)
    printlist = [
        Dates.format(log_time, "e, dd u yyyy HH:MM:SS"), "\n",
        "Starting branch-and-bound on a matrix completion problem.\n",
        Printf.@sprintf("k:                 %10d\n", k),
        Printf.@sprintf("m:                 %10d\n", m),
        Printf.@sprintf("n:                 %10d\n", n),
        Printf.@sprintf("num_indices:       %10d\n", sum(indices)),
        Printf.@sprintf("γ:                 %10g\n", γ),
        Printf.@sprintf("λ:                 %10g\n", λ),
        Printf.@sprintf("Optimality gap:    %10g\n", gap),
        Printf.@sprintf("Maximum nodes:     %10d\n", max_steps),
        Printf.@sprintf("Time limit (s):    %10d\n", time_limit),
        "-----------------------------------------------------------------------------------\n",
        "|   Explored |      Total |  Objective |  Incumbent |        Gap |    Runtime (s) |\n",
        "-----------------------------------------------------------------------------------\n",
    ]
    print_output(outfile, printlist, with_log)

    start_time = time()

    # TODO: better initial Us?
    U_altmin, V_altmin = alternating_minimization(
        A, k, indices, γ, λ,
    )
    # do a re-SVD on U * V in order to recover orthonormal U
    X_initial = U_altmin * V_altmin
    U_initial, S_initial, V_initial = svd(X_initial) # TODO: implement truncated SVD
    U_initial = U_initial[:,1:k]
    Y_initial = U_initial * U_initial'
    objective_initial = objective_function(
        X_initial, A, indices, U_initial, γ, λ,
    )
    
    best_solution = Dict(
        "objective" => objective_initial, 
        "Y" => Y_initial,
        "U" => U_initial,
        "X" => X_initial,
    )

    U_lower_initial = -ones(n, k)
    U_upper_initial = ones(n, k)
    nodes = [(U_lower_initial, U_upper_initial, 1)]

    upper = objective_initial
    lower = -1e10

    lower_bounds = Dict()
    ancestry = []

    counter = 1
    now_gap = 1e5

    while (
        now_gap > gap &&
        counter < max_steps &&
        time() - start_time ≤ time_limit
    )
        if length(nodes) != 0
            (U_lower, U_upper, node_id) = popfirst!(nodes)
        else
            break
        end

        prune_flag = false

        if !feasibility_check_U(U_lower, U_upper)
            prune_flag = true
            continue
        end

        # solve SDP relaxation of master problem
        SDP_result = SDP_relax_frob_matrixcomp(U_lower, U_upper, A, indices, γ, λ)
        
        if SDP_result["feasible"] == false
            prune_flag = true
            continue
        else
            objective_SDP = SDP_result["objective"]
            lower_bounds[node_id] = objective_SDP
            Y_SDP = SDP_result["Y"]
            U_SDP = SDP_result["U"]
            t_SDP = SDP_result["t"]
            X_SDP = SDP_result["X"]
            Θ_SDP = SDP_result["Θ"]
        end

        # if solution for SDP_result has higher objective than best found so far: prune the node
        if objective_SDP ≥ best_solution["objective"]
            prune_flag = true
        end

        # if solution for SDP_result is feasible for original problem:
        # prune this node;
        # if it is the best found so far, update best_solution
        if master_problem_frob_matrixcomp_feasible(Y_SDP, U_SDP, t_SDP, X_SDP, Θ_SDP)
            # if best found so far, update best_solution
            if objective_SDP < best_solution["objective"]
                best_solution["objective"] = objective_SDP
                upper = objective_SDP
                best_solution["Y"] = copy(Y_SDP)
                best_solution["U"] = copy(U_SDP)
                best_solution["X"] = copy(X_SDP)
                println("better solution found!")
                add_update(node_id, counter, lower, upper, start_time, outfile, with_log)
            end
            prune_flag = true
        end

        if prune_flag
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
        push!(nodes, (U_lower, U_upper_new, counter + 1))
        push!(nodes, (U_lower_new, U_upper, counter + 2))
        push!(ancestry, (node_id, [counter + 1, counter + 2]))

        (anc_node_id, anc_children_node_ids) = ancestry[1]
        if all(haskey(lower_bounds, id) for id in anc_children_node_ids)
            popfirst!(ancestry)
            pop!(lower_bounds, anc_node_id)
            if minimum(values(lower_bounds)) > lower
                lower = minimum(values(lower_bounds))
                add_update(node_id, counter, lower, upper, start_time, outfile, with_log)
            end
        end

        counter += 2
        if (counter % 1000 <= 1)
            add_update(node_id, counter, lower, upper, start_time, outfile, with_log)
        end
    end

    if with_log
        open(outfile, "a+") do f
            print(f, "\n\nU:\n")
            show(f, "text/plain", best_solution["U"])
            print(f, "\n\nY:\n")
            show(f, "text/plain", best_solution["Y"])
            print(f, "\n\nX:\n")
            show(f, "text/plain", best_solution["X"])
            print(f, "\n\nA:\n")
            show(f, "text/plain", A)
            print(f, "\n\nindices:\n")
            show(f, "text/plain", indices)
            print(f, "\n\nBest incumbent solution:\n")
            show(f, "text/plain", best_solution["objective"])
        end
    end

    return best_solution

end

function master_problem_frob_matrixcomp_feasible(Y, U, t, X, Θ)
    if !(all(abs.(U' * U - I) .≤ 1e-5))
        return false
    end
    if !(eigvals(Symmetric(Y - U * U'), 1:1)[1] ≥ -1e-6)
        return false
    end
    if !(eigvals(Symmetric([Y X; X' Θ]), 1:1)[1] ≥ -1e-6)
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
        size(U_lower) == size(U_upper) 
        && size(U_lower, 1) == size(U_upper, 1) == size(A, 1) == size(indices, 1) 
        && size(A) == size(indices)
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
            U_lower[i, j2] * U[i, j1] 
            + U_lower[i, j1] * U[i, j2] 
            - U_lower[i, j1] * U_lower[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = 1:k],
        t[i, j1, j2] ≥ (
            U_upper[i, j2] * U[i, j1] 
            + U_upper[i, j1] * U[i, j2] 
            - U_upper[i, j1] * U_upper[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = 1:k],
        t[i, j1, j2] ≤ (
            U_upper[i, j2] * U[i, j1] 
            + U_lower[i, j1] * U[i, j2] 
            - U_lower[i, j1] * U_upper[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = 1:k],
        t[i, j1, j2] ≤ (
            U_lower[i, j2] * U[i, j1] 
            + U_upper[i, j1] * U[i, j2] 
            - U_upper[i, j1] * U_lower[i, j2]
        )
    )

    # Orthogonality constraints U'U = I using new variables
    for j1 = 1:k, j2 = 1:k
        if (j1 == j2)
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≤ 1.0 + 1e-6
            )
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≥ 1.0 - 1e-6
            )
        else
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≤   1e-6
            )
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≥ - 1e-6
            )
        end
    end

    @objective(
        model,
        Min,
        (1 / 2) * sum(
            (X[i, j] - A[i, j])^2 * indices[i, j] 
            for i = 1:n, j = 1:m
        ) 
        + (1 / (2 * γ)) * sum(Θ[i, i] for i = 1:m) 
        + λ * sum(Y[i, i] for i = 1:n)
    )

    optimize!(model)

    if JuMP.termination_status(model) == MOI.OPTIMAL
        return Dict(
            "feasible" => true,
            "objective" => objective_value(model),
            "Y" => value.(Y),
            "U" => value.(U),
            "t" => value.(t),
            "X" => value.(X),
            "Θ" => value.(Θ),
        )
    else
        return Dict(
            "feasible" => false,
        )
    end
end

function alternating_minimization(
    A::Array{Float64,2},
    k::Int,
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    ϵ::Float64 = 1e-10,
    max_iters::Int = 10000,
)
    function minimize_U(
        W_current,
    )
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        @variable(model, U[1:n, 1:m])
        @objective(
            model,
            Min,
            (1 / 2) * sum(
                (
                    sum(U[i,k] * W_current[k,j] for k in 1:m) 
                    - A[i,j]
                )^2 * indices[i,j]
                for i = 1:n, j = 1:m
            )
            + (1 / (2 * γ)) * sum(
                sum(U[i,k] * W_current[k,j] for k in 1:m)^2
                for i in 1:n, j in 1:m
            )
        )
        optimize!(model)
        return value.(U), objective_value(model)
    end

    function minimize_W(
        U_current,
    )
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        @variable(model, W[1:m, 1:m])
        @objective(
            model,
            Min,
            (1 / 2) * sum(
                (
                    sum(U_current[i,k] * W[k,j] for k in 1:m) 
                    - A[i,j]
                )^2 * indices[i,j]
                for i = 1:n, j = 1:m
            )
            + (1 / (2 * γ)) * sum(
                sum(U_current[i,k] * W[k,j] for k in 1:m)^2
                for i in 1:n, j in 1:m
            )
        )
        optimize!(model)
        return value.(W), objective_value(model)
    end

    (n, m) = size(A)
    A_initial = zeros(n, m)
    for i in 1:n, j in 1:m
        if indices[i,j] == 1
            A_initial[i,j] = A[i,j]
        end
    end

    U_current, S_current, V_current = svd(A_initial)
    W_current = Diagonal(vcat(S_current[1:k], repeat([0], m-k))) * V_current' 

    counter = 0
    objective_current = 1e10

    while counter < max_iters
        counter += 1
        U_new, _ = minimize_U(W_current)
        W_new, objective_new = minimize_W(U_new)
        objective_diff = abs(objective_new - objective_current)
        # println(counter)
        # println(objective_diff)
        if objective_diff < ϵ # objectives don't oscillate!
            return U_new, W_new
        end
        U_current = U_new
        W_current = W_new
        objective_current = objective_new
    end
    return U_new, W_new
end

function objective_function(
    X::Array{Float64,2},
    A::Array{Float64,2},
    indices::Array{Float64,2},
    U::Array{Float64,2},
    γ::Float64,
    λ::Float64,
)
    if !(
        size(X) == size(A) == size(indices) 
        && size(X, 1) == size(U, 1)
    )
        error("""
        Dimension mismatch. 
        Input matrix X must have size (n, m);
        Input matrix A must have size (n, m);
        Input matrix indices must have size (n, m);
        Input matrix U must have size (n, k).
        """)
    end
    n, m = size(X)
    n, k = size(U)
    return (
        (1 / 2) * sum(
            (X[i,j] - A[i,j])^2 * indices[i,j]
            for i = 1:n, j = 1:m
        )
        + (1 / (2 * γ)) * sum(X.^2)
        + λ * sum(U.^2)
    )
end

function feasibility_check_U(U_lower, U_upper)
    (n, k) = size(U_lower)
    U_max = max.(abs.(U_lower .- 0), abs.(U_upper .- 0))
    if !all(
        sum(U_max[:,i].^2) ≥ 1
        for i in 1:size(U_max, 2)
    )
        return false
    end
    U_min = zeros(n, k)
    for i in 1:n, j in 1:k
        if U_lower[i,j] ≤ 0 ≤ U_upper[i,j]
            U_min[i,j] = 0
        else
            U_min[i,j] = min(abs(U_lower[i,j] - 0), abs(U_upper[i,j] - 0))
        end
    end
    if !all(
        sum(U_min[:,i].^2) ≤ 1
        for i in 1:size(U_min, 2)
    )
        return false
    end
    return true
end
