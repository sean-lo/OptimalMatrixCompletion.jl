using LinearAlgebra
using Random
using Compat

using Printf
using Dates
using Suppressor
using DataFrames

using JuMP
using MathOptInterface
using Gurobi
using Mosek
using MosekTools
using Polyhedra

function branchandbound_frob_matrixcomp(
    k::Int,
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    relaxation::String = "SDP", # type of relaxation to use; either "SDP" or "SOCP"
    branching_type::String = "box", # type of branching to use; either "box" or "angular" or "polyhedral" or "polyhedral_lite"
    gap::Float64 = 1e-6, # optimality gap for algorithm (proportion)
    root_only::Bool = false, # if true, only solves relaxation at root node
    max_steps::Int = 1000000,
    time_limit::Int = 3600, # time limit in seconds
    update_step::Int = 1000,
)


    function add_update!(printlist, instance, node_id, counter, lower, upper, start_time)
        now_gap = (upper / lower) - 1
        current_time_elapsed = time() - start_time
        message = Printf.@sprintf(
            "| %10d | %10d | %10f | %10f | %10f | %10.3f  s  |\n",
            node_id, counter, lower, upper, now_gap, current_time_elapsed,
        )
        print(stdout, message)
        push!(printlist, message)
        push!(
            instance["run_log"],
            (node_id, counter, lower, upper, now_gap, current_time_elapsed)
        )
        return now_gap
    end

    if !(relaxation in ["SDP", "SOCP"])
        error("""
        Invalid input for relaxation method.
        Relaxation must be either "SDP" or "SOCP"; $relaxation supplied instead.
        """)
    end
    if !(branching_type in ["box", "angular", "polyhedral", "polyhedral_lite"])
        error("""
        Invalid input for branching type.
        Branching type must be either "box" or "angular" or "polyhedral" or "polyhedral_lite"; $branching_type supplied instead.
        """)
    end

    if !(size(A) == size(indices))
        error("""
        Dimension mismatch. 
        Input matrix A must have size (n, m);
        Input matrix indices must have size (n, m).
        """)
    end

    log_time = Dates.now()

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
        Printf.@sprintf("Relaxation:        %10s\n", relaxation),
        Printf.@sprintf("Branching type:    %10s\n", branching_type),
        Printf.@sprintf("Optimality gap:    %10g\n", gap),
        Printf.@sprintf("Maximum nodes:     %10d\n", max_steps),
        Printf.@sprintf("Time limit (s):    %10d\n", time_limit),
        "-----------------------------------------------------------------------------------\n",
        "|   Explored |      Total |  Objective |  Incumbent |        Gap |    Runtime (s) |\n",
        "-----------------------------------------------------------------------------------\n",
    ]
    for message in printlist
        print(stdout, message)
    end

    instance = Dict()
    instance["params"] = Dict(
        "k" => k,
        "m" => m,
        "n" => n,
        "A" => A,
        "indices" => indices,
        "num_indices" => convert(Int, round(sum(indices))),
        "γ" => γ,
        "λ" => λ,
        "relaxation" => relaxation,
        "branching_type" => branching_type,
        "optimality_gap" => gap,
        "max_steps" => max_steps,
        "time_limit" => time_limit,
    )
    instance["run_log"] = DataFrame(
        explored = Int[],
        total = Int[],
        objective = Float64[],
        incumbent = Float64[],
        gap = Float64[],
        runtime = Float64[],
    )

    start_time = time()

    # TODO: better initial Us?
    U_altmin, V_altmin = @suppress alternating_minimization(
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
    MSE_in_initial = compute_MSE(X_initial, A, indices, kind = "in")
    MSE_out_initial = compute_MSE(X_initial, A, indices, kind = "out")
    
    solution = Dict(
        "objective_initial" => objective_initial,
        "MSE_in_initial" => MSE_in_initial,
        "MSE_out_initial" => MSE_out_initial,
        "Y_initial" => Y_initial,
        "U_initial" => U_initial,
        "X_initial" => X_initial,
        "objective" => objective_initial,
        "MSE_in" => MSE_in_initial,
        "MSE_out" => MSE_out_initial,
        "Y" => Y_initial,
        "U" => U_initial,
        "X" => X_initial,
    )

    # (1) number of nodes explored so far
    node_id = 1
    if branching_type == "box"
        U_lower_initial = -ones(n, k)
        U_lower_initial[n,:] .= 0.0
        U_upper_initial = ones(n, k)
        nodes = [(U_lower_initial, U_upper_initial, node_id)]
    elseif branching_type == "angular"
        φ_lower_initial = zeros(n-1, k)
        φ_upper_initial = fill(convert(Float64, pi), (n-1, k))
        nodes = [(φ_lower_initial, φ_upper_initial, node_id)]
    elseif branching_type in ["polyhedral", "polyhedral_lite"]
        φ_lower_initial = zeros(n-1, k)
        φ_upper_initial = fill(convert(Float64, pi), (n-1, k))
        nodes = [(φ_lower_initial, φ_upper_initial, node_id)]
    end

    upper = objective_initial
    lower = -Inf

    lower_bounds = Dict{Integer, Float64}()
    ancestry = []

    # (2) number of nodes generated in total
    counter = 1
    last_updated_counter = 1    
    now_gap = 1e5

    # (3) number of nodes with infeasible relaxation
    nodes_relax_infeasible = 0 # always pruned
    # (4) number of nodes with feasible relaxation,
    # a.k.a. number of relaxations solved
    nodes_relax_feasible = 0 
    # (3) + (4) should yield (2)

    # (5) number of nodes with feasible relaxation
    # that have objective dominated by best upper bound so far
    nodes_relax_feasible_pruned = 0 # always pruned
    # (6) number of nodes with feasible relaxation
    # that are also feasible for the master problem
    nodes_master_feasible = 0 # always pruned
    # (7) number of nodes with feasible relaxation
    # that are also feasible for the master problem,
    # and improve on best upper bound so far
    nodes_master_feasible_improvement = 0 # always pruned
    # (7) ⊂ (6)

    # (8) number of nodes with feasible relaxation,
    # that have objective NOT dominated by best upper bound so far,
    # that are not feasible for the master problem,
    # and are therefore split on
    nodes_relax_feasible_split = 0 # not pruned

    # (5) + (6) + (8) should yield (4)
    # pruned nodes: (3) + (5) + (6)
    # not pruned nodes: (8)

    while (
        now_gap > gap &&
        counter < max_steps &&
        time() - start_time ≤ time_limit
    )
        if length(nodes) != 0
            if branching_type == "box"
                (U_lower, U_upper, node_id) = popfirst!(nodes)
            elseif branching_type == "angular"
                (φ_lower, φ_upper, node_id) = popfirst!(nodes)
                # TODO: conduct feasibility check on (φ_lower, φ_upper) directly
                (U_lower, U_upper) = φ_ranges_to_U_ranges(φ_lower, φ_upper)
            elseif branching_type == "polyhedral"
                (φ_lower, φ_upper, node_id) = popfirst!(nodes)
                polyhedra = φ_ranges_to_polyhedra(φ_lower, φ_upper, false)
            elseif branching_type == "polyhedral_lite"   
                (φ_lower, φ_upper, node_id) = popfirst!(nodes)
                (U_lower, U_upper) = φ_ranges_to_U_ranges(φ_lower, φ_upper)
                polyhedra = φ_ranges_to_polyhedra(φ_lower, φ_upper, true)
            end
        else
            now_gap = add_update!(printlist, instance,node_id, counter, lower, upper, start_time)
            break
        end

        split_flag = true

        if branching_type in ["box", "angular"]
            if !(
                @suppress relax_feasibility_frob_matrixcomp(
                    n, k, relaxation, branching_type;
                    U_lower = U_lower, 
                    U_upper = U_upper
                )
            )
                nodes_relax_infeasible += 1
                split_flag = false
                continue
            end
        elseif branching_type == "polyhedral"
            if !(
                @suppress relax_feasibility_frob_matrixcomp(
                    n, k, relaxation, branching_type;
                    polyhedra = polyhedra
                )
            )
                nodes_relax_infeasible += 1
                split_flag = false
                continue
            end
        elseif branching_type == "polyhedral_lite"
            if !(
                @suppress relax_feasibility_frob_matrixcomp(
                    n, k, relaxation, branching_type;
                    U_lower = U_lower, 
                    U_upper = U_upper,
                    polyhedra = polyhedra
                )
            )
                nodes_relax_infeasible += 1
                split_flag = false
                continue
            end
        end

        # solve SDP / SOCP relaxation of master problem
        if branching_type in ["box", "angular"]
            relax_result = @suppress relax_frob_matrixcomp(n, k, relaxation, branching_type, A, indices, γ, λ; U_lower = U_lower, U_upper = U_upper)
        elseif branching_type == "polyhedral"
            relax_result = @suppress relax_frob_matrixcomp(n, k, relaxation, branching_type, A, indices, γ, λ; polyhedra = polyhedra)
        elseif branching_type == "polyhedral_lite"
            relax_result = @suppress relax_frob_matrixcomp(n, k, relaxation, branching_type, A, indices, γ, λ; U_lower = U_lower, U_upper = U_upper, polyhedra = polyhedra)
        end
        
        if relax_result["feasible"] == false # should not happen, since this should be checked by relax_feasibility_frob_matrixcomp
            nodes_relax_infeasible += 1
            split_flag = false
            continue
        elseif relax_result["termination_status"] in [
            MOI.OPTIMAL,
            MOI.LOCALLY_SOLVED, # TODO: investigate this
            MOI.SLOW_PROGRESS # TODO: investigate this
        ]
            ## TODO: comment these sections on/off to debug MOI.LOCALLY_SOLVED and MOI.SLOW_PROGRESS
            if relax_result["termination_status"] == MOI.SLOW_PROGRESS
                error("""
                Unexpected termination status code: MOI.SLOW_PROGRESS;
                k: $k
                m: $m
                n: $n
                num_indices: $(convert(Int, round(sum(indices))))
                relaxation: $relaxation
                branching_type: $branching_type
                """)
            end
            if relax_result["termination_status"] == MOI.LOCALLY_SOLVED
                error("""
                Unexpected termination status code: MOI.LOCALLY_SOLVED;
                k: $k
                m: $m
                n: $n
                num_indices: $(convert(Int, round(sum(indices))))
                relaxation: $relaxation
                branching_type: $branching_type
                """)
            end
            nodes_relax_feasible += 1
            objective_relax = relax_result["objective"]
            lower_bounds[node_id] = objective_relax
            Y_relax = relax_result["Y"]
            U_relax = relax_result["U"]
            t_relax = relax_result["t"]
            X_relax = relax_result["X"]
            Θ_relax = relax_result["Θ"]
            if node_id == 1
                lower = objective_relax
            end
        end

        # if solution for relax_result has higher objective than best found so far: prune the node
        if objective_relax ≥ solution["objective"]
            nodes_relax_feasible_pruned += 1
            split_flag = false
        end

        # if solution for relax_result is feasible for original problem:
        # prune this node;
        # if it is the best found so far, update solution
        if master_problem_frob_matrixcomp_feasible(Y_relax, U_relax, t_relax, X_relax, Θ_relax)
            nodes_master_feasible += 1
            # if best found so far, update solution
            if objective_relax < solution["objective"]
                nodes_master_feasible_improvement += 1
                solution["objective"] = objective_relax
                upper = objective_relax
                solution["Y"] = copy(Y_relax)
                solution["U"] = copy(U_relax)
                solution["X"] = copy(X_relax)
                now_gap = add_update!(printlist, instance,node_id, counter, lower, upper, start_time)
                last_updated_counter = counter
            end
            split_flag = false
        end

        if split_flag == false
            continue
        end

        # branch on variable
        # for now: branch on biggest element-wise difference between U_lower and U_upper / φ_lower and φ_upper
        nodes_relax_feasible_split += 1
        if branching_type == "box"
            (diff, index) = findmax(U_upper - U_lower)
            mid = U_lower[index] + diff / 2
            U_lower_new = copy(U_lower)
            U_lower_new[index] = mid
            U_upper_new = copy(U_upper)
            U_upper_new[index] = mid
            push!(nodes, (U_lower, U_upper_new, counter + 1))
            push!(nodes, (U_lower_new, U_upper, counter + 2))
            push!(ancestry, (node_id, [counter + 1, counter + 2]))
            counter += 2
        elseif branching_type in ["angular", "polyhedral", "polyhedral_lite"]
            (diff, index) = findmax(φ_upper - φ_lower)
            mid = φ_lower[index] + diff / 2
            φ_lower_new = copy(φ_lower)
            φ_lower_new[index] = mid
            φ_upper_new = copy(φ_upper)
            φ_upper_new[index] = mid
            push!(nodes, (φ_lower, φ_upper_new, counter + 1))
            push!(nodes, (φ_lower_new, φ_upper, counter + 2))
            push!(ancestry, (node_id, [counter + 1, counter + 2]))
            counter += 2
        end

        (anc_node_id, anc_children_node_ids) = ancestry[1]
        if all(haskey(lower_bounds, id) for id in anc_children_node_ids)
            popfirst!(ancestry)
            pop!(lower_bounds, anc_node_id)
            if minimum(values(lower_bounds)) > lower
                lower = minimum(values(lower_bounds))
                now_gap = add_update!(printlist, instance,node_id, counter, lower, upper, start_time)
                last_updated_counter = counter
            end
        end

        if node_id == 1
            now_gap = add_update!(printlist, instance,node_id, counter, lower, upper, start_time)
            last_updated_counter = counter
            if root_only
                break
            end
        end

        if ((counter ÷ update_step) > (last_updated_counter ÷ update_step))
            now_gap = add_update!(printlist, instance,node_id, counter, lower, upper, start_time)
            last_updated_counter = counter
        end
    end

    end_time = time()
    time_taken = end_time - start_time

    solution["MSE_in"] = compute_MSE(solution["X"], A, indices, kind = "in")
    solution["MSE_out"] = compute_MSE(solution["X"], A, indices, kind = "out") 

    instance["run_details"] = Dict(
        "log_time" => log_time,
        "start_time" => start_time,
        "end_time" => end_time,
        "time_taken" => time_taken,
        "nodes_explored" => node_id,
        "nodes_total" => counter,
        "nodes_relax_infeasible" => nodes_relax_infeasible,
        "nodes_relax_feasible" => nodes_relax_feasible,
        "nodes_relax_feasible_pruned" => nodes_relax_feasible_pruned,
        "nodes_master_feasible" => nodes_master_feasible,
        "nodes_master_feasible_improvement" => nodes_master_feasible_improvement,
        "nodes_relax_feasible_split" => nodes_relax_feasible_split,
    )

    push!(
        printlist,
        "\n\nInitial solution (warm start):\n",
        sprint(show, "text/plain", objective_initial),
        "\n\nMSE of sampled entries (warm start):\n",
        sprint(show, "text/plain", MSE_in_initial),
        "\n\nMSE of unsampled entries (warm start):\n",
        sprint(show, "text/plain", MSE_out_initial),
        "\n\nU:\n",
        sprint(show, "text/plain", solution["U"]),
        "\n\nY:\n",
        sprint(show, "text/plain", solution["Y"]),
        "\n\nX:\n",
        sprint(show, "text/plain", solution["X"]),
        "\n\nA:\n",
        sprint(show, "text/plain", A),
        "\n\nindices:\n",
        sprint(show, "text/plain", indices),
        "\n\nBest incumbent solution:\n",
        sprint(show, "text/plain", solution["objective"]),
        "\n\nMSE of sampled entries:\n",
        sprint(show, "text/plain", solution["MSE_in"]),
        "\n\nMSE of unsampled entries:\n",
        sprint(show, "text/plain", solution["MSE_out"]),
    )

    return solution, printlist, instance

end

function master_problem_frob_matrixcomp_feasible(
    Y, 
    U, 
    t, 
    X, 
    Θ,
    ;
    orthogonality_tolerance::Float64 = 0.0, # previous value 1e-5
    projection_tolerance::Float64 = 0.0, # previous value 1e-6
    lifted_variable_tolerance::Float64 = 0.0, # previous value 1e-6
)
    return (
        all( (abs.(U' * U - I)) .≤ orthogonality_tolerance )
        && sum(Y[i,i] for i in 1:size(Y,1)) ≤ size(U, 2)
        && eigvals(Symmetric(Y - U * U'), 1:1)[1] ≥ - projection_tolerance
        && eigvals(Symmetric([Y X; X' Θ]), 1:1)[1] ≥ - lifted_variable_tolerance
    )
end

function relax_feasibility_frob_matrixcomp(
    n::Int,
    k::Int,
    relaxation::String,
    branching_type::String,
    ;
    U_lower::Array{Float64,2} = begin
        U_lower = -ones(n,k)
        U_lower[end,:] .= 0.0
        U_lower
    end,
    U_upper::Array{Float64,2} = ones(n,k),
    polyhedra::Union{Vector, Nothing} = nothing,
    orthogonality_tolerance::Float64 = 0.0,
)
    if !(relaxation in ["SDP", "SOCP"])
        error("""
        Invalid input for relaxation method.
        Relaxation must be either "SDP" or "SOCP"; $relaxation supplied instead.
        """)
    end
    if !(branching_type in ["box", "angular", "polyhedral", "polyhedral_lite"])
        error("""
        Invalid input for branching type.
        Branching type must be either "box" or "angular" or "polyhedral" or "polyhedral_lite"; $branching_type supplied instead.
        """)
    end
    if !(
        size(U_lower) == (n,k)
        && size(U_upper) == (n,k)
        && (
            isnothing(polyhedra)
            || size(polyhedra, 1) == k
        )
    )
        error("""
        Dimension mismatch. 
        Input matrix U_lower must have size (n, k); 
        Input matrix U_upper must have size (n, k);
        If provided, input vector polyhedra must have size (k,).
        """)
    end

    (n, k) = size(U_lower)

    if relaxation == "SDP"
        model = Model(Mosek.Optimizer)
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
    elseif relaxation == "SOCP"
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)
    else
        error("""
        relaxation must be either "SDP" or "SOCP"!
        """)
    end

    @variable(model, U[1:n, 1:k])
    @variable(model, t[1:n, 1:k, 1:k])

    # Lower bounds and upper bounds on U
    @constraint(model, [i=1:n, j=1:k], U_lower[i,j] ≤ U[i,j] ≤ U_upper[i,j])

    # Polyhedral bounds on U, if supplied
    if !isnothing(polyhedra)
        for j in 1:k
            if !isnothing(polyhedra[j])
                @constraint(model, U[:,j] in polyhedra[j])
            end
        end
    end

    # McCormick inequalities at U_lower and U_upper here
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = j1:k],
        t[i, j1, j2] ≥ (
            U_lower[i, j2] * U[i, j1] 
            + U_lower[i, j1] * U[i, j2] 
            - U_lower[i, j1] * U_lower[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = j1:k],
        t[i, j1, j2] ≥ (
            U_upper[i, j2] * U[i, j1] 
            + U_upper[i, j1] * U[i, j2] 
            - U_upper[i, j1] * U_upper[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = j1:k],
        t[i, j1, j2] ≤ (
            U_upper[i, j2] * U[i, j1] 
            + U_lower[i, j1] * U[i, j2] 
            - U_lower[i, j1] * U_upper[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = j1:k],
        t[i, j1, j2] ≤ (
            U_lower[i, j2] * U[i, j1] 
            + U_upper[i, j1] * U[i, j2] 
            - U_upper[i, j1] * U_lower[i, j2]
        )
    )

    # Orthogonality constraints U'U = I using new variables
    for j1 = 1:k, j2 = j1:k
        if (j1 == j2)
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≤ 1.0 + orthogonality_tolerance
            )
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≥ 1.0 - orthogonality_tolerance
            )
        else
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≤ 0.0 + orthogonality_tolerance
            )
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≥ 0.0 - orthogonality_tolerance
            )
        end
    end

    @constraint(
        model,
        [i = 1:n, j1 = 2:k, j2 = 1:(j1-1)],
        t[i,j1,j2] == 0.0
    )

    @objective(
        model,
        Min,
        0
    )

    optimize!(model)

    return (JuMP.termination_status(model) == MOI.OPTIMAL)
end

function relax_frob_matrixcomp(
    n::Int,
    k::Int,
    relaxation::String,
    branching_type::String,
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    U_lower::Array{Float64,2} = begin
        U_lower = -ones(n,k)
        U_lower[end,:] .= 0.0
        U_lower
    end,
    U_upper::Array{Float64,2} = ones(n,k),
    polyhedra::Union{Vector, Nothing} = nothing,
    orthogonality_tolerance::Float64 = 0.0,
    solver_output::Int = 0,
)
    if !(relaxation in ["SDP", "SOCP"])
        error("""
        Invalid input for relaxation method.
        Relaxation must be either "SDP" or "SOCP"; $relaxation supplied instead.
        """)
    end
    if !(branching_type in ["box", "angular", "polyhedral", "polyhedral_lite"])
        error("""
        Invalid input for branching type.
        Branching type must be either "box" or "angular" or "polyhedral" or "polyhedral_lite"; $branching_type supplied instead.
        """)
    end
    if !(
        size(U_lower) == (n,k)
        && size(U_upper) == (n,k)
        && size(A, 1) == size(indices, 1) == n
        && size(A, 2) == size(indices, 2)
        && (
            isnothing(polyhedra)
            || size(polyhedra, 1) == k
        )
    )
        error("""
        Dimension mismatch. 
        Input matrix U_lower must have size (n, k); 
        Input matrix U_upper must have size (n, k); 
        Input matrix A must have size (n, m);
        Input matrix indices must have size (n, m);
        If provided, input vector polyhedra must have size (k,).""")
    end

    (n, k) = size(U_lower)
    (n, m) = size(A)

    if relaxation == "SDP"
        model = Model(Mosek.Optimizer)
        if solver_output == 0
            set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
        end
    elseif relaxation == "SOCP"
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", solver_output)
    end

    @variable(model, X[1:n, 1:m])
    @variable(model, Y[1:n, 1:n], Symmetric)
    @variable(model, Θ[1:m, 1:m], Symmetric)
    @variable(model, U[1:n, 1:k])
    @variable(model, t[1:n, 1:k, 1:k])

    if relaxation == "SDP"
        @constraint(model, LinearAlgebra.Symmetric([Y X; X' Θ]) in PSDCone())
        @constraint(model, LinearAlgebra.Symmetric([Y U; U' I]) in PSDCone())

        @constraint(model, LinearAlgebra.Symmetric(I - Y) in PSDCone())
    elseif relaxation == "SOCP"
        # Second-order cone constraints
        # TODO: see if can improve these by knowledge on bounds on U

        # # Y[i,j]^2 <= Y[i,i] * Y[j,j]
        # @constraint(model, 
        #     [i in 1:n, j in i:n],
        #     [
        #         Y[i,i]; 
        #         0.5 * Y[j,j]; 
        #         Y[i,j]
        #     ] in RotatedSecondOrderCone()
        # )
        # || 2 * Y[i,j]; Y[i,i] - Y[j,j] ||₂ ≤ Y[i,i] + Y[j,j]
        @constraint(model, 
            [i in 1:n, j in i:n],
            [
                Y[i,i] + Y[j,j];
                Y[i,i] - Y[j,j];
                2 * Y[i,j]
            ] in SecondOrderCone()
        )

        # # X[i,j]^2 <= Y[i,i] * Θ[j,j]
        # @constraint(model, 
        #     [i in 1:n, j in 1:m],
        #     [
        #         Y[i,i]; 
        #         0.5 * Θ[j,j]; 
        #         X[i,j]
        #     ] in RotatedSecondOrderCone()
        # )
        # || 2 * X[i,j]; Y[i,i] - Θ[j,j] ||₂ ≤ Y[i,i] + Θ[j,j]
        @constraint(model, 
            [i in 1:n, j in 1:m],
            [
                Y[i,i] + Θ[j,j];
                Y[i,i] - Θ[j,j];
                2 * X[i,j]
            ] in SecondOrderCone()
        )
        
        # # Θ[i,j]^2 <= Θ[i,i] * Θ[j,j]
        # @constraint(model, 
        #     [i in 1:m, j in i:m],
        #     [
        #         Θ[i,i]; 
        #         0.5 * Θ[j,j]; 
        #         Θ[i,j]
        #     ] in RotatedSecondOrderCone()
        # )
        # || 2 * Θ[i,j]; Θ[i,i] - Θ[j,j] ||₂ ≤ Θ[i,i] + Θ[j,j]
        @constraint(model,
            [i in 1:m, j in i:m],
            [
                Θ[i,i] + Θ[j,j];
                Θ[i,i] - Θ[j,j];
                2 * Θ[i,j]
            ] in SecondOrderCone() 
        )
        
        # # Y[i,i] >= sum(U[i,j]^2 for j in 1:k)
        # @constraint(model, 
        #     [i in 1:n],
        #     [
        #         Y[i,i]; 
        #         0.5; 
        #         U[i,:]
        #     ] in RotatedSecondOrderCone()
        # )
        # || 2 * U[i,:]; Y[i,i] - 1 ||₂ ≤ Y[i,i] + 1
        @constraint(model, 
            [i in 1:n],
            [
                Y[i,i] + 1;
                Y[i,i] - 1;
                2 * U[i,:]
            ] in SecondOrderCone()
        )

        # TODO: see if can improve these (McCormick-like) by knowledge on bounds on U
        # (\alpha = +-1 currently but at other nodes? what is the current centerpoint of my box? if i linearize there do i get a better approx?)

        # Adamturk and Gomez:
        # # || U[i,:] + U[j,:] ||²₂ ≤ Y[i,i] + Y[j,j] + 2 * Y[i,j]
        # @constraint(model, 
        #     [i in 1:n, j in i:n],
        #     [
        #         Y[i,i] + Y[j,j] + 2 * Y[i,j];
        #         0.5;
        #         U[i,:] + U[j,:]
        #     ] in RotatedSecondOrderCone()
        # )
        # # || U[i,:] - U[j,:] ||²₂ ≤ Y[i,i] + Y[j,j] - 2 * Y[i,j]
        # @constraint(model, 
        #     [i in 1:n, j in i:n],
        #     [
        #         Y[i,i] + Y[j,j] -+ 2 * Y[i,j];
        #         0.5;
        #         U[i,:] - U[j,:]
        #     ] in RotatedSecondOrderCone()
        # )
        # || 2 * (U[i,:] + U[j,:]); Y[i,i] + Y[j,j] + 2 * Y[i,j] - 1 ||₂ ≤ Y[i,i] + Y[j,j] + 2 * Y[i,j] + 1
        @constraint(model, 
            [i in 1:n, j in i:n],
            [
                Y[i,i] + Y[j,j] + 2 * Y[i,j] + 1;
                Y[i,i] + Y[j,j] + 2 * Y[i,j] - 1;
                2 * (U[i,:] + U[j,:])
            ] in SecondOrderCone()
        )
        # || 2 * (U[i,:] + U[j,:]); Y[i,i] + Y[j,j] - 2 * Y[i,j] - 1 ||₂ ≤ Y[i,i] + Y[j,j] - 2 * Y[i,j] + 1
        @constraint(model, 
            [i in 1:n, j in i:n],
            [
                Y[i,i] + Y[j,j] - 2 * Y[i,j] + 1;
                Y[i,i] + Y[j,j] - 2 * Y[i,j] - 1;
                2 * (U[i,:] - U[j,:])
            ] in SecondOrderCone()
        )
    end

    # Trace constraint on Y
    @constraint(model, sum(Y[i,i] for i in 1:n) <= k)

    # Lower bounds and upper bounds on U
    @constraint(model, [i=1:n, j=1:k], U_lower[i,j] ≤ U[i,j] ≤ U_upper[i,j])

    # Polyhedral bounds on U, if supplied
    if !isnothing(polyhedra)
        for j in 1:k
            if !isnothing(polyhedra[j])
                @constraint(model, U[:,j] in polyhedra[j])
            end
        end
    end
        
    # McCormick inequalities at U_lower and U_upper here
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = j1:k],
        t[i, j1, j2] ≥ (
            U_lower[i, j2] * U[i, j1] 
            + U_lower[i, j1] * U[i, j2] 
            - U_lower[i, j1] * U_lower[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = j1:k],
        t[i, j1, j2] ≥ (
            U_upper[i, j2] * U[i, j1] 
            + U_upper[i, j1] * U[i, j2] 
            - U_upper[i, j1] * U_upper[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = j1:k],
        t[i, j1, j2] ≤ (
            U_upper[i, j2] * U[i, j1] 
            + U_lower[i, j1] * U[i, j2] 
            - U_lower[i, j1] * U_upper[i, j2]
        )
    )
    @constraint(
        model,
        [i = 1:n, j1 = 1:k, j2 = j1:k],
        t[i, j1, j2] ≤ (
            U_lower[i, j2] * U[i, j1] 
            + U_upper[i, j1] * U[i, j2] 
            - U_upper[i, j1] * U_lower[i, j2]
        )
    )

    # Orthogonality constraints U'U = I using new variables
    for j1 = 1:k, j2 = j1:k
        if (j1 == j2)
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≤ 1.0 + orthogonality_tolerance
            )
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≥ 1.0 - orthogonality_tolerance
            )
        else
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≤ 0.0 + orthogonality_tolerance
            )
            @constraint(
                model,
                sum(t[i, j1, j2] for i = 1:n) ≥ 0.0 - orthogonality_tolerance
            )
        end
    end
    
    @constraint(
        model,
        [i = 1:n, j1 = 2:k, j2 = 1:(j1-1)],
        t[i,j1,j2] == 0.0
    )

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

    if JuMP.termination_status(model) in [
        MOI.OPTIMAL,
        MOI.LOCALLY_SOLVED,
    ]
        results = Dict(
            "feasible" => true,
            "termination_status" => JuMP.termination_status(model),
            "objective" => objective_value(model),
            "Y" => value.(Y),
            "U" => value.(U),
            "t" => value.(t),
            "X" => value.(X),
            "Θ" => value.(Θ),
        )
    elseif JuMP.termination_status(model) in [
        MOI.INFEASIBLE,
        MOI.DUAL_INFEASIBLE,
        MOI.LOCALLY_INFEASIBLE,
        MOI.INFEASIBLE_OR_UNBOUNDED,
    ]
        results = Dict(
            "feasible" => false,
            "termination_status" => JuMP.termination_status(model),
        )
    elseif JuMP.termination_status(model) == MOI.SLOW_PROGRESS
        if has_values(model)
            results = Dict(
                "feasible" => true,
                "termination_status" => JuMP.termination_status(model),
                "objective" => objective_value(model),
                "Y" => value.(Y),
                "U" => value.(U),
                "t" => value.(t),
                "X" => value.(X),
                "Θ" => value.(Θ),
            )
        else
            results = Dict(
                "feasible" => false,
                "termination_status" => JuMP.termination_status(model),
                "model" => model,
            )
        end
    else
        error("""
        unexpected termination status: $(JuMP.termination_status(model))
        """)
    end

    return results
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
                for i in 1:n, j in 1:m
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
                for i in 1:n, j in 1:m
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

function compute_MSE(X, A, indices; kind = "out")
    """Computes MSE of entries in `X` and `A` that are not in `indices`."""
    if kind == "out"
        if length(indices) == sum(indices)
            return 0.0
        else
            return sum((X - A).^2 .* (1 .- indices)) / (length(indices) - sum(indices))
        end
    elseif kind == "in"
        if sum(indices) == 0.0
            return 0.0
        else
            return sum((X - A).^2 .* indices) / sum(indices)
        end
    elseif kind == "all"
        return sum((X - A).^2) / length(indices)
    else
        error("""
        Input argument `kind` not recognized!
        Must be one of "out", "in", or "all".
        """)
    end
end

function φ_ranges_to_U_ranges(
    φ_lower::Array{Float64,2},
    φ_upper::Array{Float64,2},
)

    function φ_to_cos(
        φ_L::Float64,
        φ_U::Float64,
    )
        if !(
            0 ≤ φ_L ≤ φ_U ≤ pi
        )
            error("""
            Domain error.
            Input value φ_L must be in range [0, π];
            Input value φ_U must be in range [0, π];
            φ_L and φ_U must satisfy φ_L ≤ φ_U.
            """)
        end
        return [cos(φ_U), cos(φ_L)]
    end

    function φ_to_sin(
        φ_L::Float64,
        φ_U::Float64,
    )
        if !(
            0 ≤ φ_L ≤ φ_U ≤ pi
        )
            error("""
            Domain error.
            Input value φ_L must be in range [0, π];
            Input value φ_U must be in range [0, π];
            φ_L and φ_U must satisfy φ_L ≤ φ_U.
            """)
        end
        if φ_U ≤ pi / 2
            return [sin(φ_L), sin(φ_U)]
        elseif pi / 2 ≤ φ_L
            return [sin(φ_U), sin(φ_L)]
        else
            return [min(sin(φ_U), sin(φ_L)), 1.0]
        end
    end


    if !(
        size(φ_lower) == size(φ_upper)
    )
        error("""
        Dimension mismatch. 
        Input matrix φ_lower must have size (n-1, k); 
        Input matrix φ_upper must have size (n-1, k).
        """)
    end

    n = size(φ_lower, 1) + 1
    k = size(φ_lower, 2)
    
    U_lower = ones(n, k)
    U_upper = ones(n, k)

    for j in 1:k
        cos_column = reduce(hcat, [
            φ_to_cos(φ_L, φ_U)
            for (φ_L, φ_U) in zip(φ_lower[:,j], φ_upper[:,j])
        ])
        sin_column = reduce(hcat, [
            φ_to_sin(φ_L, φ_U)
            for (φ_L, φ_U) in zip(φ_lower[:,j], φ_upper[:,j])
        ])

        for i in 1:(n-1)
            U_lower[i,j] = cos_column[1,i]
            U_upper[i,j] = cos_column[2,i]
            for i2 in 1:(i-1)
                # multiply by sin_lower or sin_upper depending on sign
                if 0 ≤ U_lower[i,j]
                    U_lower[i,j] *= sin_column[1,i2]
                else
                    U_lower[i,j] *= sin_column[2,i2]
                end
                if 0 ≤ U_upper[i,j]
                    U_upper[i,j] *= sin_column[2,i2]
                else
                    U_upper[i,j] *= sin_column[1,i2]
                end
            end
        end
        for i2 in 1:(n-1)
            U_lower[n,j] *= sin_column[1,i2]
            U_upper[n,j] *= sin_column[2,i2]
        end
    end

    return (U_lower, U_upper)
end

function product_ranges(
    a_L::Float64,
    a_U::Float64,
    b_L::Float64,
    b_U::Float64,
)
    if !(
        a_L ≤ a_U
        && b_L ≤ b_U
    )
        error("""
        Domain error.
        """)
    end
    if 0 ≤ a_L
        if 0 ≤ b_L
            return [a_L * b_L, a_U * b_U]
        elseif b_U ≤ 0
            return [a_U * b_L, a_L * b_U]
        else # b_L < 0 < b_U
            return [a_U * b_L, a_U * b_U]
        end
    elseif a_U ≤ 0
        if 0 ≤ b_L
            return [a_L * b_U, a_U * b_L]
        elseif b_U ≤ 0
            return [a_U * b_U, a_L * b_L]
        else # b_L < 0 < b_U
            return [a_L * b_U, a_L * b_L]
        end
    else # a_L < 0 < a_U
        if 0 ≤ b_L
            return [a_L * b_U, a_U * b_U]
        elseif b_U ≤ 0
            return [a_U * b_L, a_L * b_L]
        else # b_L < 0 < b_U
            return [min(a_U * b_L, a_L * b_U), max(a_L * b_L, a_U * b_U)]
        end
    end
end

function φ_ranges_to_polyhedra(
    φ_lower::Array{Float64,2}, 
    φ_upper::Array{Float64,2}, 
    lite::Bool,
)
    
    function angles_to_vector(
        ϕ::Vector{Float64},
    )
        n = size(ϕ, 1) + 1
        vector = ones(n)
        for (i, a) in enumerate(ϕ)
            (c, s) = (cos(a), sin(a))
            vector[i] *= c
            for j in (i+1):n
                vector[j] *= s
            end
        end
        return vector
    end

    function index_to_angles(
        α::Vector{Int},
        gamma::Int,
    )
        return [a * pi / 2^gamma for a in α]
    end

    function angles_to_facet(
        ϕ1::Vector{Float64}, 
        ϕ2::Vector{Float64},
    )
        return [
            angles_to_vector(collect(β))
            for β in Iterators.product(
                collect(
                    Iterators.zip(ϕ1, ϕ2)
                )...
            )
        ]
    end

    function angles_to_facet_lite(
        ϕ1::Vector{Float64}, 
        ϕ2::Vector{Float64},
    )
        inds = []
        for (i1, i2) in zip(ϕ1, ϕ2)
            if !isapprox(i1, 0.0, atol=1e-14)
                push!(inds, 1)
            elseif !isapprox(i2, pi, atol=1e-14)
                push!(inds, 2)
            else
                error()
            end
        end
        ϕref = [
            (ind == 1) ? ϕ1[i] : ϕ2[i]
            for (i, ind) in enumerate(inds)
        ]
        angles = [ϕref]
        for (i, ind) in enumerate(inds)
            ϕ = copy(ϕref)
            ϕ[i] = (
                (ind == 1) ? ϕ2[i] : ϕ1[i]
            )
            push!(angles, ϕ)
        end
        return [
            angles_to_vector(collect(β))
            for β in angles
        ]
    end

    function indexes_to_facet(
        α1::Vector{Int},
        α2::Vector{Int},
        gamma::Int,
        lite::Bool,
    )
        if lite
            return angles_to_facet_lite(
                index_to_angles(α1, gamma), 
                index_to_angles(α2, gamma),
            )
        else
            return angles_to_facet(
                index_to_angles(α1, gamma), 
                index_to_angles(α2, gamma),
            )
        end
    end

    function angles_to_halfspace(        
        ϕ1::Vector{Float64}, 
        ϕ2::Vector{Float64},
    )
        # Returns the Polyhedra.HalfSpace defined by ϕ1, ϕ2
        # which does not contain the origin
        f_lite = angles_to_facet_lite(ϕ1, ϕ2)
        n = size(f_lite[1], 1)
        model = Model(Gurobi.Optimizer)
        @variable(model, c[1:n])
        @constraint(model, [i=2:n], Compat.dot(c, (f_lite[1] - f_lite[i])) == 0.0)
        @constraint(model, sum(c) == 1)
        @objective(model, Min, Compat.dot(c, f_lite[1]))
        optimize!(model)
        c = value.(c)
        b = objective_value(model)
        if b < 0
            return HalfSpace(c, b)
        elseif b > 0
            return HalfSpace(-c, -b)
        end
    end

    function facet_to_pol(f)
        p = polyhedron(
            conichull(f...) 
            + convexhull(f...)
        )
        for knot in f
            p = p ∩ HalfSpace(knot, 1)
        end
        return p
    end

    function angles_to_pol_lite(ϕ1, ϕ2)
        if isapprox(maximum(ϕ2 - ϕ1), pi, atol = 1e-14)
            return nothing
        end
        # TODO: explore more knots
        knot = angles_to_vector((ϕ1 + ϕ2) / 2) 
        p = @suppress angles_to_halfspace(ϕ1, ϕ2)
        return p ∩ HalfSpace(knot, 1)
    end
    
    if !(
        size(φ_lower) == size(φ_upper)
    )
        error("""
        Dimension mismatch.
        Input matrix φ_lower must have size (n-1, k); 
        Input matrix φ_upper must have size (n-1, k).
        """)
    end

    n = size(φ_lower, 1) + 1
    k = size(φ_lower, 2)
    
    if lite
        polyhedra = [
            angles_to_pol_lite(
                φ_lower[:,j],
                φ_upper[:,j],
            )
            for j in 1:k
        ]
    else
        polyhedra = [
            facet_to_pol(
                angles_to_facet(
                    φ_lower[:,j],
                    φ_upper[:,j],
                )
            )
            for j in 1:k
        ]
    end
    return polyhedra
end
