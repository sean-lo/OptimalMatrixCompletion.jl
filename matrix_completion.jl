using LinearAlgebra
using Arpack
using Random
using Compat

using Printf
using Dates
using Suppressor
using DataFrames
using OrderedCollections
using Parameters

using JuMP
using MathOptInterface
using Gurobi
using Mosek
using MosekTools
using Polyhedra

@with_kw mutable struct BBNode
    U_lower::Union{Matrix{Float64}, Nothing} = nothing
    U_upper::Union{Matrix{Float64}, Nothing} = nothing
    φ_lower::Union{Matrix{Float64}, Nothing} = nothing
    φ_upper::Union{Matrix{Float64}, Nothing} = nothing
    matrix_cuts::Union{Vector{Tuple}, Nothing} = nothing
    LB::Union{Float64, Nothing} = nothing
    node_id::Int
    parent_id::Int
end

function branchandbound_frob_matrixcomp(
    k::Int,
    A::Array{Float64,2},
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    branching_region::String = "box", # region of branching to use; either "box" or "angular" or "polyhedral" or "hybrid"
    branching_type::String = "lexicographic", # determining which coordinate to branch on: either "lexicographic" or "bounds" or "gradient"
    branch_point::String = "midpoint", # determine which value to branch on: either "midpoint" or "current_point"
    node_selection::String = "breadthfirst", # determining which node selection strategy to use: either "breadthfirst" or "bestfirst" or "depthfirst"
    gap::Float64 = 1e-6, # optimality gap for algorithm (proportion)
    use_matrix_cuts::Bool = true,
    root_only::Bool = false, # if true, only solves relaxation at root node
    altmin_flag::Bool = true,
    max_steps::Int = 1000000,
    time_limit::Int = 3600, # time limit in seconds
    update_step::Int = 1000,
)


    function add_update!(
        printlist, instance, nodes_explored, counter, 
        lower, upper, start_time,
        ;
        altmin_flag::Bool = false
    )
        now_gap = (upper / lower) - 1
        current_time_elapsed = time() - start_time
        message = Printf.@sprintf(
            "| %10d | %10d | %10f | %10f | %10f | %10.3f  s  |",
            nodes_explored, # Explored
            counter, # Total
            lower, # Objective
            upper, # Incumbent
            now_gap, # Gap
            current_time_elapsed, # Runtime
        )
        if altmin_flag
            message *= " - A\n"
        else
            message *= "\n"
        end
        print(stdout, message)
        push!(printlist, message)
        push!(
            instance["run_log"],
            (nodes_explored, counter, lower, upper, now_gap, current_time_elapsed)
        )
        return now_gap
    end

    if !(branching_region in ["box", "angular", "polyhedral", "hybrid"])
        error("""
        Invalid input for branching region.
        Branching region must be either "box" or "angular" or "polyhedral" or "hybrid"; $branching_region supplied instead.
        """)
    end
    if !(branching_type in ["lexicographic", "bounds", "gradient"])
        error("""
        Invalid input for branching type.
        Branching type must be either "lexicographic" or "bounds" or "gradient"; $branching_type supplied instead.
        """)
    end
    if !(branch_point in ["midpoint", "current_point"])
        error("""
        Invalid input for branch point.
        Branch point must be either "midpoint" or "current_point"; $branch_point supplied instead.
        """)
    end
    if !(node_selection in ["breadthfirst", "bestfirst", "depthfirst"])
        error("""
        Invalid input for node selection.
        Node selection must be either "breadthfirst" or "bestfirst" or "depthfirst"; $node_selection supplied instead.
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
    Random.seed!(0)

    (n, m) = size(A)
    printlist = [
        Dates.format(log_time, "e, dd u yyyy HH:MM:SS"), "\n",
        "Starting branch-and-bound on a matrix completion problem.\n",
        Printf.@sprintf("k:                 %15d\n", k),
        Printf.@sprintf("m:                 %15d\n", m),
        Printf.@sprintf("n:                 %15d\n", n),
        Printf.@sprintf("num_indices:       %15d\n", sum(indices)),
        Printf.@sprintf("γ:                 %15g\n", γ),
        Printf.@sprintf("λ:                 %15g\n", λ),
        Printf.@sprintf("Branching region:  %15s\n", branching_region),
        Printf.@sprintf("Branching type:    %15s\n", branching_type),
        Printf.@sprintf("Branching point:   %15s\n", branch_point),
        Printf.@sprintf("Node selection:    %15s\n", node_selection),
        Printf.@sprintf("Use matrix cuts?:  %15s\n", use_matrix_cuts),
        Printf.@sprintf("Optimality gap:    %15g\n", gap),
        Printf.@sprintf("Maximum nodes:     %15d\n", max_steps),
        Printf.@sprintf("Time limit (s):    %15d\n", time_limit),
        "-----------------------------------------------------------------------------------\n",
        "|   Explored |      Total |      Lower |      Upper |        Gap |    Runtime (s) |\n",
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
        "branching_region" => branching_region,
        "node_selection" => node_selection,
        "use_matrix_cuts" => use_matrix_cuts,
        "optimality_gap" => gap,
        "max_steps" => max_steps,
        "time_limit" => time_limit,
    )
    instance["run_log"] = DataFrame(
        explored = Int[],
        total = Int[],
        lower = Float64[],
        upper = Float64[],
        gap = Float64[],
        runtime = Float64[],
    )

    start_time = time()
    solve_time_relaxation_feasibility = 0.0
    solve_time_relaxation = 0.0
    solve_time_altmin = 0.0
    solve_time_U_ranges = 0.0
    solve_time_polyhedra = 0.0

    # TODO: better initial Us?
    altmin_A_initial = zeros(n, m)
    for i in 1:n, j in 1:m
        if indices[i,j] == 1
            altmin_A_initial[i,j] = A[i,j]
        end
    end
    altmin_U_initial, _, _ = svd(altmin_A_initial)

    altmin_results = @suppress alternating_minimization(
        A, n, k, indices, γ, λ,
        ;
        U_initial = altmin_U_initial,
    )
    solve_time_altmin = altmin_results["solve_time"]
    # do a re-SVD on U * V in order to recover orthonormal U
    X_initial = altmin_results["U"] * altmin_results["V"]
    U_initial = svd(X_initial).U[:,1:k]
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

    ranges = []

    nodes = []
    upper = objective_initial
    lower = -Inf
    if branching_region == "box"
        U_lower_initial = -ones(n, k)
        U_lower_initial[n,:] .= 0.0
        U_upper_initial = ones(n, k)
        initial_node = BBNode(
            U_lower = U_lower_initial, 
            U_upper = U_upper_initial, 
            matrix_cuts = [],
            node_id = 1,
            LB = lower,
            parent_id = 0,
        )
    elseif branching_region in ["angular", "polyhedral", "hybrid"]
        φ_lower_initial = zeros(n-1, k)
        φ_upper_initial = fill(convert(Float64, pi), (n-1, k))
        initial_node = BBNode(
            φ_lower = φ_lower_initial, 
            φ_upper = φ_upper_initial, 
            node_id = 1,
            LB = lower,
            parent_id = 0,
        )
    end
    push!(nodes, initial_node)

    # leaves' mapping from node_id to lower_bound
    lower_bounds = Dict{Int, Float64}() 
    minimum_lower_bounds = Inf
    ancestry = Dict{Int, Vector{Int}}()

    # (1) number of nodes explored so far
    nodes_explored = 0
    # (2) number of nodes generated in total
    counter = 1
    last_updated_counter = 1    
    now_gap = 1e5

    # (3) number of nodes whose parent has 
    # relaxation dominated by best solution found so far
    nodes_dominated = 0 # always pruned
    # (4) number of nodes with infeasible relaxation
    nodes_relax_infeasible = 0 # always pruned
    # (5) number of nodes with feasible relaxation,
    # a.k.a. number of relaxations solved
    nodes_relax_feasible = 0 
    # (3) + (4) + (5) should yield (1)

    # (6) number of nodes with feasible relaxation
    # that have objective dominated by best upper bound so far
    nodes_relax_feasible_pruned = 0 # always pruned
    # (7) number of nodes with feasible relaxation
    # that are also feasible for the master problem
    nodes_master_feasible = 0 # always pruned
    # (8) number of nodes with feasible relaxation
    # that are also feasible for the master problem,
    # and improve on best upper bound so far
    nodes_master_feasible_improvement = 0 # always pruned
    # (8) ⊂ (7)

    # (9) number of nodes with feasible relaxation,
    # that have objective NOT dominated by best upper bound so far,
    # that are not feasible for the master problem,
    # and are therefore split on
    nodes_relax_feasible_split = 0 # not pruned

    # (6) + (7) + (9) should yield (5)
    # pruned nodes: (4) + (6) + (7)
    # not pruned nodes: (9)

    while (
        now_gap > gap &&
        counter < max_steps &&
        time() - start_time ≤ time_limit
    )
        if length(nodes) != 0
            if node_selection == "breadthfirst"
                current_node = popfirst!(nodes)
            elseif node_selection == "bestfirst"
                # break ties by depth-first search (choosing the node most recently added to queue)
                (min_LB, min_LB_index) = findmin(node.LB for node in reverse(nodes))
                current_node = popat!(
                    nodes, 
                    length(nodes) + 1 - min_LB_index,
                )
            elseif node_selection == "depthfirst" # NOTE: may not work well
                current_node = pop!(nodes)
            end
            nodes_explored += 1
        else
            break
        end

        if branching_region == "box"
            nothing
        elseif branching_region == "angular"
            # TODO: conduct feasibility check on (φ_lower, φ_upper) directly
            U_ranges_results = φ_ranges_to_U_ranges(
                current_node.φ_lower, 
                current_node.φ_upper,
            )
            current_node.U_lower = U_ranges_results["U_lower"]
            current_node.U_upper = U_ranges_results["U_upper"]
            solve_time_U_ranges += U_ranges_results["time_taken"]
        elseif branching_region == "polyhedral"
            polyhedra_results = φ_ranges_to_polyhedra(
                current_node.φ_lower, 
                current_node.φ_upper, 
                false,
            )
            polyhedra = polyhedra_results["polyhedra"]
            solve_time_polyhedra += polyhedra_results["time_taken"]
        elseif branching_region == "hybrid"
            φ_lower = current_node.φ_lower
            φ_upper = current_node.φ_upper
            U_ranges_results = φ_ranges_to_U_ranges(
                current_node.φ_lower, 
                current_node.φ_upper,
            )
            current_node.U_lower = U_ranges_results["U_lower"]
            current_node.U_upper = U_ranges_results["U_upper"]
            solve_time_U_ranges += U_ranges_results["time_taken"]
            polyhedra_results = φ_ranges_to_polyhedra(
                current_node.φ_lower, 
                current_node.φ_upper, 
                true,
            )
            polyhedra = polyhedra_results["polyhedra"]
            solve_time_polyhedra += polyhedra_results["time_taken"]
        end

        split_flag = true

        # possible, since we may not explore tree breadth-first
        # (should not be possible for breadth-first search)
        if current_node.LB > solution["objective"]
            split_flag = false
            nodes_dominated += 1
        end

        if !use_matrix_cuts && split_flag
            if branching_region in ["box", "angular"]
                relax_feasibility_result = @suppress relax_feasibility_frob_matrixcomp(
                    n, k, branching_region;
                    U_lower = current_node.U_lower, 
                    U_upper = current_node.U_upper,
                )
            elseif branching_region == "polyhedral"
                relax_feasibility_result = @suppress relax_feasibility_frob_matrixcomp(
                    n, k, branching_region;
                    polyhedra = polyhedra,
                )
            elseif branching_region == "hybrid"
                relax_feasibility_result = @suppress relax_feasibility_frob_matrixcomp(
                    n, k, branching_region;
                    U_lower = current_node.U_lower, 
                    U_upper = current_node.U_upper,
                    polyhedra = polyhedra,
                )
            end
            solve_time_relaxation_feasibility += relax_feasibility_result["time_taken"]
            if !relax_feasibility_result["feasible"]
                nodes_relax_infeasible += 1
                split_flag = false
            end
        end

        # solve SDP relaxation of master problem
        if split_flag
            matrix_cuts = nothing
            if use_matrix_cuts
                matrix_cuts = current_node.matrix_cuts
            end               
            if branching_region == "box"
                relax_result = @suppress relax_frob_matrixcomp(
                    n, k, branching_region, A, indices, γ, λ; 
                    U_lower = current_node.U_lower, 
                    U_upper = current_node.U_upper,
                    matrix_cuts = matrix_cuts,
                )
            elseif branching_region == "angular"
                relax_result = @suppress relax_frob_matrixcomp(
                    n, k, branching_region, A, indices, γ, λ; 
                    U_lower = current_node.U_lower, 
                    U_upper = current_node.U_upper,
                )
            elseif branching_region == "polyhedral"
                relax_result = @suppress relax_frob_matrixcomp(
                    n, k, branching_region, A, indices, γ, λ; 
                    polyhedra = polyhedra,
                )
            elseif branching_region == "hybrid"
                relax_result = @suppress relax_frob_matrixcomp(
                    n, k, branching_region, A, indices, γ, λ; 
                    U_lower = current_node.U_lower, 
                    U_upper = current_node.U_upper, 
                    polyhedra = polyhedra,
                )
            end
            solve_time_relaxation += relax_result["solve_time"]
            if relax_result["feasible"] == false # should not happen, since this should be checked by relax_feasibility_frob_matrixcomp
                nodes_relax_infeasible += 1
                split_flag = false
            elseif relax_result["termination_status"] in [
                MOI.OPTIMAL,
                MOI.LOCALLY_SOLVED, # TODO: investigate this
                MOI.SLOW_PROGRESS # TODO: investigate this
            ]
                nodes_relax_feasible += 1
                objective_relax = relax_result["objective"]
                lower_bounds[current_node.node_id] = objective_relax
                Y_relax = relax_result["Y"]
                U_relax = relax_result["U"]
                X_relax = relax_result["X"]
                Θ_relax = relax_result["Θ"]
                α_relax = relax_result["α"]
                if current_node.node_id == 1
                    lower = objective_relax
                end
                # if solution for relax_result has higher objective than best found so far: prune the node
                if objective_relax ≥ solution["objective"]
                    nodes_relax_feasible_pruned += 1
                    delete!(lower_bounds, current_node.node_id)
                    split_flag = false            
                end
            end
        end

        # if solution for relax_result is feasible for original problem:
        # prune this node;
        # if it is the best found so far, update solution
        if split_flag
            if master_problem_frob_matrixcomp_feasible(Y_relax, U_relax, X_relax, Θ_relax, use_matrix_cuts)
                nodes_master_feasible += 1
                # if best found so far, update solution
                if objective_relax < solution["objective"]
                    nodes_master_feasible_improvement += 1
                    solution["objective"] = objective_relax
                    upper = objective_relax
                    solution["Y"] = copy(Y_relax)
                    solution["U"] = copy(U_relax)
                    solution["X"] = copy(X_relax)
                    now_gap = add_update!(
                        printlist, instance, nodes_explored, counter, lower, upper, start_time,
                    )
                    last_updated_counter = counter
                end
                split_flag = false
            end
        end

        # perform alternating minimization heuristic
        altmin_flag_now = altmin_flag && (rand() > 0.95)
        if split_flag
            if altmin_flag_now
                # TODO: check if it's not extremely close to projs already
                # round Y, obtaining a Y_rounded
                U_rounded = cholesky(relax_result["Y"]).U[1:k, 1:n]'
                Y_rounded = U_rounded * U_rounded'
                altmin_results_BB = @suppress alternating_minimization(
                    A, n, k, indices, γ, λ;
                    U_initial = Matrix(U_rounded),
                    U_lower = current_node.U_lower,
                    U_upper = current_node.U_upper,
                )

                solve_time_altmin += altmin_results_BB["solve_time"]
                X_local = altmin_results_BB["U"] * altmin_results_BB["V"]
                U_local = svd(X_local).U[:,1:k] 
                # no guarantees that this will be within U_lower and U_upper
                Y_local = U_local * U_local'
                # guaranteed to be a projection matrix since U_local is a svd result
                
                objective_local = objective_function(
                    X_local, A, indices, U_local, γ, λ,
                )

                if objective_local < solution["objective"]
                    # TODO: include a counter here
                    solution["objective"] = objective_local
                    upper = objective_local
                    solution["Y"] = copy(Y_local)
                    solution["U"] = copy(U_local)
                    solution["X"] = copy(X_local)
                    now_gap = add_update!(
                        printlist, instance, nodes_explored, counter, lower, upper, start_time,
                        ; 
                        altmin_flag = true,
                    )
                    last_updated_counter = counter
                end

            end
        end

        if split_flag
            # branch on variable
            # for now: branch on biggest element-wise difference between U_lower and U_upper / φ_lower and φ_upper
            nodes_relax_feasible_split += 1
            if branching_region == "box"
                if use_matrix_cuts
                    append!(nodes, 
                        collect(create_matrix_cut_child_nodes(
                            Y_relax, U_relax,
                            current_node.U_lower, current_node.U_upper,
                            current_node.matrix_cuts,
                            counter, current_node.node_id, 
                            objective_relax,
                        ))
                    )
                    ancestry[current_node.node_id] = [counter + i for i in 1:2^k]
                    counter += 2^k
                else
                    # preliminaries: defaulting to branching_type
                    
                    # finding coordinates (i, j) to branch on
                    if (
                        branching_type == "lexicographic"
                        ||
                        any((current_node.U_upper .< 0.0) .| (current_node.U_lower .> 0.0))
                    )
                        (_, ind) = findmax(current_node.U_upper - current_node.U_lower)
                    elseif branching_type == "bounds" # TODO: UNTESTED
                        (_, ind) = findmin(
                            min.(
                                current_node.U_upper - U_relax,
                                U_relax - current_node.U_lower,
                            ) ./ (
                                current_node.U_upper - current_node.U_lower
                            )
                        )
                    elseif branching_type == "gradient"
                        deriv_U = - γ * α_relax * α_relax' * U_relax # shape: (n, k)
                        deriv_U_change = zeros(n,k)
                        for i in 1:n, j in 1:k
                            if deriv_U[i,j] < 0.0
                                deriv_U_change[i,j] = deriv_U[i,j] * (
                                    current_node.U_upper[i,j] - U_relax[i,j]
                                )
                            else
                                deriv_U_change[i,j] = deriv_U[i,j] * (
                                    current_node.U_lower[i,j] - U_relax[i,j]
                                )
                            end
                        end
                        (_, ind) = findmin(deriv_U_change)
                    end
                    # finding branch_val)
                    if any((current_node.U_upper .< 0.0) .| (current_node.U_lower .> 0.0))
                        diff = current_node.U_upper[ind] - current_node.U_lower[ind]
                        branch_val = current_node.U_lower[ind] + diff / 2
                    elseif branching_type == "bounds" # custom branch_point
                        if (current_node.U_upper[ind] - U_relax[ind] <
                            U_relax[ind] - current_node.U_lower[ind])
                            branch_val = U_relax[ind] - (current_node.U_upper[ind] - U_relax[ind])
                        else
                            branch_val = U_relax[ind] + (U_relax[ind] - current_node.U_lower[ind])
                        end
                    elseif branch_point == "midpoint"
                        diff = current_node.U_upper[ind] - current_node.U_lower[ind]
                        branch_val = current_node.U_lower[ind] + diff / 2
                    elseif branch_point == "current_point"
                        branch_val = U_relax[ind]
                    end
                    # constructing child nodes
                    U_lower_left = current_node.U_lower
                    U_upper_left = copy(current_node.U_upper)
                    U_upper_left[ind] = branch_val
                    U_lower_right = copy(current_node.U_lower)
                    U_lower_right[ind] = branch_val
                    U_upper_right = current_node.U_upper
                    left_child_node = BBNode(
                        U_lower = U_lower_left,
                        U_upper = U_upper_left,
                        node_id = counter + 1,
                        # initialize a node's LB with the objective of relaxation of parent
                        LB = objective_relax,
                        parent_id = current_node.node_id,
                    )
                    right_child_node = BBNode(
                        U_lower = U_lower_right,
                        U_upper = U_upper_right,
                        node_id = counter + 2,
                        # initialize a node's LB with the objective of relaxation of parent
                        LB = objective_relax,
                        parent_id = current_node.node_id,
                    )
                    push!(nodes, left_child_node, right_child_node)
                    ancestry[current_node.node_id] = [counter + 1, counter + 2]
                    counter += 2
                end
            elseif branching_region in ["angular", "polyhedral", "hybrid"]
                φ_relax = zeros(n-1, k)
                for j in 1:k
                    φ_relax[:,j] = U_col_to_φ_col(U_relax[:,j])
                end
                # if !all(current_node.φ_lower .≤ φ_relax .≤ current_node.φ_upper)
                #     println(current_node.φ_lower)
                #     println(φ_relax)
                #     println(current_node.φ_upper)
                #     error("""""")
                # end
                # finding coordinates (i, j) to branch on
                if (
                    branching_type == "lexicographic"
                    ||
                    any((current_node.U_upper .< 0.0) .| (current_node.U_lower .> 0.0))
                )
                    (_, ind) = findmax(current_node.φ_upper - current_node.φ_lower)
                elseif branching_type == "bounds" # TODO: UNTESTED
                    # error("""
                    # Branching type "box" not yet implemented for "angular", "polyhedral", or "hybrid" branching regions.
                    # """)
                    (_, ind) = findmin(
                        min.(
                            # WARNING: it's possible for φ_relax to be outside the ranges elementwise
                            current_node.φ_upper - φ_relax,
                            φ_relax - current_node.φ_lower,
                        ) ./ (
                            current_node.φ_upper - current_node.φ_lower
                        )
                    )
                elseif branching_type == "gradient"
                    deriv_U = - γ * α_relax * α_relax' * U_relax # shape: (n, k)
                    deriv_φ = zeros(n-1, k)
                    for j in 1:k
                        deriv_φ[:,j] = compute_jacobian(φ_relax[:,j])' * deriv_U[:,j]
                    end
                    deriv_φ_change = zeros(n-1,k)
                    for i in 1:n-1, j in 1:k
                        if deriv_φ[i,j] < 0.0
                            deriv_φ_change[i,j] = deriv_φ[i,j] * (
                                current_node.φ_upper[i,j] - φ_relax[i,j]
                            )
                        else
                            deriv_φ_change[i,j] = deriv_φ[i,j] * (
                                current_node.φ_lower[i,j] - φ_relax[i,j]
                            )
                        end
                    end
                    (_, ind) = findmin(deriv_φ_change)
                end
                # finding branch_val
                if (
                    any((current_node.U_upper .< 0.0) .| (current_node.U_lower .> 0.0))
                    || # it's possible for φ_relax to be outside the ranges elementwise
                    current_node.φ_lower[ind] > φ_relax[ind]
                    ||
                    φ_relax[ind] > current_node.φ_upper[ind]
                )
                    diff = current_node.φ_upper[ind] - current_node.φ_lower[ind]
                    branch_val = current_node.φ_lower[ind] + diff / 2
                elseif branching_type == "bounds"
                    if (current_node.φ_upper[ind] - φ_relax[ind] < 
                        φ_relax[ind] - current_node.φ_lower[ind])
                        branch_val = φ_relax[ind] - (current_node.φ_upper[ind] - φ_relax[ind])
                    else
                        branch_val = φ_relax[ind] + (φ_relax[ind] - current_node.φ_lower[ind])
                    end
                elseif branch_point == "midpoint"
                    diff = current_node.φ_upper[ind] - current_node.φ_lower[ind]
                    branch_val = current_node.φ_lower[ind] + diff / 2
                elseif branch_point == "current_point"
                    branch_val = φ_relax[ind]
                end
                # constructing child nodes
                φ_lower_left = current_node.φ_lower
                φ_upper_left = copy(current_node.φ_upper)
                φ_upper_left[ind] = branch_val
                φ_lower_right = copy(current_node.φ_lower)
                φ_lower_right[ind] = branch_val
                φ_upper_right = current_node.φ_upper
                left_child_node = BBNode(
                    φ_lower = φ_lower_left,
                    φ_upper = φ_upper_left,
                    node_id = counter + 1,
                    # initialize a node's LB with the objective of relaxation of parent
                    LB = objective_relax,
                    parent_id = current_node.node_id,
                )
                right_child_node = BBNode(
                    φ_lower = φ_lower_right,
                    φ_upper = φ_upper_right,
                    node_id = counter + 2,
                    # initialize a node's LB with the objective of relaxation of parent
                    LB = objective_relax,
                    parent_id = current_node.node_id,
                )
                push!(nodes, left_child_node, right_child_node)
                ancestry[current_node.node_id] = [counter + 1, counter + 2]
                counter += 2
            end
        end

        # cleanup actions - to be done regardless of whether split_flag was true or false
        if current_node.node_id != 1
            ancestry[current_node.parent_id] = setdiff(ancestry[current_node.parent_id], [current_node.node_id])
            if length(ancestry[current_node.parent_id]) == 0
                delete!(ancestry, current_node.parent_id)
                delete!(lower_bounds, current_node.parent_id)
            end
        end

        # update minimum of lower bounds
        if length(lower_bounds) == 0
            now_gap = add_update!(
                printlist, instance, nodes_explored, counter, 
                lower, upper, start_time,
            )
            last_updated_counter = counter
        else
            minimum_lower_bounds = minimum(values(lower_bounds))
            if minimum_lower_bounds > lower
                lower = minimum_lower_bounds
                now_gap = add_update!(
                    printlist, instance, nodes_explored, counter, 
                    lower, upper, start_time,
                )
                last_updated_counter = counter
            elseif (
                current_node.node_id == 1 
                || (counter ÷ update_step) > (last_updated_counter ÷ update_step)
                || now_gap ≤ gap 
                || counter ≥ max_steps
                || time() - start_time > time_limit
            )
                now_gap = add_update!(
                    printlist, instance, nodes_explored, counter, 
                    lower, upper, start_time,
                )
                last_updated_counter = counter

                item = [
                    current_node.node_id,
                    current_node.U_lower,
                    current_node.U_upper,
                ]
                if branching_region != "box"
                    push!(
                        item, 
                        current_node.φ_lower, 
                        current_node.φ_upper,
                    )
                end
                push!(ranges, item)
                if root_only
                    break
                end
            end
        end
    end

    end_time = time()
    time_taken = end_time - start_time

    solution["MSE_in"] = compute_MSE(solution["X"], A, indices, kind = "in")
    solution["MSE_out"] = compute_MSE(solution["X"], A, indices, kind = "out") 

    instance["run_details"] = OrderedDict(
        "log_time" => log_time,
        "start_time" => start_time,
        "end_time" => end_time,
        "time_taken" => time_taken,
        "solve_time_altmin" => solve_time_altmin,
        "solve_time_relaxation_feasibility" => solve_time_relaxation_feasibility,
        "solve_time_relaxation" => solve_time_relaxation,
        "solve_time_U_ranges" => solve_time_U_ranges,
        "solve_time_polyhedra" => solve_time_polyhedra,
        "nodes_explored" => nodes_explored,
        "nodes_total" => counter,
        "nodes_dominated" => nodes_dominated,
        "nodes_relax_infeasible" => nodes_relax_infeasible,
        "nodes_relax_feasible" => nodes_relax_feasible,
        "nodes_relax_feasible_pruned" => nodes_relax_feasible_pruned,
        "nodes_master_feasible" => nodes_master_feasible,
        "nodes_master_feasible_improvement" => nodes_master_feasible_improvement,
        "nodes_relax_feasible_split" => nodes_relax_feasible_split,
    )
    push!(
        printlist, 
        "\n\nRun details:\n"
    )
    for (k, v) in instance["run_details"]
        if startswith(k, "nodes")
            note = Printf.@sprintf(
                "%33s: %10d\n",
                k, v,
            )
        elseif startswith(k, "time") || startswith(k, "solve_time")
            note = Printf.@sprintf(
                "%33s: %10.3f\n",
                k, v,
            )
        else
            note = Printf.@sprintf(
                "%33s: %s\n",
                k, v,
            )
        end
        push!(printlist, note)
    end

    for item in ranges
        push!(
            printlist,
            Printf.@sprintf("\n\nnode_id: %10d\n", item[1]),
            "\nU_lower:\n",
            sprint(show, "text/plain", item[2]),
            "\nU_upper:\n",
            sprint(show, "text/plain", item[3]),
        )
        if branching_region != "box"
            push!(
                printlist,
                "\nφ_lower:\n",
                sprint(show, "text/plain", item[4]),
                "\nφ_upper:\n",
                sprint(show, "text/plain", item[5]),
            )
        end
    end
    push!(
        printlist,
        "\n--------------------------------\n",
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
    Y::Matrix{Float64}, 
    U::Matrix{Float64}, 
    X::Matrix{Float64}, 
    Θ::Matrix{Float64}, 
    use_matrix_cuts::Bool,
    ;
    orthogonality_tolerance::Float64 = 0.0,
    projection_tolerance::Float64 = 1e-6, # needs to be greater than 0 because of Arpack library
    lifted_variable_tolerance::Float64 = 1e-6, # needs to be greater than 0 because of Arpack library
)
    if use_matrix_cuts
        return (
            eigs(Symmetric(U * U' - Y), nev=1, which=:SR, tol=1e-6)[1][1]
            ≥ 
            - projection_tolerance
        )
    else 
        return (
            all( (abs.(U' * U - I)) .≤ orthogonality_tolerance )
            && sum(Y[i,i] for i in 1:size(Y,1)) ≤ size(U, 2)
            && (
                eigs(Symmetric(Y - U * U'), nev=1, which=:SR, tol=1e-6)[1][1]
                ≥ - projection_tolerance
            )
            && (
                eigs(Symmetric([Y X; X' Θ]), nev=1, which=:SR, tol=1e-6)[1][1]
                ≥ - lifted_variable_tolerance
            )
        )
    end
end

function relax_feasibility_frob_matrixcomp( # this is the version without matrix_cuts
    n::Int,
    k::Int,
    branching_region::String,
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
    if !(branching_region in ["box", "angular", "polyhedral", "hybrid"])
        error("""
        Invalid input for branching region.
        Branching region must be either "box" or "angular" or "polyhedral" or "hybrid"; $branching_region supplied instead.
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

    start_time = time()

    (n, k) = size(U_lower)

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)

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

    # 2-norm of columns of U are ≤ 1
    @constraint(
        model,
        [j = 1:k],
        [
            1;
            U[:,j]
        ] in SecondOrderCone()
    )

    @objective(
        model,
        Min,
        0
    )

    optimize!(model)

    end_time = time()
    
    return Dict(
        "feasible" => (JuMP.termination_status(model) == MOI.OPTIMAL),
        "time_taken" => end_time - start_time,
    )
end

function relax_frob_matrixcomp(
    n::Int,
    k::Int,
    branching_region::String,
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
    matrix_cuts::Union{Vector, Nothing} = nothing,
    orthogonality_tolerance::Float64 = 0.0,
    solver_output::Int = 0,
)
    function compute_α(Y, γ, A, indices)
        (n, m) = size(A)
        α = zeros(size(A))
        for j in 1:m
            for i in 1:n
                if indices[i,j] ≥ 0.5
                    α[i,j] = (
                        - γ * sum(
                            Y[i,l] * A[l,j] * indices[l,j]
                            for l in 1:n
                        )
                    ) / (
                        1 + γ * sum(
                            Y[i,l] * indices[l,j]
                            for l in 1:n
                        )
                    ) + A[i,j]
                end
            end    
        end
        return α
    end

    if !(branching_region in ["box", "angular", "polyhedral", "hybrid"])
        error("""
        Invalid input for branching region.
        Branching region must be either "box" or "angular" or "polyhedral" or "hybrid"; $branching_region supplied instead.
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

    model = Model(Mosek.Optimizer)
    if solver_output == 0
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
    end

    use_matrix_cuts = !isnothing(matrix_cuts)

    @variable(model, X[1:n, 1:m])
    @variable(model, Y[1:n, 1:n], Symmetric)
    @variable(model, Θ[1:m, 1:m], Symmetric)
    @variable(model, U[1:n, 1:k])
    if !use_matrix_cuts
        @variable(model, t[1:n, 1:k, 1:k])
    end

    @constraint(model, LinearAlgebra.Symmetric([Y X; X' Θ]) in PSDCone())
    @constraint(model, LinearAlgebra.Symmetric([Y U; U' I]) in PSDCone())
    @constraint(model, LinearAlgebra.Symmetric(I - Y) in PSDCone())

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

    # matrix cuts on U, if supplied
    if use_matrix_cuts 
        if length(matrix_cuts) > 0
            for (x, Û, directions) in matrix_cuts
                for j in 1:k
                    if directions[j] == "left"
                        @constraint(
                            model, 
                            -1 ≤ Compat.dot(x, U[:,j]), 
                        )
                        @constraint(
                            model, 
                            Compat.dot(x, U[:,j])
                            ≤ Compat.dot(x, Û[:,j]), 
                        )
                        @constraint(
                            model,
                            Û[:,j]' * x * x' * U[:,j] 
                            .+ Compat.dot(x, Û[:,j] - U[:,j])
                            .≥ Compat.dot((x * x'), Y)
                        )
                    elseif directions[j] == "right"
                        @constraint(
                            model,
                            Compat.dot(x, Û[:,j]) 
                            ≤ Compat.dot(x, U[:,j]),
                        )
                        @constraint(
                            model,
                            Compat.dot(x, U[:,j]) ≤ 1,
                        )
                        @constraint(
                            model,
                            Û[:,j]' * x * x' * U[:,j] 
                            .+ Compat.dot(x, U[:,j] - Û[:,j])
                            .≥ Compat.dot((x * x'), Y)
                        )
                    end
                end
            end
        end
    else
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
    end
    
    # 2-norm of columns of U are ≤ 1
    @constraint(
        model,
        [j = 1:k],
        [
            1;
            U[:,j]
        ] in SecondOrderCone()
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
    results = Dict(
        "model" => model,
        "solve_time" => solve_time(model),
        "termination_status" => JuMP.termination_status(model),
    )

    if JuMP.termination_status(model) in [
        MOI.OPTIMAL,
        MOI.LOCALLY_SOLVED,
    ]
        results["feasible"] = true
        results["objective"] = objective_value(model)
        results["α"] = compute_α(value.(Y), γ, A, indices)
        results["Y"] = value.(Y)
        results["U"] = value.(U)
        results["X"] = value.(X)
        results["Θ"] = value.(Θ)
    elseif JuMP.termination_status(model) in [
        MOI.INFEASIBLE,
        MOI.DUAL_INFEASIBLE,
        MOI.LOCALLY_INFEASIBLE,
        MOI.INFEASIBLE_OR_UNBOUNDED,
    ]
        results["feasible"] = false
    elseif JuMP.termination_status(model) == MOI.SLOW_PROGRESS
        if has_values(model)
            results["feasible"] = true
            results["objective"] = objective_value(model)
            results["α"] = compute_α(value.(Y), γ, A, indices)
            results["Y"] = value.(Y)
            results["U"] = value.(U)
            results["X"] = value.(X)
            results["Θ"] = value.(Θ)
        else
            results["feasible"] = false
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
    n::Int,
    k::Int,
    indices::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    ;
    U_initial::Matrix{Float64},
    U_lower::Array{Float64,2} = begin
        U_lower = -ones(n,k)
        U_lower[end,:] .= 0.0
        U_lower
    end,
    U_upper::Array{Float64,2} = ones(n,k),
    ϵ::Float64 = 1e-10,
    max_iters::Int = 10000,
)
    # TODO: make the models in the main function body

    altmin_start_time = time()

    (n, m) = size(A)
        
    U_current = U_initial

    counter = 0
    objective_current = 1e10

    model_U = Model(Gurobi.Optimizer)
    set_silent(model_U)
    @variable(model_U, U[1:n, 1:k])
    @constraint(model_U, U .≤ U_upper)
    @constraint(model_U, U .≥ U_lower)
    
    model_V = Model(Gurobi.Optimizer)
    set_silent(model_V)
    @variable(model_V, V[1:k, 1:m])

    while counter < max_iters
        counter += 1
        # Optimize over V, given U 
        @objective(
            model_V,
            Min,
            (1 / 2) * sum(
                (
                    sum(U_current[i,ind] * V[ind,j] for ind in 1:k) 
                    - A[i,j]
                )^2 * indices[i,j]
                for i in 1:n, j in 1:m
            )
            + (1 / (2 * γ)) * sum(
                sum(U_current[i,ind] * V[ind,j] for ind in 1:k)^2
                for i in 1:n, j in 1:m
            )
        )
        optimize!(model_V)
        global V_new = value.(model_V[:V])

        # Optimize over U, given V 
        @objective(
            model_U,
            Min,
            (1 / 2) * sum(
                (
                    sum(U[i,ind] * V_new[ind,j] for ind in 1:k) 
                    - A[i,j]
                )^2 * indices[i,j]
                for i in 1:n, j in 1:m
            )
            + (1 / (2 * γ)) * sum(
                sum(U[i,ind] * V_new[ind,j] for ind in 1:k)^2
                for i in 1:n, j in 1:m
            )
        )
        optimize!(model_U)
        global U_new = value.(model_U[:U])

        objective_new = objective_value(model_U)
        
        objective_diff = abs(objective_new - objective_current)
        if objective_diff < ϵ # objectives don't oscillate!
            break
        end
        U_current = U_new
        V_current = V_new
        objective_current = objective_new
    end

    altmin_end_time = time()

    return Dict(
        "U" => U_new, 
        "V" => V_new, 
        "solve_time" => (altmin_end_time - altmin_start_time),
    )
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
) # TODO: let 2nd column depend on 1st column, 3rd column depend on 1st and 2nd, etc.

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
            φ_L = $φ_L, φ_U = $φ_U.
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
            φ_L = $φ_L, φ_U = $φ_U.
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

    start_time = time()

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

    end_time = time()

    return Dict(
        "U_upper" => U_upper,
        "U_lower" => U_lower,
        "time_taken" => end_time - start_time,
    )
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
        if termination_status(model) != OPTIMAL
            error("""
            Model not optimal!
            $(model)
            """)
        end
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
        p = @suppress angles_to_halfspace(ϕ1, ϕ2)
        return hrep([p])
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

    start_time = time()

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

    end_time = time()

    return Dict(
        "polyhedra" => polyhedra,
        "time_taken" => end_time - start_time,
    )
end

function U_col_to_φ_col(
    U_col::Vector{Float64},
)
    (n,) = size(U_col)
    tmp = copy(U_col)
    φ_col = ones(n-1)
    for i in 1:(n-1)
        φ_col[i] = φ_col[i] * acos(tmp[i])
        for j in (i+1):n
            tmp[j] = tmp[j] / sin(acos(tmp[i]))
        end
    end
    return φ_col
end

function compute_jacobian(
    φ::Vector{Float64},
)
    n = size(φ, 1) + 1
    jacobian = ones(n, n-1)
    # set entries above main diagonal to zero
    for j in 2:(n-1)
        for i in 1:(j-1)
            jacobian[i,j] = 0.0
        end
    end
    # set sines
    for j in 1:(n-1)
        jacobian[j,j] *= - sin(φ[j])
    end
    for φ_index in 1:(n-1)
        for j in setdiff(1:n-1, φ_index)
            for i in max(j, φ_index+1):n
                jacobian[i,j] *= sin(φ[φ_index])
            end
        end
    end
    # set cosines
    for φ_index in 1:(n-1)
        for j in 1:φ_index-1
            jacobian[φ_index, j] *= cos(φ[φ_index])
        end
        for i in φ_index+1:n
            jacobian[i, φ_index] *= cos(φ[φ_index])
        end
    end                
    return jacobian
end

function create_matrix_cut_child_nodes(
    Y::Matrix{Float64},
    U::Matrix{Float64},
    U_lower::Matrix{Float64},
    U_upper::Matrix{Float64},
    matrix_cuts::Vector{Tuple},
    counter::Int,
    node_id::Int,
    objective_relax::Float64,
)
    # returns a tuple:
    # x: (n,) most negative eigenvector of U U' - Y
    # U: (n, k) 
    if !(
        size(U) == size(U_lower) == size(U_upper)
        && size(Y, 1) == size(U, 1)
    )
        error("""
        Dimension mismatch.
        Input matrix Y must have size (n, n); $(size(Y)) instead.
        Input matrix U must have size (n, k); $(size(U)) instead.
        Input matrix U_lower must have size (n, k); $(size(U_lower)) instead.
        Input matrix U_upper must have size (n, k); $(size(U_upper)) instead.
        """)
    end
    (n, k) = size(U)
    
    _, x = eigs(U * U' - Y, nev=1, which=:SR, tol=1e-6)
    
    return (
        BBNode(
            U_lower = U_lower,
            U_upper = U_upper,
            matrix_cuts = vcat(matrix_cuts, [(x, U, directions)]),
            node_id = counter + ind,
            # initialize a node's LB with the objective of relaxation of parent
            LB = objective_relax,
            parent_id = node_id,
        )
        for (ind, directions) in enumerate(
            Iterators.product(repeat([["left", "right"]], k)...)
        )
    )
end