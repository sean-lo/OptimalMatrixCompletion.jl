module OptimalMatrixCompletion

using LinearAlgebra
using Arpack
using Random

using Printf
using Dates
using Suppressor
using DataFrames
using OrderedCollections
using DataStructures
using Parameters
using Combinatorics

using JuMP
using MathOptInterface
using Mosek
using MosekTools

export matrix_completion_branchandbound
export matrix_completion_SDP_relaxation
export compute_SDP_relaxation_objective
export evaluate_objective
export alternating_minimization
export BBNode
export BBTree

export generate_rank1_matrix_completion_Shor_constraints_indexes
export BBNodeDisjunctiveCuts
export BBNodeShorInfo

@with_kw mutable struct BBNodeDisjunctiveCuts
    cuts::Vector{Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}} = Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}[]
end

@with_kw mutable struct BBNodeShorInfo
    constraints_indexes::Vector{NTuple{4, Int}} = NTuple{4, Int}[]
    SOC_constraints_indexes::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[]
end

@with_kw mutable struct BBNode
    node_id::Int
    parent_id::Int
    U_lower::Matrix{Float64}
    U_upper::Matrix{Float64}
    LB::Float64
    depth::Int
    master_feasible::Bool = false
    disjunctive_cuts::Union{
        Nothing,
        BBNodeDisjunctiveCuts,
    } = nothing
    Shor_info::Union{
        Nothing, 
        BBNodeShorInfo,
    } = nothing
end

@with_kw mutable struct BBTree
    nodes::Dict{Int, BBNode}
    node_ids::Vector{Int}
    counter::Int
    last_updated_counter::Int
    nodes_explored::Int
    nodes_remaining::Int
    best_upper_bound::Float64
    best_lower_bound::Float64
    now_gap::Float64
    lower_bounds::PriorityQueue
end


function add_message!(
    printlist::Vector{String}, 
    message_list::Vector{String},
)
    for message in message_list
        print(stdout, message)
        flush(stdout)
        push!(printlist, message)
    end
    return
end

@doc raw"""
    matrix_completion_branchandbound(
        k::Int,
        A::Array{Float64, 2},
        indices::BitMatrix,
        γ::Float64,
        ;
        <keyword arguments>
    )

Complete matrix `A` with observed indices in `indices` with rank-`k` matrix `X`.

We solve (with regularization parameter `γ`):
```math
\min_{\mathbf{X}}
\frac{1}{2} \sum_{(i,j) \in \mathcal{I}} (X_{i,j} - A_{i,j})^2 
+ \frac{1}{2 \gamma} \|\mathbf{X}\|_F^2
\quad 
\text{s.t. } \text{Rank}(\mathbf{X}) \leq k
```

# Arguments
- `node_selection::String = "breadthfirst"`: the node selection strategy to use: either "breadthfirst" or "bestfirst" or "depthfirst" or "bestfirst_depthfirst";
- `bestfirst_depthfirst_cutoff::Int = 10000`: in the situation with `node_selection = "bestfirst_depthfirst"`, the number of nodes in the queue before the algorithm switches from `"bestfirst"` to `"depthfirst"`;
- `gap::Float64 = 1e-4`: relative optimality gap for branch-and-bound algorithm;
- `use_disjunctive_cuts::Bool = true`: whether to use eigenvector disjunctions, highly recommended to be `true`;
- `disjunctive_cuts_type::Union{String, Nothing} = nothing`: number of pieces in piecewise linear upper-approximation; either "linear" (2 pieces) or "linear2" (3 pieces) or "linear3" (4 pieces);
- `disjunctive_cuts_breakpoints::Union{String, Nothing} = nothing`: number of eigenvectors to use in constructing separation oracle, either "smallest_1_eigvec" (most negative eigenvector) or "smallest_2_eigvec" (combination of first and second most negative eigenvectors);
- `add_Shor_valid_inequalities::Bool = false`: whether to add Shor SDP inequalities to strengthen SDP relaxation at each node;
- `Shor_valid_inequalities_noisy_rank1_num_entries_present::Vector{Int} = [1, 2, 3, 4]`: if `add_Shor_valid_inequalities` is true, the set of 2-by-2 determinant minors to model with Shor SDP inequalities, based on the number of filled entries (should be some subset of `[1, 2, 3, 4]`);
- `add_Shor_valid_inequalities_fraction::Float64 = 1.0`: if `add_Shor_valid_inequalities` is true, the proportion of 2-by-2 determinant minors to model with Shor SDP inequalities;
- `add_Shor_valid_inequalities_iterative::Bool = false`: if `add_Shor_valid_inequalities` is true, whether to add them iteratively from parent node to child node;
- `max_update_Shor_indices_probability::Float64 = 1.0`: if `add_Shor_valid_inequalities_iterative` is true, the maximum probability of adding inequalities at a node;
- `min_update_Shor_indices_probability::Float64 = 0.1`, if `add_Shor_valid_inequalities_iterative` is true, the minimum probability of adding inequalities at a node;
- `update_Shor_indices_probability_decay_rate::Float64 = 1.1`: if `add_Shor_valid_inequalities_iterative` is true, the base of the exponential decay of the probability of adding inequalities at a node, as a function of depth in the tree;
- `update_Shor_indices_n_minors::Int = 100`: if `add_Shor_valid_inequalities_iterative` is true, the number of Shor SDP inequalities to add at a node whenever adding is performed;
- `root_only::Bool = false`: if true, only solves relaxation at root node
- `altmin_flag::Bool = true`: whether to perform alternating minimization at nodes in the branch-and-bound tree, highly recommended to be `true`;
- `max_altmin_probability::Float64 = 1.0`: if `altmin_flag` is true, the maximum probability of performing alternating minimization at a node;
- `min_altmin_probability::Float64 = 0.005`: if `altmin_flag` is true, the minimum probability of performing alternating minimization at a node;
- `altmin_probability_decay_rate::Float64 = 1.1`: if `altmin_flag` is true, the base of the exponential decay of the probability of performing alternating minimization at a node, as a function of depth in the tree;
- `altmin_root_n_iters::Int = 1`: if `altmin_flag` is true, how many times to run alternating minimization at the root node to get a good solution;
- `use_max_steps::Bool = false`: whether to terminate the algorithm based on the number of branch-and-bound nodes explored;
- `max_steps::Int = 1000000`: if `use_max_steps` is true, the upper limit on number of branch-and-bound nodes explored;
- `time_limit::Int = 3600`: time limit in seconds
- `update_step::Int = 1000`: number of branch-and-bound nodes explored per printed update
- `verbosity::Int = 0`: verbosity level
    - 0 is silent
    - 1 includes one line per update
    - 2 includes all alternating minimization runs
    - 3 includes one line per SDP relaxation solve
    - 4 includes all SDP solver output

"""
function matrix_completion_branchandbound(
    k::Int,
    A::Array{Float64,2},
    indices::BitMatrix,
    γ::Float64,
    ;
    node_selection::String = "breadthfirst", # determining which node selection strategy to use: either "breadthfirst" or "bestfirst" or "depthfirst" or "bestfirst_depthfirst"
    bestfirst_depthfirst_cutoff::Int = 10000,
    gap::Float64 = 1e-4, # optimality gap for algorithm (proportion)
    use_disjunctive_cuts::Bool = true,
    disjunctive_cuts_type::Union{String, Nothing} = nothing,
    disjunctive_cuts_breakpoints::Union{String, Nothing} = nothing, # either "smallest_1_eigvec" or "smallest_2_eigvec"
    add_Shor_valid_inequalities::Bool = false,
    Shor_valid_inequalities_noisy_rank1_num_entries_present::Vector{Int} = [1, 2, 3, 4],
    add_Shor_valid_inequalities_fraction::Float64 = 1.0,
    add_Shor_valid_inequalities_iterative::Bool = false,
    max_update_Shor_indices_probability::Float64 = 1.0,
    min_update_Shor_indices_probability::Float64 = 0.1,
    update_Shor_indices_probability_decay_rate::Float64 = 1.1,
    update_Shor_indices_n_minors::Int = 100,
    root_only::Bool = false, # if true, only solves relaxation at root node
    altmin_flag::Bool = true,
    max_altmin_probability::Float64 = 1.0,
    min_altmin_probability::Float64 = 0.005,
    altmin_probability_decay_rate::Float64 = 1.1,
    altmin_root_n_iters::Int = 1,
    use_max_steps::Bool = false,
    max_steps::Int = 1000000,
    time_limit::Int = 3600, # time limit in seconds
    update_step::Int = 1000,
    verbosity::Int = 1,
)

    function compute_gap(lower::Float64, upper::Float64)
        if lower < 0
            return Inf
        else
            return (upper / lower) - 1
        end
    end

    function add_update!(
        printlist::Vector{String}, 
        instance::Dict{String, Any}, 
        tree::BBTree,
        current_time_elapsed::Float64,
        ;
        altmin_flag::Bool = false,
        print_message::Bool = true,
    )
        tree.now_gap = compute_gap(tree.best_lower_bound, tree.best_upper_bound)
        message = Printf.@sprintf(
            "| %10d | %10d | %10d | %10f | %10f | %10f | %10.3f  s  |",
            tree.nodes_explored, # Explored
            tree.counter, # Total
            tree.nodes_remaining, # Remaining
            tree.best_lower_bound, # Objective
            tree.best_upper_bound, # Incumbent
            tree.now_gap, # Gap
            current_time_elapsed, # Runtime
        )
        if altmin_flag
            message *= " - A\n"
        else
            message *= "\n"
        end
        print_message && add_message!(printlist, String[message])
        push!(
            instance["run_log"],
            (
                tree.nodes_explored, tree.counter, tree.nodes_remaining, 
                tree.best_lower_bound, tree.best_upper_bound, tree.now_gap, current_time_elapsed,
            )
        )
        tree.last_updated_counter = tree.counter
    end

    if use_disjunctive_cuts
        if !(disjunctive_cuts_type in ["linear", "linear2", "linear3"])
            error("""
            Invalid input for disjunctive cuts type.
            Disjunctive cuts type must be either "linear" or "linear2" or "linear3";
            $disjunctive_cuts_type supplied instead.
            """)
        end
        if !(disjunctive_cuts_breakpoints in ["smallest_1_eigvec", "smallest_2_eigvec"])
            error("""
            Invalid input for disjunctive cuts breakpoints.
            Disjunctive cuts type must be either "smallest_1_eigvec" or "smallest_2_eigvec";
            $disjunctive_cuts_breakpoints supplied instead.
            """)
        end
    end
    if !(node_selection in ["breadthfirst", "bestfirst", "depthfirst", "bestfirst_depthfirst"])
        error("""
        Invalid input for node selection.
        Node selection must be either "breadthfirst" or "bestfirst" or "depthfirst" or "bestfirst_depthfirst"; $node_selection supplied instead.
        """)
    end

    if !(size(A) == size(indices))
        error("""
        Dimension mismatch. 
        Input matrix A must have size (n, m);
        Input matrix indices must have size (n, m).
        """)
    end

    (n, m) = size(A)
    if !(n ≤ m)
        error("""
        Input matrix A must have size (n, m) with n <= m.
        Current size is $(size(A)).
        """)
    end

    if add_Shor_valid_inequalities
        if !(0.0 ≤ add_Shor_valid_inequalities_fraction ≤ 1.0)
            error(
                """
                Argument `add_Shor_valid_inequalities_fraction` = $add_Shor_valid_inequalities_fraction out of bounds [0.0, 1.0].
                """
            )
        end
    else
        add_Shor_valid_inequalities_fraction = nothing
    end

    if altmin_flag
        if !(
            0.0 ≤ max_altmin_probability ≤ 1.0
        )
            error("""
            Argument `max_altmin_probability` = $max_altmin_probability out of bounds [0.0, 1.0].
            """)
        end
        if !(
            0.0 < min_altmin_probability < 1.0
        )
            error("""
            Argument `min_altmin_probability` = $min_altmin_probability out of bounds (0.0, 1.0).
            """)
        end
        if !(
            1.0 < altmin_probability_decay_rate
        )
            error("""
            Argument `altmin_probability_decay_rate` = $altmin_probability_decay_rate out of bounds (1.0, ∞).
            """)
        end
    else
        max_altmin_probability = nothing
        min_altmin_probability = nothing
        altmin_probability_decay_rate = nothing
    end
    
    if use_disjunctive_cuts && add_Shor_valid_inequalities && add_Shor_valid_inequalities_iterative
        if !(
            0.0 ≤ max_update_Shor_indices_probability ≤ 1.0
        )
            error("""
            Argument `max_update_Shor_indices_probability` = $max_update_Shor_indices_probability out of bounds [0.0, 1.0].
            """)
        end
        if !(
            0.0 < min_update_Shor_indices_probability < 1.0
        )
            error("""
            Argument `min_update_Shor_indices_probability` = $min_update_Shor_indices_probability out of bounds (0.0, 1.0).
            """)
        end
        if !(
            1.0 < update_Shor_indices_probability_decay_rate
        )
            error("""
            Argument `update_Shor_indices_probability_decay_rate` = $update_Shor_indices_probability_decay_rate out of bounds (1.0, ∞).
            """)
        end
        if !(
            1.0 ≤ update_Shor_indices_n_minors
        )
            error("""
            Argument `update_Shor_indices_n_minors` = $update_Shor_indices_n_minors out of bounds [1.0, ∞).
            """)
        end
    else
        max_update_Shor_indices_probability = nothing
        min_update_Shor_indices_probability = nothing
        update_Shor_indices_probability_decay_rate = nothing
        update_Shor_indices_n_minors = nothing
    end

    log_time = Dates.now()
    Random.seed!(0)

    printlist = String[]

    (verbosity ≥ 1) && add_message!(printlist, String[
        Dates.format(log_time, "e, dd u yyyy HH:MM:SS"), 
        "\n",
        "Starting branch-and-bound on a matrix completion problem.\n",
        Printf.@sprintf("k:                                              %15d\n", k),
        Printf.@sprintf("m:                                              %15d\n", m),
        Printf.@sprintf("n:                                              %15d\n", n),
        Printf.@sprintf("num_indices:                                    %15d\n", sum(indices)),
        Printf.@sprintf("γ:                                              %15g\n", γ),
        "\n",
        Printf.@sprintf("Node selection:                                 %15s\n", node_selection),
        (node_selection == "bestfirst_depthfirst" ?
        Printf.@sprintf("Bestfirst-depthfirst cutoff:                    %15s\n", bestfirst_depthfirst_cutoff) : ""),
        Printf.@sprintf("Optimality gap:                                 %15g\n", gap),
        Printf.@sprintf("Only solve root node?:                          %15s\n", root_only),
        (!root_only ?
        Printf.@sprintf("Do altmin at child nodes?:                      %15s\n", altmin_flag) : ""),
        (!root_only && altmin_flag ? 
        Printf.@sprintf("Max altmin probability:                         %15s\n", max_altmin_probability) : ""),
        (!root_only && altmin_flag ? 
        Printf.@sprintf("Min altmin probability:                         %15s\n", min_altmin_probability) : ""),
        (!root_only && altmin_flag ? 
        Printf.@sprintf("Altmin probability decay rate:                  %15s\n", altmin_probability_decay_rate) : ""),
        (altmin_flag ? 
        Printf.@sprintf("Number of initial altmin seeds:                 %15s\n", altmin_root_n_iters) : ""),
        Printf.@sprintf("Cap on nodes?                                   %15s\n", use_max_steps),
        (use_max_steps ?
        Printf.@sprintf("Maximum nodes:                                  %15d\n", max_steps) : ""),
        Printf.@sprintf("Time limit (s):                                 %15d\n", time_limit),
        "\n",
        Printf.@sprintf("Use disjunctive cuts?:                          %15s\n", use_disjunctive_cuts),
    ])
    if use_disjunctive_cuts
        (verbosity ≥ 1) && add_message!(printlist, String[
            Printf.@sprintf("Disjunctive cuts type:                          %15s\n", disjunctive_cuts_type),
            Printf.@sprintf("Disjunction breakpoints:                        %15s\n", disjunctive_cuts_breakpoints),
            Printf.@sprintf("Use Shor LMI inequalities?:                     %15s\n", add_Shor_valid_inequalities),
            (add_Shor_valid_inequalities ?
            Printf.@sprintf("Proportion of Shor LMI inequalities?:           %15f\n", add_Shor_valid_inequalities_fraction) : ""),
            (add_Shor_valid_inequalities ? 
            Printf.@sprintf("(rank-1) Apply Shor LMI with            %15s entries.\n", Shor_valid_inequalities_noisy_rank1_num_entries_present) : ""),
            (add_Shor_valid_inequalities ? 
            Printf.@sprintf("Apply Shor LMI inequalities iteratively?        %15s\n", add_Shor_valid_inequalities_iterative) : ""),
            (add_Shor_valid_inequalities && add_Shor_valid_inequalities_iterative ? 
            Printf.@sprintf("(Iterative) Max update Shor indices prob.:      %15s\n", max_update_Shor_indices_probability) : ""),
            (add_Shor_valid_inequalities && add_Shor_valid_inequalities_iterative ? 
            Printf.@sprintf("(Iterative) Min update Shor indices prob.:      %15s\n", min_update_Shor_indices_probability) : ""),
            (add_Shor_valid_inequalities && add_Shor_valid_inequalities_iterative ? 
            Printf.@sprintf("(Iterative) Shor indices prob. decay rate:      %15s\n", update_Shor_indices_probability_decay_rate) : ""),
            (add_Shor_valid_inequalities && add_Shor_valid_inequalities_iterative ? 
            Printf.@sprintf("(Iterative) update Shor indices batch size:     %15s\n", update_Shor_indices_n_minors) : ""),        
        ])
    end

    start_time = time()
    solve_time_altmin = 0.0
    dict_solve_times_altmin = DataFrame(
        node_id = Int[],
        depth = Int[],
        solve_time = Float64[],
    )
    dict_num_iterations_altmin = DataFrame(
        node_id = Int[],
        depth = Int[],
        n_iters = Int[],
    )
    solve_time_relaxation_feasibility = 0.0
    solve_time_relaxation = 0.0
    dict_solve_times_relaxation = DataFrame(
        node_id = Int[],
        depth = Int[],
        solve_time = Float64[],
    )

    # (1) number of nodes explored so far
    nodes_explored = 0
    # (2) number of nodes generated in total
    counter = 1
    last_updated_counter = 1    

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
    # pruned nodes: (3) + (4) + (6) + (7)
    # not pruned nodes: (9)

    # (10) number of nodes on which alternating minimization heuristic is performed
    nodes_relax_feasible_split_altmin = 0 # not pruned
    # (10) ⊂ (9)
    # (11) number of nodes on which alternating minimization heuristic is performed and results in an improvement on the incumbent solution
    nodes_relax_feasible_split_altmin_improvement = 0
    # (11) ⊂ (10)

    instance = Dict{String, Any}()
    instance["run_log"] = DataFrame(
        explored = Int[],
        total = Int[],
        remaining = Int[],
        lower = Float64[],
        upper = Float64[],
        gap = Float64[],
        runtime = Float64[],
    )
    instance["run_details"] = OrderedDict(
        "k" => k,
        "m" => m,
        "n" => n,
        "A" => A,
        "indices" => indices,
        "num_indices" => convert(Int, round(sum(indices))),
        "γ" => γ,
        "node_selection" => node_selection,
        "bestfirst_depthfirst_cutoff" => bestfirst_depthfirst_cutoff,
        "optimality_gap" => gap,
        "root_only" => root_only,
        "altmin_flag" => altmin_flag,
        "max_altmin_probability" => max_altmin_probability,
        "min_altmin_probability" => min_altmin_probability,
        "altmin_probability_decay_rate" => altmin_probability_decay_rate,
        "altmin_root_n_iters" => altmin_root_n_iters,
        "use_max_steps" => use_max_steps,
        "max_steps" => max_steps,
        "time_limit" => time_limit,
        "use_disjunctive_cuts" => use_disjunctive_cuts,
        "disjunctive_cuts_type" => disjunctive_cuts_type,
        "disjunctive_cuts_breakpoints" => disjunctive_cuts_breakpoints,
        "add_Shor_valid_inequalities" => add_Shor_valid_inequalities,
        "add_Shor_valid_inequalities_fraction" => add_Shor_valid_inequalities_fraction,
        "add_Shor_valid_inequalities_iterative" => add_Shor_valid_inequalities_iterative,
        "max_update_Shor_indices_probability" => max_update_Shor_indices_probability,
        "min_update_Shor_indices_probability" => min_update_Shor_indices_probability,
        "update_Shor_indices_probability_decay_rate" => update_Shor_indices_probability_decay_rate,
        "update_Shor_indices_n_minors" => update_Shor_indices_n_minors,
        "Shor_valid_inequalities_noisy_rank1_num_entries_present" => Shor_valid_inequalities_noisy_rank1_num_entries_present,
        "log_time" => log_time,
        "start_time" => start_time,
        "end_time" => start_time,
        "time_taken" => 0.0,
        "solve_time_altmin" => solve_time_altmin,
        "dict_solve_times_altmin" => dict_solve_times_altmin,
        "dict_num_iterations_altmin" => dict_num_iterations_altmin,
        "solve_time_relaxation_feasibility" => solve_time_relaxation_feasibility,
        "solve_time_relaxation" => solve_time_relaxation,
        "dict_solve_times_relaxation" => dict_solve_times_relaxation,
        "root_node_timeout" => false,
        "nodes_explored" => nodes_explored,
        "nodes_total" => counter,
        "nodes_dominated" => nodes_dominated,
        "nodes_relax_infeasible" => nodes_relax_infeasible,
        "nodes_relax_feasible" => nodes_relax_feasible,
        "nodes_relax_feasible_pruned" => nodes_relax_feasible_pruned,
        "nodes_master_feasible" => nodes_master_feasible,
        "nodes_master_feasible_improvement" => nodes_master_feasible_improvement,
        "nodes_relax_feasible_split" => nodes_relax_feasible_split,
        "nodes_relax_feasible_split_altmin" => nodes_relax_feasible_split_altmin,
        "nodes_relax_feasible_split_altmin_improvement" => nodes_relax_feasible_split_altmin_improvement,
    )
    
    altmin_start_time = time()
    altmin_A_initial = zeros(n, m)
    altmin_A_initial[indices] = A[indices]
    altmin_U_initial_base = svd(altmin_A_initial).U[:,1:k]
    
    objective_initial = Inf
    best_ind = 0
    X_initial_array = Matrix{Float64}[] 
    sc = maximum(abs.(altmin_U_initial_base))
    (verbosity ≥ 1) && add_message!(printlist, [
        "\n",
        "------------------------------------------------------------------------------------------------\n",
    ])
    for altmin_iter in 1:altmin_root_n_iters
        if altmin_iter == 1
            altmin_U_initial = altmin_U_initial_base
        else
            altmin_U_initial = altmin_U_initial_base + sc * randn((n, k))
        end
        altmin_results = alternating_minimization(
            A, n, k, indices, γ, use_disjunctive_cuts,
            ;
            disjunctive_cuts_type = disjunctive_cuts_type,
            U_initial = altmin_U_initial,
            time_limit = time_limit,
        )
        alternating_minimization_printout(
            printlist, 
            altmin_results,
            0, 1.0, verbosity,
        )
        solve_time_altmin += altmin_results["solve_time"]
        push!(
            dict_solve_times_altmin,
            [
                0,
                0,
                altmin_results["solve_time"],
            ]
        )
        # do a re-SVD on U * V in order to recover orthonormal U
        X_initial_ = altmin_results["U"] * altmin_results["V"]
        push!(X_initial_array, X_initial_)
        U_initial_ = svd(X_initial_).U[:,1:k]
        objective_initial_ = evaluate_objective(
            X_initial_, A, indices, U_initial_, γ, 
        )
        if objective_initial_ < objective_initial
            objective_initial = objective_initial_
            best_ind = altmin_iter
        end
        (verbosity ≥ 1) && add_message!(printlist, [
            @sprintf(
                "Altmin run %02d: \t Objective %e in %3.3f s.\n", 
                altmin_iter, 
                objective_initial_, 
                (time() - altmin_start_time),
            )
        ])
    end

    # Select best alternating result in terms of upper bound
    X_initial = X_initial_array[best_ind]
    # do a re-SVD on U * V in order to recover orthonormal U
    U_initial = svd(X_initial).U[:,1:k]
    Y_initial = U_initial * U_initial'
    objective_initial = evaluate_objective(
        X_initial, A, indices, U_initial, γ, 
    )
    MSE_in_initial = compute_MSE(X_initial, A, indices, kind = "in")
    MSE_out_initial = compute_MSE(X_initial, A, indices, kind = "out")
    MSE_all_initial = compute_MSE(X_initial, A, indices, kind = "all")
    altmin_end_time = time()
    (verbosity ≥ 1) && add_message!(printlist, [
        "Alternating minimization completed: $(altmin_root_n_iters) runs.\n",
        @sprintf("Time:                %3.3f s.\n", altmin_end_time - altmin_start_time),
        @sprintf("Solution:            %10f\n", objective_initial),
        @sprintf("MSE (in/out):       (%6.4f, %6.4f)\n", MSE_in_initial, MSE_out_initial),
        "------------------------------------------------------------------------------------------------\n",
        "\n",
    ])

    objective_initial_time_found = time() - start_time
    solution = Dict(
        "objective_initial" => objective_initial,
        "objective_initial_time_found" => objective_initial_time_found,
        "MSE_in_initial" => MSE_in_initial,
        "MSE_out_initial" => MSE_out_initial,
        "MSE_all_initial" => MSE_all_initial,
        "Y_initial" => Y_initial,
        "U_initial" => U_initial,
        "X_initial" => X_initial,
        "objective" => objective_initial,
        "objective_time_found" => objective_initial_time_found,
        "MSE_in" => MSE_in_initial,
        "MSE_out" => MSE_out_initial,
        "MSE_all" => MSE_all_initial,
        "Y" => Y_initial,
        "U" => U_initial,
        "X" => X_initial,
    )

    if !use_disjunctive_cuts
        ranges = Tuple{Int, Matrix{Float64}, Matrix{Float64}}[]
    end
    nodes = Dict{Int, BBNode}()
    U_lower_initial = -ones(n, k)
    # Symmetry-breaking constraints
    for i in 1:k
        U_lower_initial[n-k+i:n,i] .= 0.0
    end
    U_upper_initial = ones(n, k)
    initial_node = BBNode(
        U_lower = U_lower_initial, 
        U_upper = U_upper_initial, 
        LB = -Inf,
        depth = 0,
        node_id = 1,
        parent_id = 0,
    )

    if use_disjunctive_cuts
        initial_node.disjunctive_cuts = BBNodeDisjunctiveCuts()
    end

    if add_Shor_valid_inequalities
        if !add_Shor_valid_inequalities_iterative
            initial_node_Shor_constraints_indexes = generate_rank1_matrix_completion_Shor_constraints_indexes(
                indices, 
                Shor_valid_inequalities_noisy_rank1_num_entries_present
            )
            initial_node_Shor_constraints_indexes = randsubseq(
                initial_node_Shor_constraints_indexes,
                add_Shor_valid_inequalities_fraction,
            )
            Shor_non_SOC_constraints_indexes = unique(vcat(
                [
                    [(i1,j1), (i1,j2), (i2,j1), (i2,j2)]
                    for (i1, i2, j1, j2) in initial_node_Shor_constraints_indexes
                ]...
            ))
            initial_node_Shor_SOC_constraints_indexes = setdiff(
                vec(collect(Iterators.product(1:n, 1:m))), 
                Shor_non_SOC_constraints_indexes
            )
            initial_node.Shor_info = BBNodeShorInfo(
                constraints_indexes = initial_node_Shor_constraints_indexes,
                SOC_constraints_indexes = initial_node_Shor_SOC_constraints_indexes,
            )
        else # add_Shor_valid_inequalities_iterative
            initial_node.Shor_info = BBNodeShorInfo(
                constraints_indexes = NTuple{4, Int}[],
                SOC_constraints_indexes = vec(collect(Iterators.product(1:n, 1:m))),
            )
        end
    end

    nodes[1] = initial_node

    (verbosity ≥ 1) && add_message!(printlist, String[
        "------------------------------------------------------------------------------------------------\n",
        "|   Explored |      Total |  Remaining |      Lower |      Upper |        Gap |    Runtime (s) |\n",
        "------------------------------------------------------------------------------------------------\n",
    ])

    # leaves' mapping from node_id to lower_bound
    tree = BBTree(
        nodes = nodes,
        node_ids = [1],
        counter = counter,
        nodes_explored = nodes_explored,
        last_updated_counter = last_updated_counter,
        nodes_remaining = 1,
        best_upper_bound = objective_initial,
        best_lower_bound = -Inf,
        now_gap = Inf,
        lower_bounds = PriorityQueue([1=>Inf]),
    )

    while (
        tree.now_gap > gap 
        && !(use_max_steps && (tree.counter ≥ max_steps))
        && time() - start_time ≤ time_limit
    )
        if length(tree.nodes) == 0
            break
        end

        if node_selection == "bestfirst_depthfirst"
            if length(tree.nodes) > bestfirst_depthfirst_cutoff
                node_selection_here = "depthfirst"
            else
                node_selection_here = "bestfirst"
            end
        else
            node_selection_here = node_selection
        end

        current_node = retrieve_node_from_tree!(tree, node_selection_here)

        split_flag = true

        # possible, since we may not explore tree breadth-first
        # (should not be possible for breadth-first search)
        if current_node.LB > tree.best_upper_bound
            split_flag = false
            nodes_dominated += 1
        end

        if !use_disjunctive_cuts && split_flag
            relax_feasibility_result = @suppress matrix_completion_check_SDP_relaxation_feasibility(
                n, k, A, indices;
                U_lower = current_node.U_lower, 
                U_upper = current_node.U_upper,
                time_limit = time_limit,
            )
            solve_time_relaxation_feasibility += relax_feasibility_result["time_taken"]
            if !relax_feasibility_result["feasible"]
                nodes_relax_infeasible += 1
                split_flag = false
            end
        end

        # solve SDP relaxation of master problem
        if split_flag
            if use_disjunctive_cuts
                relax_result = matrix_completion_SDP_relaxation(
                    current_node,
                    n, k, A, indices, γ, use_disjunctive_cuts;
                    disjunctive_cuts_type = disjunctive_cuts_type,
                    add_Shor_valid_inequalities = add_Shor_valid_inequalities,
                    time_limit = Int(round(time_limit - (time() - start_time))),
                    solver_output = (verbosity ≥ 4 ? 1 : 0),
                )
            else
                relax_result = matrix_completion_SDP_relaxation(
                    current_node,
                    n, k, A, indices, γ, false; 
                    U_lower = current_node.U_lower, 
                    U_upper = current_node.U_upper,
                    time_limit = Int(round(time_limit - (time() - start_time))),
                    solver_output = (verbosity ≥ 4 ? 1 : 0),
                )
            end
            solve_time_relaxation += relax_result["solve_time"]
            push!(
                dict_solve_times_relaxation,
                [
                    current_node.node_id,
                    current_node.depth,
                    relax_result["solve_time"],
                ]
            )
            if current_node.node_id == 1
                instance["run_details"]["root_node_timeout"] = (relax_result["termination_status"] == MOI.TIME_LIMIT)
            end
            if relax_result["feasible"] == false # should not happen, since this should be checked by matrix_completion_check_SDP_relaxation_feasibility
                nodes_relax_infeasible += 1
                split_flag = false
            elseif relax_result["termination_status"] in [
                MOI.OPTIMAL,
                MOI.LOCALLY_SOLVED,
                MOI.SLOW_PROGRESS,
                MOI.TIME_LIMIT,
            ]
                nodes_relax_feasible += 1
                objective_relax = relax_result["objective"]
                current_node.LB = objective_relax
                Y_relax = relax_result["Y"]
                U_relax = relax_result["U"]
                X_relax = relax_result["X"]
                Θ_relax = relax_result["Θ"]
                if current_node.node_id == 1
                    tree.best_lower_bound = objective_relax
                end
                # if solution for relax_result has higher objective than best found so far: prune the node
                if objective_relax > tree.best_upper_bound
                    nodes_relax_feasible_pruned += 1
                    split_flag = false            
                end
            end
        end

        # if solution for relax_result is feasible for original problem:
        # prune this node;
        # if it is the best found so far, update solution
        if (
            split_flag 
            && relax_result["termination_status"] in [
                MOI.OPTIMAL,
                MOI.LOCALLY_SOLVED, 
            ]
        )
            if matrix_completion_master_feasible(Y_relax, U_relax, X_relax, Θ_relax, use_disjunctive_cuts)
                current_node.master_feasible = true
                nodes_master_feasible += 1
                # if best found so far, update solution
                if objective_relax < tree.best_upper_bound
                    nodes_master_feasible_improvement += 1
                    update_solution!(
                        solution,
                        objective_relax,
                        (time() - start_time),
                        Y_relax,
                        U_relax,
                        X_relax,
                    )
                    tree.best_upper_bound = objective_relax
                    add_update!(
                        printlist, instance, tree, 
                        (time() - start_time),
                        ;
                        print_message = (verbosity ≥ 1),
                    )
                end
                split_flag = false
            end
        elseif (
            split_flag 
            && relax_result["termination_status"] in [
                MOI.TIME_LIMIT,
            ]
        )
            add_update!(
                printlist, instance, tree, 
                (time() - start_time),
                ;
                print_message = (verbosity ≥ 1),
            )
            verbosity ≥ 1 && add_message!(printlist, [
                "Time limit reached.\n"
            ])
        end

        # perform alternating minimization heuristic
        if altmin_flag
            altmin_probability = (
                current_node.depth > log(
                    altmin_probability_decay_rate, 
                    max_altmin_probability / min_altmin_probability
                )
                ?
                min_altmin_probability 
                :
                max_altmin_probability / (altmin_probability_decay_rate ^ current_node.depth)
            )
            altmin_flag_now = (rand() < altmin_probability)
        else
            altmin_flag_now = false
        end
        if split_flag
            if altmin_flag_now
                U_rounded = svd(relax_result["Y"]).U[:,1:k] # NOTE: need not be in disjunctive regions for $U$
                if use_disjunctive_cuts
                    altmin_results_BB = alternating_minimization(
                        A, n, k, indices, γ, use_disjunctive_cuts;
                        disjunctive_cuts_type = disjunctive_cuts_type,
                        U_initial = Matrix(U_rounded),
                        disjunctive_cuts = current_node.disjunctive_cuts.cuts,
                        time_limit = time_limit,
                    )
                else
                    altmin_results_BB = alternating_minimization(
                        A, n, k, indices, γ, use_disjunctive_cuts;
                        U_initial = Matrix(U_rounded),
                        U_lower = current_node.U_lower,
                        U_upper = current_node.U_upper,
                        time_limit = time_limit,
                    )
                end
                alternating_minimization_printout(
                    printlist, 
                    altmin_results_BB,
                    current_node.node_id, 
                    altmin_probability,
                    verbosity,
                )
                nodes_relax_feasible_split_altmin += 1
                solve_time_altmin += altmin_results_BB["solve_time"]
                push!(
                    dict_solve_times_altmin,
                    [
                        current_node.node_id,
                        current_node.depth,
                        altmin_results_BB["solve_time"],
                    ]
                )
                push!(
                    dict_num_iterations_altmin,
                    [
                        current_node.node_id,
                        current_node.depth,
                        altmin_results_BB["n_iters"],
                    ]
                )
                # it's possible for alternating minimization here to diverge, 
                # especially when `use_disjunctive_cuts = false`
                # in this case, don't compute local solution
                if altmin_results_BB["converged"]
                    X_local = altmin_results_BB["U"] * altmin_results_BB["V"]
                    U_local = svd(X_local).U[:,1:k] 
                    # no guarantees that this will be within U_lower and U_upper
                    Y_local = U_local * U_local'
                    # guaranteed to be a projection matrix since U_local is a svd result
                    objective_local = evaluate_objective(
                        X_local, A, indices, U_local, γ, 
                    )
                    if objective_local < tree.best_upper_bound
                        nodes_relax_feasible_split_altmin_improvement += 1
                        update_solution!(
                            solution, 
                            objective_local, 
                            (time() - start_time),
                            Y_local,
                            U_local,
                            X_local,
                        )
                        tree.best_upper_bound = objective_local
                        add_update!(
                            printlist, instance, tree, 
                            (time() - start_time),
                            ; 
                            altmin_flag = true,
                            print_message = (verbosity ≥ 1),
                        )
                    end
                end
            end
        end

        if split_flag
            # branch on variable
            # for now: branch on biggest element-wise difference between U_lower and U_upper
            nodes_relax_feasible_split += 1
            if use_disjunctive_cuts
                if add_Shor_valid_inequalities && add_Shor_valid_inequalities_iterative
                    update_Shor_indices_probability = (
                        current_node.depth > log(
                            update_Shor_indices_probability_decay_rate, 
                            max_update_Shor_indices_probability / min_update_Shor_indices_probability
                        )
                        ?
                        min_update_Shor_indices_probability 
                        :
                        max_update_Shor_indices_probability / (update_Shor_indices_probability_decay_rate ^ current_node.depth)
                    )
                    update_Shor_indices_flag_now = (rand() < update_Shor_indices_probability)
                else
                    update_Shor_indices_flag_now = false
                end
                matrix_cut_child_nodes = create_matrix_cut_child_nodes(
                    current_node,
                    disjunctive_cuts_type,
                    disjunctive_cuts_breakpoints,
                    ;
                    relax_result = relax_result,
                    indices = indices,
                    counter = tree.counter,
                    objective_relax = objective_relax,
                    update_Shor_indices_flag = update_Shor_indices_flag_now,
                    Shor_valid_inequalities_noisy_rank1_num_entries_present = Shor_valid_inequalities_noisy_rank1_num_entries_present,
                    n_minors = update_Shor_indices_n_minors,
                )
                add_nodes_to_tree!(
                    tree,
                    matrix_cut_child_nodes,
                    objective_relax,
                    current_node.node_id,
                )
            else
                ## McCormick branching

                # finding coordinates (i, j) to branch on
                (_, ind) = findmax(current_node.U_upper - current_node.U_lower)
                
                # finding branch_val
                diff = current_node.U_upper[ind] - current_node.U_lower[ind]
                branch_val = current_node.U_lower[ind] + diff / 2
                # constructing child nodes
                U_lower_left = current_node.U_lower
                U_upper_left = copy(current_node.U_upper)
                U_upper_left[ind] = branch_val
                U_lower_right = copy(current_node.U_lower)
                U_lower_right[ind] = branch_val
                U_upper_right = current_node.U_upper
                left_child_node = BBNode(
                    node_id = tree.counter + 1,
                    parent_id = current_node.node_id,
                    U_lower = U_lower_left,
                    U_upper = U_upper_left,
                    # initialize a node's LB with the objective of relaxation of parent
                    LB = objective_relax,
                    depth = current_node.depth + 1,
                )
                right_child_node = BBNode(
                    node_id = tree.counter + 2,
                    parent_id = current_node.node_id,
                    U_lower = U_lower_right,
                    U_upper = U_upper_right,
                    # initialize a node's LB with the objective of relaxation of parent
                    LB = objective_relax,
                    depth = current_node.depth + 1,
                )
                add_nodes_to_tree!(
                    tree,
                    [left_child_node, right_child_node],
                    objective_relax,
                    current_node.node_id,
                )
            end
        end

        # cleanup actions - to be done regardless of whether split_flag was true or false

        # Remove nodes that have bad bounds
        prune_dominated_nodes!(tree)

        # update minimum of lower bounds
        lower_bounds_updated = update_tree_lower_bounds!(tree)

        # print update
        if (
            lower_bounds_updated
            || current_node.node_id == 1 
            || (tree.counter ÷ update_step) > (tree.last_updated_counter ÷ update_step)
            || tree.now_gap ≤ gap 
            || (use_max_steps && tree.counter ≥ max_steps)
            || time() - start_time > time_limit
        )
            print_update_here = (verbosity ≥ 1)
        else
            print_update_here = (verbosity ≥ 3)
        end
        add_update!(
            printlist, instance, tree, 
            (time() - start_time),
            ;
            print_message = print_update_here,
        )

        if !use_disjunctive_cuts
            item = (
                current_node.node_id,
                current_node.U_lower,
                current_node.U_upper,
            )
            push!(ranges, item)
        end

        if root_only
            break
        end
    end

    end_time = time()
    time_taken = end_time - start_time

    solution["MSE_in"] = compute_MSE(solution["X"], A, indices, kind = "in")
    solution["MSE_out"] = compute_MSE(solution["X"], A, indices, kind = "out") 
    solution["MSE_all"] = compute_MSE(solution["X"], A, indices, kind = "all")

    instance["run_details"]["end_time"] = end_time
    instance["run_details"]["time_taken"] = time_taken
    instance["run_details"]["solve_time_altmin"] = solve_time_altmin
    instance["run_details"]["dict_solve_times_altmin"] = dict_solve_times_altmin
    instance["run_details"]["dict_num_iterations_altmin"] = dict_num_iterations_altmin
    instance["run_details"]["solve_time_relaxation_feasibility"] = solve_time_relaxation_feasibility
    instance["run_details"]["solve_time_relaxation"] = solve_time_relaxation
    instance["run_details"]["dict_solve_times_relaxation"] = dict_solve_times_relaxation

    instance["run_details"]["nodes_explored"] = tree.nodes_explored
    instance["run_details"]["nodes_total"] = tree.counter
    instance["run_details"]["nodes_dominated"] = nodes_dominated
    instance["run_details"]["nodes_relax_infeasible"] = nodes_relax_infeasible
    instance["run_details"]["nodes_relax_feasible"] = nodes_relax_feasible
    instance["run_details"]["nodes_relax_feasible_pruned"] = nodes_relax_feasible_pruned
    instance["run_details"]["nodes_master_feasible"] = nodes_master_feasible
    instance["run_details"]["nodes_master_feasible_improvement"] = nodes_master_feasible_improvement
    instance["run_details"]["nodes_relax_feasible_split"] = nodes_relax_feasible_split
    instance["run_details"]["nodes_relax_feasible_split_altmin"] = nodes_relax_feasible_split_altmin
    instance["run_details"]["nodes_relax_feasible_split_altmin_improvement"] = nodes_relax_feasible_split_altmin_improvement

    (verbosity ≥ 1) && add_message!(printlist, String["\n\nRun details:\n"])
    (verbosity ≥ 1) && add_message!(printlist, String[
        if startswith(k, "nodes")
            Printf.@sprintf("%46s: %10d\n", k, v)
        elseif startswith(k, "time") || startswith(k, "solve_time")
            Printf.@sprintf("%46s: %10.3f\n", k, v)
        elseif startswith(k, "dict")
            ""
        else
            Printf.@sprintf("%46s: %s\n", k, v)
        end
        for (k, v) in instance["run_details"]
            if !(k in ["A", "indices"])
    ])
    if !use_disjunctive_cuts
        for item in ranges
            (verbosity ≥ 1) && add_message!(printlist, String[
                Printf.@sprintf("\n\nnode_id: %10d\n", item[1]),
                "\nU_lower:\n",
                sprint(show, "text/plain", item[2]),
                "\nU_upper:\n",
                sprint(show, "text/plain", item[3]),
            ])               
        end
    end
    (verbosity ≥ 1) && add_message!(printlist, String[
        "\n--------------------------------\n",
        "\n\nInitial solution (warm start):\n",
        sprint(show, "text/plain", objective_initial),
        "\n\nMSE of sampled entries (warm start):\n",
        sprint(show, "text/plain", MSE_in_initial),
        "\n\nMSE of unsampled entries (warm start):\n",
        sprint(show, "text/plain", MSE_out_initial),
        "\n\nBest incumbent solution:\n",
        sprint(show, "text/plain", solution["objective"]),
        "\n\nMSE of sampled entries:\n",
        sprint(show, "text/plain", solution["MSE_in"]),
        "\n\nMSE of unsampled entries:\n",
        sprint(show, "text/plain", solution["MSE_out"]),
        "\n\n\n",
    ])

    return solution, printlist, instance
end


function update_solution!(
    solution::Dict{String, Any},
    objective_now::Float64,
    time_found::Float64,
    Y_now::Matrix{Float64},
    U_now::Matrix{Float64},
    X_now::Matrix{Float64},
)
    solution["objective"] = objective_now
    solution["objective_time_found"] = time_found
    solution["Y"] = copy(Y_now)
    solution["U"] = copy(U_now)
    solution["X"] = copy(X_now)
end

function retrieve_node_from_tree!(
    tree::BBTree,
    node_selection_here::String,
)
    if node_selection_here == "breadthfirst"
        id = popfirst!(tree.node_ids)
        delete!(tree.lower_bounds, id)
    elseif node_selection_here == "bestfirst"
        id = dequeue!(tree.lower_bounds)
        deleteat!(tree.node_ids, findfirst(isequal(id), tree.node_ids))
    elseif node_selection_here == "depthfirst" # NOTE: may not work well
        id = pop!(tree.node_ids)
        delete!(tree.lower_bounds, id)
    end
    current_node = pop!(tree.nodes, id)
    tree.nodes_explored += 1
    tree.nodes_remaining -= 1
    return current_node
end


function add_nodes_to_tree!(
    tree::BBTree,
    child_nodes::Vector{BBNode},
    parent_objective::Float64,
    parent_node_id::Int,
)
    merge!(
        tree.nodes, 
        Dict(
            (tree.counter + i) => node
            for (i, node) in enumerate(child_nodes)
        )
    )
    new_node_ids = collect(tree.counter+1:tree.counter+length(child_nodes))
    append!(tree.node_ids, new_node_ids)
    for id in new_node_ids
        enqueue!(tree.lower_bounds, id => parent_objective)
    end
    tree.counter += length(child_nodes)
    tree.nodes_remaining += length(child_nodes)
end

function update_tree_lower_bounds!(tree::BBTree)
    if length(tree.lower_bounds) == 0
        return true
    end
    _, minval = peek(tree.lower_bounds)
    updated = false
    if minval > tree.best_lower_bound
        tree.best_lower_bound = minval
        updated = true
    end
    return updated
end

function prune_dominated_nodes!(tree::BBTree)
    new_lower_bounds = PriorityQueue{Int, Float64}()
    removed_node_ids = Int[]
    while !isempty(tree.lower_bounds)
        (node_id, lower_bound) = dequeue_pair!(tree.lower_bounds)
        if lower_bound > tree.best_upper_bound
            push!(removed_node_ids, node_id)
            for (node_id_, _) in tree.lower_bounds
                push!(removed_node_ids, node_id_)
            end
            break
        end
        new_lower_bounds[node_id] = lower_bound
    end
    tree.lower_bounds = new_lower_bounds

    for node_id in removed_node_ids
        delete!(tree.nodes, node_id)
    end
    
    tree.node_ids = setdiff(tree.node_ids, removed_node_ids)
    tree.nodes_remaining = length(tree.nodes)
    
    return
end

"""
    matrix_completion_master_feasible(
        Y::Matrix{Float64}, 
        U::Matrix{Float64}, 
        X::Matrix{Float64}, 
        Θ::Matrix{Float64}, 
        use_disjunctive_cuts::Bool,
        ;
        orthogonality_tolerance::Float64 = 0.0,
        projection_tolerance::Float64 = 1e-6,
        lifted_variable_tolerance::Float64 = 1e-6,
    )

Determine if `(Y, U, X, Θ)` is feasible for the master problem.
"""
function matrix_completion_master_feasible(
    Y::Matrix{Float64}, 
    U::Matrix{Float64}, 
    X::Matrix{Float64}, 
    Θ::Matrix{Float64}, 
    use_disjunctive_cuts::Bool,
    ;
    orthogonality_tolerance::Float64 = 0.0,
    projection_tolerance::Float64 = 1e-6, # needs to be greater than 0 because of Arpack library
    lifted_variable_tolerance::Float64 = 1e-6, # needs to be greater than 0 because of Arpack library
)
    if use_disjunctive_cuts
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

function matrix_completion_check_SDP_relaxation_feasibility( # this is the version without disjunctive_cuts
    n::Int,
    k::Int,
    A::Array{Float64, 2},
    indices::BitMatrix,
    ;
    U_lower::Array{Float64,2} = begin
        U_lower = -ones(n,k)
        for i in 1:k
            U_lower[n-k+i:n,i] .= 0.0
        end
        U_lower
    end,
    U_upper::Array{Float64,2} = ones(n,k),
    orthogonality_tolerance::Float64 = 0.0,
    time_limit::Int = 3600,
)
    if !(
        size(U_lower) == (n,k)
        && size(U_upper) == (n,k)
    )
        error("""
        Dimension mismatch. 
        Input matrix U_lower must have size (n, k); 
        Input matrix U_upper must have size (n, k).
        """)
    end

    start_time = time()

    (n, k) = size(U_lower)

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
    set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    set_optimizer_attribute(model, "MSK_IPAR_MAX_NUM_WARNINGS", 0)
    set_time_limit_sec(model, time_limit)

    @variable(model, U[1:n, 1:k])
    @variable(model, t[1:n, 1:k, 1:k])

    # Lower bounds and upper bounds on U
    @constraint(model, [i=1:n, j=1:k], U_lower[i,j] ≤ U[i,j] ≤ U_upper[i,j])

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

function matrix_completion_SDP_relaxation(
    node::BBNode,
    n::Int,
    k::Int,
    A::Array{Float64,2},
    indices::BitMatrix, # for objective computation
    γ::Float64,
    use_disjunctive_cuts::Bool,
    ;
    disjunctive_cuts_type::Union{String, Nothing} = nothing,
    add_Shor_valid_inequalities::Bool = false,
    U_lower::Array{Float64,2} = begin
        U_lower = -ones(n,k)
        for i in 1:k
            U_lower[n-k+i:n,i] .= 0.0
        end
        U_lower
    end,
    U_upper::Array{Float64,2} = ones(n,k),
    orthogonality_tolerance::Float64 = 0.0,
    solver_output::Int = 0,
    time_limit::Int = 3600, # forces Mosek to not use a time limit
)

    if use_disjunctive_cuts
        if !(disjunctive_cuts_type in ["linear", "linear2", "linear3"])
            error("""
            Invalid input for disjunctive cuts type.
            Disjunctive cuts type must be either "linear" or "linear2" or "linear3";
            $disjunctive_cuts_type supplied instead.
            """)
        end
    end

    if !(
        size(U_lower) == (n,k)
        && size(U_upper) == (n,k)
        && size(A, 1) == size(indices, 1) == n
        && size(A, 2) == size(indices, 2)
    )
        error("""
        Dimension mismatch. 
        Input matrix U_lower must have size (n, k); 
        Input matrix U_upper must have size (n, k); 
        Input matrix A must have size (n, m);
        Input matrix indices must have size (n, m).""")
    end

    (n, k) = size(U_lower)
    (n, m) = size(A)

    model = Model(Mosek.Optimizer)
    if solver_output == 0
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
    end
    set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    set_optimizer_attribute(model, "MSK_IPAR_MAX_NUM_WARNINGS", 0)

    set_time_limit_sec(model, time_limit)

    if add_Shor_valid_inequalities && k > 1
        @variable(model, Xt[1:k, 1:n, 1:m])
        @expression(model, X[i=1:n, j=1:m], sum(Xt[:,i,j]))
    else
        @variable(model, X[1:n, 1:m])
    end
    @variable(model, Y[1:n, 1:n], Symmetric)
    @variable(model, Θ[1:m, 1:m], Symmetric)
    @variable(model, U[1:n, 1:k])
    if !use_disjunctive_cuts
        @variable(model, t[1:n, 1:k, 1:k])
    end
    if add_Shor_valid_inequalities
        @variable(model, W[1:n, 1:m] ≥ 0)
        Shor_constraints_coords = unique(vcat(
            [
                [(i1,j1), (i1,j2), (i2,j1), (i2,j2)]
                for (i1, i2, j1, j2) in node.Shor_info.constraints_indexes
            ]...
        ))
        Shor_constraints_coords_i = sort(unique([x[1] for x in Shor_constraints_coords]))
        Shor_constraints_coords_j = sort(unique([x[2] for x in Shor_constraints_coords]))
        if k == 1
            @variable(model, V1[
                Shor_constraints_coords_i, 
                [tuple(x...) for x in Combinatorics.combinations(Shor_constraints_coords_j, 2)]
            ])
            @variable(model, V2[
                [tuple(x...) for x in Combinatorics.combinations(Shor_constraints_coords_i, 2)], 
                Shor_constraints_coords_j
            ])
            @variable(model, V3[
                [tuple(x...) for x in Combinatorics.combinations(Shor_constraints_coords_i, 2)], 
                [tuple(x...) for x in Combinatorics.combinations(Shor_constraints_coords_j, 2)]
            ])
        else
            # Generate only some Wt, H, V
            @variable(model, Wt[
                1:k, 
                Shor_constraints_coords
            ] ≥ 0)
            @variable(model, H[
                [tuple(x...) for x in Combinatorics.combinations(1:k, 2)], 
                Shor_constraints_coords
            ])
            @variable(model, V1[
                1:k, 
                Shor_constraints_coords_i, 
                [tuple(x...) for x in Combinatorics.combinations(Shor_constraints_coords_j, 2)]
            ])
            @variable(model, V2[
                1:k, 
                [tuple(x...) for x in Combinatorics.combinations(Shor_constraints_coords_i, 2)], 
                Shor_constraints_coords_j
            ])
            @variable(model, V3[
                1:k, 
                [tuple(x...) for x in Combinatorics.combinations(Shor_constraints_coords_i, 2)], 
                [tuple(x...) for x in Combinatorics.combinations(Shor_constraints_coords_j, 2)]
            ])
        end
    end

    @constraint(model, cons_Y_X_Θ_PSD, LinearAlgebra.Symmetric([Y X; X' Θ]) in PSDCone())
    @constraint(model, cons_Y_U_I_PSD, LinearAlgebra.Symmetric([Y U; U' I]) in PSDCone())
    @constraint(model, cons_I_Y_PSD, LinearAlgebra.Symmetric(I - Y) in PSDCone())
    # Trace constraint on Y
    @constraint(model, cons_Y_trace, sum(Y[i,i] for i in 1:n) <= k)

    # Lower bounds and upper bounds on U
    @constraint(model, cons_U_box[i=1:n, j=1:k], U_lower[i,j] ≤ U[i,j] ≤ U_upper[i,j])

    # matrix cuts on U, if supplied
    if use_disjunctive_cuts && length(node.disjunctive_cuts.cuts) > 0
        linear_disjunctive_all_constraints = Dict{String, Any}[]
        L = length(node.disjunctive_cuts.cuts)
        for (l, (breakpoint_vec, Û, directions)) in enumerate(node.disjunctive_cuts.cuts)
            linear_disjunctive_constraints = Dict{String, Any}()
            linear_disjunctive_constraints["v_LB"] = ConstraintRef[]
            linear_disjunctive_constraints["v_UB"] = ConstraintRef[]

            # stores (each part of) the RHS of the linear inequality in U and Y
            expressions = zeros(AffExpr, k)

            # Constraints linking v (or w) to previous fitted Us and breakpoint vectors
            v = @expression(model, U' * breakpoint_vec) # stores each U[:,t]' * x[:,t]
            v̂ = Û' * breakpoint_vec        # stores fitted version: each Ű[:,t]' * x[:,t]   
            
            # Constraints linking v to breakpoints
            for j in 1:k
                if disjunctive_cuts_type == "linear"
                    if directions[j] == "left"
                        push!(
                            linear_disjunctive_constraints["v_LB"],
                            @constraint(model, -1 ≤ v[j])
                        )
                        push!(
                            linear_disjunctive_constraints["v_UB"],
                            @constraint(model, v[j] ≤ v̂[j])
                        )
                        expressions[j] = @expression(model, - v[j] + v̂[j] * v[j] + v̂[j])
                    elseif directions[j] == "right"
                        push!(
                            linear_disjunctive_constraints["v_LB"],
                            @constraint(model, v̂[j] ≤ v[j])
                        )
                        push!(
                            linear_disjunctive_constraints["v_UB"],
                            @constraint(model, v[j] ≤ 1)
                        )
                        expressions[j] = @expression(model, + v[j] + v̂[j] * v[j] - v̂[j])
                    end
                elseif disjunctive_cuts_type == "linear2"
                    if directions[j] == "left"
                        push!(
                            linear_disjunctive_constraints["v_LB"],
                            @constraint(model, -1 ≤ v[j])
                        )
                        push!(
                            linear_disjunctive_constraints["v_UB"],
                            @constraint(model, v[j] ≤ - abs(v̂[j]))
                        )
                        expressions[j] = @expression(model, - v[j] - abs(v̂[j]) * v[j] - abs(v̂[j]))
                    elseif directions[j] == "middle"
                        push!(
                            linear_disjunctive_constraints["v_LB"],
                            @constraint(model, - abs(v̂[j]) ≤ v[j])
                        )
                        push!(
                            linear_disjunctive_constraints["v_UB"],
                            @constraint(model, v[j] ≤ abs(v̂[j]))
                        )
                        expressions[j] = @expression(model, (v̂[j])^2)
                    elseif directions[j] == "right"
                        push!(
                            linear_disjunctive_constraints["v_LB"],
                            @constraint(model, abs(v̂[j]) ≤ v[j])
                        )
                        push!(
                            linear_disjunctive_constraints["v_UB"],
                            @constraint(model, v[j] ≤ 1)
                        )
                        expressions[j] = @expression(model, + v[j] + abs(v̂[j]) * v[j] - abs(v̂[j]))
                    end
                elseif disjunctive_cuts_type == "linear3"
                    if directions[j] == "left"
                        push!(
                            linear_disjunctive_constraints["v_LB"],
                            @constraint(model, -1 ≤ v[j])
                        )
                        push!(
                            linear_disjunctive_constraints["v_UB"],
                            @constraint(model, v[j] ≤ - abs(v̂[j]))
                        )
                        expressions[j] = @expression(model, - v[j] - abs(v̂[j]) * v[j] - abs(v̂[j]))
                    elseif directions[j] == "inner_left"
                        push!(
                            linear_disjunctive_constraints["v_LB"],
                            @constraint(model, - abs(v̂[j]) ≤ v[j])
                        )
                        push!(
                            linear_disjunctive_constraints["v_UB"],
                            @constraint(model, v[j] ≤ 0)
                        )
                        expressions[j] = @expression(model, - abs(v̂[j]) * v[j])
                    elseif directions[j] == "inner_right"
                        push!(
                            linear_disjunctive_constraints["v_LB"],
                            @constraint(model, 0 ≤ v[j])
                        )
                        push!(
                            linear_disjunctive_constraints["v_UB"],
                            @constraint(model, v[j] ≤ abs(v̂[j]))
                        )
                        expressions[j] = @expression(model, abs(v̂[j]) * v[j])
                    elseif directions[j] == "right"
                        push!(
                            linear_disjunctive_constraints["v_LB"],
                            @constraint(model, abs(v̂[j]) ≤ v[j])
                        )
                        push!(
                            linear_disjunctive_constraints["v_UB"],
                            @constraint(model, v[j] ≤ 1)
                        )
                        expressions[j] = @expression(model, abs(v̂[j]) * v[j])
                    end
                end
            end
            # aggregated constraint
            linear_disjunctive_constraints["cut"] = @constraint(
                model,
                sum(expressions) ≥ LinearAlgebra.dot(Y, breakpoint_vec * breakpoint_vec'),
            )
            push!(linear_disjunctive_all_constraints, linear_disjunctive_constraints)
        end
    elseif !use_disjunctive_cuts
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

    if add_Shor_valid_inequalities
        if k == 1
            for (i,j) in node.Shor_info.SOC_constraints_indexes
                @constraint(
                    model,
                    [0.5, W[i,j], X[i,j]] in RotatedSecondOrderCone() 
                )
            end
            @constraint(
                model,
                [j=1:m],
                Θ[j,j] == sum(W[i,j] for i in 1:n)
            )
            for (i1, i2, j1, j2) in node.Shor_info.constraints_indexes
                @constraint(
                    model, 
                    LinearAlgebra.Symmetric([
                        1           X[i1,j1]            X[i1,j2]            X[i2,j1]            X[i2,j2];
                        X[i1,j1]    W[i1,j1]            V1[i1,(j1,j2)]      V2[(i1,i2),j1]      V3[(i1,i2),(j1,j2)];
                        X[i1,j2]    V1[i1,(j1,j2)]      W[i1,j2]            V3[(i1,i2),(j1,j2)] V2[(i1,i2),j2];
                        X[i2,j1]    V2[(i1,i2),j1]      V3[(i1,i2),(j1,j2)] W[i2,j1]            V1[i2,(j1,j2)];
                        X[i2,j2]    V3[(i1,i2),(j1,j2)] V2[(i1,i2),j2]      V1[i2,(j1,j2)]      W[i2,j2];
                    ]) in PSDCone()
                )
            end
        else
            for (i,j) in node.Shor_info.SOC_constraints_indexes
                @constraint(
                    model,
                    [0.5, W[i,j], X[i,j]] in RotatedSecondOrderCone() 
                )
            end
            @constraint(
                model, 
                [(i,j) in Shor_constraints_coords], 
                W[i,j] == sum(Wt[:,(i,j)]) + 2 * sum(H[:,(i,j)])
            )
            @constraint(
                model,
                [j=1:m],
                Θ[j,j] == sum(W[i,j] for i in 1:n)
            )
            for (i1, i2, j1, j2) in node.Shor_info.constraints_indexes
                @constraint(
                    model, 
                    [t=1:k],
                    LinearAlgebra.Symmetric([
                        1            Xt[t,i1,j1]            Xt[t,i1,j2]            Xt[t,i2,j1]            Xt[t,i2,j2];
                        Xt[t,i1,j1]  Wt[t,(i1,j1)]          V1[t,i1,(j1,j2)]       V2[t,(i1,i2),j1]       V3[t,(i1,i2),(j1,j2)];
                        Xt[t,i1,j2]  V1[t,i1,(j1,j2)]       Wt[t,(i1,j2)]          V3[t,(i1,i2),(j1,j2)]  V2[t,(i1,i2),j2];
                        Xt[t,i2,j1]  V2[t,(i1,i2),j1]       V3[t,(i1,i2),(j1,j2)]  Wt[t,(i2,j1)]          V1[t,i2,(j1,j2)];
                        Xt[t,i2,j2]  V3[t,(i1,i2),(j1,j2)]  V2[t,(i1,i2),j2]       V1[t,i2,(j1,j2)]       Wt[t,(i2,j2)];
                    ]) in PSDCone()
                )
            end
            XWH_matrix = Array{AffExpr}(undef, (n, m, k+1, k+1))
            for (i,j) in Shor_constraints_coords
                XWH_matrix[i,j,1,1] = 1.0
                for t in 1:k
                    XWH_matrix[i,j,t+1,1] = Xt[t,i,j]
                    XWH_matrix[i,j,1,t+1] = Xt[t,i,j]
                    XWH_matrix[i,j,t+1,t+1] = Wt[t,(i,j)]
                end
                for (t1, t2) in Combinatorics.combinations(1:k, 2)
                    XWH_matrix[i,j,t1+1,t2+1] = H[(t1,t2),(i,j)]
                    XWH_matrix[i,j,t2+1,t1+1] = H[(t1,t2),(i,j)]
                end
                @constraint(
                    model,
                    LinearAlgebra.Symmetric(XWH_matrix[i,j,:,:]) in PSDCone()
                )
            end
        end
    end
    
    # 2-norm of columns of U are ≤ 1
    @constraint(
        model,
        cons_U_columns[j = 1:k],
        [1; U[:,j]] in SecondOrderCone()
    )

    if add_Shor_valid_inequalities
        @objective(
            model,
            Min,
            (1 / 2) * sum(
                (A[i,j]^2 - 2 * A[i,j] * X[i,j] + W[i,j]) * indices[i, j] 
                for i = 1:n, j = 1:m
            ) 
            + (1 / (2 * γ)) * sum(Θ[i, i] for i = 1:m) 
        )
    else
        @objective(
            model,
            Min,
            (1 / 2) * sum(
                (A[i,j] - X[i,j])^2 * indices[i, j] 
                for i = 1:n, j = 1:m
            ) 
            + (1 / (2 * γ)) * sum(Θ[i, i] for i = 1:m) 
        )
    end

    optimize!(model)
    results = Dict(
        "model" => model,
        "solve_time" => solve_time(model),
        "termination_status" => JuMP.termination_status(model),
    )

    if (
        JuMP.termination_status(model) in [
            MOI.OPTIMAL,
            MOI.LOCALLY_SOLVED,
        ] 
        || (
            JuMP.termination_status(model) in [
                MOI.SLOW_PROGRESS,
                MOI.TIME_LIMIT,
            ]
            && has_values(model)
        )
    )
        results["feasible"] = true
        # recompute objective value manually
        if add_Shor_valid_inequalities
            results["objective"] = compute_SDP_relaxation_objective(
                value.(X), value.(Y), value.(Θ), value.(U),
                A, indices, γ, 
                ;
                add_Shor_valid_inequalities = true,
                W = value.(W),
            )
        else
            results["objective"] = compute_SDP_relaxation_objective(
                value.(X), value.(Y), value.(Θ), value.(U),
                A, indices, γ, 
                ;
                add_Shor_valid_inequalities = false,
            )
        end
        results["Y"] = value.(Y)
        results["U"] = value.(U)
        results["X"] = value.(X)
        results["Θ"] = value.(Θ)
        if add_Shor_valid_inequalities && k > 1
            results["Xt"] = value.(Xt)
        end
        if !use_disjunctive_cuts
            results["t"] = value.(t)
        end
        if add_Shor_valid_inequalities
            results["W"] = value.(W)
            results["V1"] = value.(V1)
            results["V2"] = value.(V2)
            results["V3"] = value.(V3)
            if k == 1
                nothing
            else
                results["Xt"] = value.(Xt)
                results["Wt"] = value.(Wt)
                results["H"] = value.(H)
            end
        end
    elseif (
        JuMP.termination_status(model) in [
            MOI.INFEASIBLE,
            MOI.DUAL_INFEASIBLE,
            MOI.LOCALLY_INFEASIBLE,
            MOI.INFEASIBLE_OR_UNBOUNDED,
        ] 
        || (
            JuMP.termination_status(model) in [
                MOI.SLOW_PROGRESS,
                MOI.TIME_LIMIT,
            ]
            && !has_values(model)
        )
    )
        results["feasible"] = false
    else
        error("""
        unexpected termination status: $(JuMP.termination_status(model))
        """)
    end

    return results
end

function compute_SDP_relaxation_objective(
    X::Matrix{Float64},
    Y::Matrix{Float64},
    Θ::Matrix{Float64},
    U::Matrix{Float64},
    A::Matrix{Float64},
    indices::BitMatrix,
    γ::Float64,
    ;
    add_Shor_valid_inequalities::Bool = false,
    W::Matrix{Float64} = zeros(size(X)),
)
    (n, k) = size(U)
    (n, m) = size(A)

    if add_Shor_valid_inequalities
        return (
            (1 / 2) * sum(
                (A[i,j]^2 - 2 * A[i,j] * X[i,j] + W[i,j]) * indices[i, j] 
                for i = 1:n, j = 1:m
            ) 
            + (1 / (2 * γ)) * sum(Θ[i, i] for i = 1:m) 
        )
    else
        return (
            (1 / 2) * sum(
                (A[i,j] - X[i,j])^2 * indices[i, j] 
                for i = 1:n, j = 1:m
            ) 
            + (1 / (2 * γ)) * sum(Θ[i, i] for i = 1:m) 
        )
    end
end

function alternating_minimization(
    A::Array{Float64,2},
    n::Int,
    k::Int,
    indices::BitMatrix,
    γ::Float64,
    use_disjunctive_cuts::Bool,
    ;
    disjunctive_cuts_type::Union{String, Nothing} = nothing,
    U_initial::Matrix{Float64},
    U_lower::Array{Float64,2} = begin
        U_lower = -ones(n,k)
        for i in 1:k
            U_lower[n-k+i:n,i] .= 0.0
        end
        U_lower
    end,
    U_upper::Array{Float64,2} = ones(n,k),
    disjunctive_cuts::Vector{Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}} = Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}[],
    ϵ::Float64 = 1e-5,
    orthogonality_tolerance::Float64 = 1e-8,
    max_iters::Int = 100,
    time_limit::Int = 3600,
    solver_output::Int = 0,
)
    # Note: only used in the noisy case
    altmin_start_time = time()

    (n, m) = size(A)
        
    U_current = U_initial

    counter = 0
    objective_current = 1e10

    model_U = Model(Mosek.Optimizer)
    if solver_output == 0
        set_optimizer_attribute(model_U, "MSK_IPAR_LOG", 0)
    end
    set_optimizer_attribute(model_U, "MSK_IPAR_NUM_THREADS", 1)
    set_optimizer_attribute(model_U, "MSK_IPAR_MAX_NUM_WARNINGS", 0)
    
    @variable(model_U, U[1:n, 1:k])
    @variable(model_U, t[1:n, 1:n, 1:k])

    @constraint(model_U, U .≤ U_upper)
    @constraint(model_U, U .≥ U_lower)

    # impose linear constraints on U (but not jointly in U and Y)
    if use_disjunctive_cuts
        # Second-order-cone approximation of 2 x 2 minors of U'U ⪯ I (Atamturk, Gomez)
        @constraint(
            model_U,
            [j1 = 1:(k-1), j2 = (j1+1):k],
            [
                sqrt(2);
                U[:,j1] + U[:,j2]
            ] in SecondOrderCone()
        )
        @constraint(
            model_U,
            [j1 = 1:(k-1), j2 = (j1+1):k],
            [
                sqrt(2);
                U[:,j1] - U[:,j2]
            ] in SecondOrderCone()
        )

        # Linear constraints on U due to disjunctions
        if length(disjunctive_cuts) > 0
            for (l, (breakpoint_vec, Û, directions)) in enumerate(disjunctive_cuts)
                # Constraints linking v (or w) to previous fitted Us and breakpoint vectors
                v = @variable(model_U, [1:k]) # stores each U[:,t]' * x[:,t]
                @constraint(model_U, v .== U' * breakpoint_vec)
                v̂ = Û' * breakpoint_vec     # stores fitted version: each Ű[:,t]' * x[:,t]   

                # Constraints linking v to breakpoints
                for j in 1:k
                    if disjunctive_cuts_type == "linear"   
                        if directions[j] == "left"
                            @constraint(model_U, -1 ≤ v[j])
                            @constraint(model_U, v[j] ≤ v̂[j])
                        elseif directions[j] == "right"
                            @constraint(model_U, v̂[j] ≤ v[j])
                            @constraint(model_U, v[j] ≤ 1)
                        end
                    elseif disjunctive_cuts_type == "linear2"
                        if directions[j] == "left"
                            @constraint(model_U, -1 ≤ v[j])
                            @constraint(model_U, v[j] ≤ - abs(v̂[j]))
                        elseif directions[j] == "middle"
                            @constraint(model_U, - abs(v̂[j]) ≤ v[j])
                            @constraint(model_U, v[j] ≤ abs(v̂[j]))
                        elseif directions[j] == "right"
                            @constraint(model_U, abs(v̂[j]) ≤ v[j])
                            @constraint(model_U, v[j] ≤ 1)
                        end
                    elseif disjunctive_cuts_type == "linear3"
                        if directions[j] == "left"
                            @constraint(model_U, -1 ≤ v[j])
                            @constraint(model_U, v[j] ≤ - abs(v̂[j]))
                        elseif directions[j] == "inner_left"
                            @constraint(model_U, - abs(v̂[j]) ≤ v[j])
                            @constraint(model_U, v[j] ≤ 0)
                        elseif directions[j] == "inner_right"
                            @constraint(model_U, 0 ≤ v[j])
                            @constraint(model_U, v[j] ≤ abs(v̂[j]))
                        elseif directions[j] == "right"
                            @constraint(model_U, abs(v̂[j]) ≤ v[j])
                            @constraint(model_U, v[j] ≤ 1)
                        end
                    end
                end
            end
        end
    else
        # McCormick inequalities at U_lower and U_upper here
        @constraint(
            model_U,
            [i = 1:n, j1 = 1:k, j2 = j1:k],
            t[i, j1, j2] ≥ (
                U_lower[i, j2] * U[i, j1] 
                + U_lower[i, j1] * U[i, j2] 
                - U_lower[i, j1] * U_lower[i, j2]
            )
        )
        @constraint(
            model_U,
            [i = 1:n, j1 = 1:k, j2 = j1:k],
            t[i, j1, j2] ≥ (
                U_upper[i, j2] * U[i, j1] 
                + U_upper[i, j1] * U[i, j2] 
                - U_upper[i, j1] * U_upper[i, j2]
            )
        )
        @constraint(
            model_U,
            [i = 1:n, j1 = 1:k, j2 = j1:k],
            t[i, j1, j2] ≤ (
                U_upper[i, j2] * U[i, j1] 
                + U_lower[i, j1] * U[i, j2] 
                - U_lower[i, j1] * U_upper[i, j2]
            )
        )
        @constraint(
            model_U,
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
                    model_U,
                    sum(t[i, j1, j2] for i = 1:n) ≤ 1.0 + orthogonality_tolerance
                )
                @constraint(
                    model_U,
                    sum(t[i, j1, j2] for i = 1:n) ≥ 1.0 - orthogonality_tolerance
                )
            else
                @constraint(
                    model_U,
                    sum(t[i, j1, j2] for i = 1:n) ≤ 0.0 + orthogonality_tolerance
                )
                @constraint(
                    model_U,
                    sum(t[i, j1, j2] for i = 1:n) ≥ 0.0 - orthogonality_tolerance
                )
            end
        end

        @constraint(
            model_U,
            [i = 1:n, j1 = 2:k, j2 = 1:(j1-1)],
            t[i,j1,j2] == 0.0
        )
    end

    # 2-norm of columns of U are ≤ 1
    @constraint(
        model_U,
        [j = 1:k],
        [
            1;
            U[:,j]
        ] in SecondOrderCone()
    )
    
    model_V = Model(Mosek.Optimizer)
    if solver_output == 0
        set_optimizer_attribute(model_V, "MSK_IPAR_LOG", 0)
    end
    set_optimizer_attribute(model_V, "MSK_IPAR_NUM_THREADS", 1)
    set_optimizer_attribute(model_V, "MSK_IPAR_MAX_NUM_WARNINGS", 0)

    @variable(model_V, V[1:k, 1:m])

    objectives = Float64[]
    converged = false
    U_new = zeros(n, k)
    V_new = zeros(k, m)
    while (
        counter < max_iters
        && time() - altmin_start_time < time_limit
    )
        counter += 1
        # Optimize over V, given U 
        MOI.set(model_V, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
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
        V_new = value.(model_V[:V])

        # Optimize over U, given V
        MOI.set(model_U, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
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
        U_new = value.(model_U[:U])

        try
            objective_new = JuMP.objective_value(model_U)
            push!(objectives, objective_new)
            objective_diff = abs((objective_new - objective_current) / objective_current)
            if objective_diff < ϵ # objectives don't oscillate!
                converged = true
            elseif (
                length(objectives) > 5
                && all(
                    (objectives[end-i] > objectives[end-5])
                    for i in 0:4
                )
            )
                converged = true
            end

            if converged
                altmin_end_time = time()
                return Dict(
                    "converged" => converged,
                    "U" => U_new, 
                    "V" => V_new,
                    "solve_time" => (altmin_end_time - altmin_start_time),
                    "n_iters" => counter,
                    "max_iters" => max_iters,
                    "objectives" => objectives,
                )
            end

            U_current = U_new
            V_current = V_new
            objective_current = objective_new
        catch
            break
        end
    end

    altmin_end_time = time()

    return Dict(
        "converged" => converged,
        "U" => U_new, 
        "V" => V_new, 
        "solve_time" => (altmin_end_time - altmin_start_time),
        "n_iters" => counter,
        "max_iters" => max_iters,
        "objectives" => objectives,
    )
end

function alternating_minimization_printout(
    printlist::Vector{String},
    altmin_results::Dict{String, Any},
    node_id::Int,
    altmin_probability::Float64,
    verbosity::Int,
)
    if altmin_results["converged"]
        (verbosity ≥ 2) && add_message!(printlist, [
            @sprintf(
                "    Altmin at node %5d (w.p. %.3f) converged        in %3d / %3d iterations: %5.2f seconds.\n",
                node_id,
                altmin_probability,
                altmin_results["n_iters"],
                altmin_results["max_iters"],
                altmin_results["solve_time"],
            )
        ])
    else
        (verbosity ≥ 2) && add_message!(printlist, [
            @sprintf(
                "    Altmin at node %5d (w.p. %.3f) did not converge in %3d / %3d iterations: %5.2f seconds.\n",
                node_id,
                altmin_probability,
                altmin_results["n_iters"],
                altmin_results["max_iters"],
                altmin_results["solve_time"],
            )
        ])
    end
    if length(altmin_results["objectives"]) > 5
        obj_values_print = altmin_results["objectives"][end-5:end]
    else
        obj_values_print = altmin_results["objectives"]
    end
    (verbosity ≥ 2) && add_message!(printlist, [
        @sprintf(
            "    Objective values:      %s\n",
            join(
                [
                    @sprintf("%.4e", obj) for obj in obj_values_print
                ],
                ", "
            ),
        ),
        "\n",
    ])
end

function evaluate_objective(
    X::Array{Float64,2},
    A::Array{Float64,2},
    indices::BitMatrix,
    U::Array{Float64,2},
    γ::Float64,
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
    )
end

"""
    compute_MSE(
        X, 
        A, 
        indices; 
        kind = "out"
    )

Computes mean-squared error of entries in `X` and `A` based on `indices`.

If `kind == "out"`, computes out-of-sample MSE. If `kind == "in"`, computes in-sample MSE. If `kind == "all"`, computes overall MSE.
"""
function compute_MSE(
    X, 
    A, 
    indices
    ; 
    kind = "out"
)
    if kind == "out"
        if length(indices) == sum(indices)
            return 0.0
        else
            return (
                sum((X - A).^2 .* (1 .- indices)) 
                / (length(indices) - sum(indices))
            )
        end
    elseif kind == "in"
        if sum(indices) == 0.0
            return 0.0
        else
            return (
                sum((X - A).^2 .* indices) 
                / sum(indices)
            )
        end
    elseif kind == "all"
        return (
            sum((X - A).^2) 
            / length(indices)
        )
    else
        error("""
        Input argument `kind` not recognized!
        Must be one of "out", "in", or "all".
        """)
    end
end

function create_matrix_cut_child_nodes(
    node::BBNode,
    disjunctive_cuts_type::String,
    disjunctive_cuts_breakpoints::String,
    ;
    relax_result::Dict{String, Any},
    indices::BitMatrix,
    counter::Int,
    objective_relax::Float64,
    update_Shor_indices_flag::Bool = false,
    Shor_valid_inequalities_noisy_rank1_num_entries_present::Vector{Int} = [4],
    n_minors::Union{Int, Nothing} = 100,
)
    # disjunctive_cuts is a vector of tuples: each containing:
    # 1. breakpoint_vec: (n,) vector which is negative w.r.t. U U' - Y
    # 2. U: (n, k): previous fitted version of Û
    # 3. (if disjunctive_cuts_type == "linear") 
    # directions: (k,) vector of elements from {"left", "right"} 
    # OR 3. (if disjunctive_cuts_type == "linear2")
    # directions: (k,) vector of elements from {"left", "middle", "right"}
    # OR 3. (if disjunctive_cuts_type == "linear3")
    # directions: (k,) vector of elements from {"left", "inner_left", "inner_right", "right"}
    if !(disjunctive_cuts_type in ["linear", "linear2", "linear3"])
        error("""
        Invalid input for disjunctive cuts type.
        Disjunctive cuts type must be either "linear" or "linear2" or "linear3";
        $disjunctive_cuts_type supplied instead.
        """)
    end
    if !(disjunctive_cuts_breakpoints in ["smallest_1_eigvec", "smallest_2_eigvec"])
        error("""
        Invalid input for disjunctive cuts breakpoints.
        Disjunctive cuts type must be either "smallest_1_eigvec" or "smallest_2_eigvec";
        $disjunctive_cuts_breakpoints supplied instead.
        """)
    end

    Y::Matrix{Float64} = relax_result["Y"]
    U::Matrix{Float64} = relax_result["U"]
    X::Matrix{Float64} = relax_result["X"]
    if !(
        size(U) == size(node.U_lower) == size(node.U_upper)
        && size(Y, 1) == size(U, 1) == size(X, 1)
    )
        error("""
        Dimension mismatch.
        Input matrix Y must have size (n, n); $(size(Y)) instead.
        Input matrix U must have size (n, k); $(size(U)) instead.
        Input matrix U_lower must have size (n, k); $(size(node.U_lower)) instead.
        Input matrix U_upper must have size (n, k); $(size(node.U_upper)) instead.
        """)
    end
    (n, k) = size(U)
    (n, m) = size(X)
    
    if disjunctive_cuts_breakpoints == "smallest_1_eigvec"
        eigvals, eigvecs, _ = eigs(Symmetric(U * U' - Y), nev=1, which=:SR, tol=1e-6)
        breakpoint_vec = eigvecs[:,1]
    elseif disjunctive_cuts_breakpoints == "smallest_2_eigvec"
        eigvals, eigvecs, _ = eigs(Symmetric(U * U' - Y), nev=2, which=:SR, tol=1e-6)
        if eigvals[2] < -1e-10
            weights = abs.(eigvals[1:2]) ./ sqrt(sum(eigvals[1:2].^2))
            breakpoint_vec = weights[1] * eigvecs[:,1] + weights[2] * eigvecs[:,2]
        else
            breakpoint_vec = eigvecs[:,1]
        end
    end

    if disjunctive_cuts_type in ["linear", "linear2", "linear3"]
        if disjunctive_cuts_type == "linear"
            directions_list = enumerate(
                Iterators.product(repeat([["left", "right"]], k)...)
            )
        elseif disjunctive_cuts_type == "linear2"
            directions_list = enumerate(
                Iterators.product(repeat([["left", "middle", "right"]], k)...)
            )
        elseif disjunctive_cuts_type == "linear3"
            directions_list = enumerate(
                Iterators.product(repeat([["left", "inner_left", "inner_right", "right"]], k)...)
            )
        end
    end

    if update_Shor_indices_flag
        minors = generate_violated_Shor_minors(
            reshape(X, (1, n, m)), 
            indices,
            Shor_valid_inequalities_noisy_rank1_num_entries_present,
            node.Shor_info.constraints_indexes,
            n_minors,
        )
        new_Shor_constraints_indexes = [m[2] for m in minors]
        Shor_constraints_indexes = union(
            node.Shor_info.constraints_indexes,
            new_Shor_constraints_indexes
        )
        Shor_non_SOC_constraints_indexes = unique(vcat(
            [
                [(i1,j1), (i1,j2), (i2,j1), (i2,j2)]
                for (i1, i2, j1, j2) in Shor_constraints_indexes
            ]...
        ))
        Shor_SOC_constraints_indexes = setdiff(
            node.Shor_info.SOC_constraints_indexes, 
            Shor_non_SOC_constraints_indexes
        )
    end

    matrix_cut_child_nodes = BBNode[]
    for (ind, directions) in directions_list
        new_disjunctive_cuts = vcat(node.disjunctive_cuts.cuts, [(breakpoint_vec, U, collect(directions))])
        new_node = BBNode(
            node_id = counter + ind,
            parent_id = node.node_id,
            U_lower = node.U_lower,
            U_upper = node.U_upper,
            LB = objective_relax,
            depth = node.depth + 1,
            disjunctive_cuts = BBNodeDisjunctiveCuts(cuts=new_disjunctive_cuts),
        )
        if update_Shor_indices_flag
            new_node.Shor_info = BBNodeShorInfo(
                constraints_indexes = Shor_constraints_indexes,
                SOC_constraints_indexes = Shor_SOC_constraints_indexes,
            )
        else
            new_node.Shor_info = node.Shor_info
        end
        push!(matrix_cut_child_nodes, new_node)
    end
    return matrix_cut_child_nodes
end

function generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices::BitMatrix, # for Shor indexes, in the noisy case
    num_entries_present_list::Vector{Int},
)
    # Used in basis pursuit: Shor LMIs in rank-1
    (n, m) = size(indices)

    Shor_constraints_indexes = NTuple{4, Int}[]

    for num_entries_present in num_entries_present_list
        if num_entries_present == 4    
            for (i1, i2) in Combinatorics.combinations(1:n, 2)
                for (j1, j2) in Combinatorics.combinations(findall(indices[i1,:] .& indices[i2,:]), 2)
                    push!(Shor_constraints_indexes, (i1, i2, j1, j2))
                end
            end
        elseif num_entries_present == 3
            for (i1, i2) in Combinatorics.combinations(1:n, 2)
                for j1 in findall(indices[i1,:] .& indices[i2,:])
                    for j2 in findall(indices[i1,:] .⊻ indices[i2,:])
                        push!(Shor_constraints_indexes, (i1, i2, sort([j1, j2])...))
                    end
                end
            end
        elseif num_entries_present == 2
            # (a): [1 0; 1 0] and [0 1; 0 1]
            for (i1, i2) in Combinatorics.combinations(1:n, 2)
                for j1 in findall(indices[i1,:] .& indices[i2,:])
                    for j2 in findall(.~(indices[i1,:] .| indices[i2,:]))
                        push!(Shor_constraints_indexes, (i1, i2, sort([j1, j2])...))
                    end
                end
            end
            # (b): all other cases
            for (i1, i2) in Combinatorics.combinations(1:n, 2)
                for (j1, j2) in Combinatorics.combinations(findall(indices[i1,:] .⊻ indices[i2,:]), 2)
                    push!(Shor_constraints_indexes, (i1, i2, j1, j2))
                end
            end
        elseif num_entries_present == 1
            for (i1, i2) in Combinatorics.combinations(1:n, 2)
                for j1 in 1:m
                    if indices[i1,j1] + indices[i2,j1] == 1
                        # Looks for j2 in 1:n, 
                        # where indices[i1,j2] + indices[i2,j2] == 0
                        for j2 in findall(.~(indices[i1,:] .| indices[i2,:]))
                            push!(Shor_constraints_indexes, (i1, i2, sort([j1, j2])...))
                        end
                    end
                end
            end
        elseif num_entries_present == 0
            for (i1, i2) in Combinatorics.combinations(1:n, 2)
                for j1 in 1:(m-1)
                    if indices[i1,j1] == indices[i2,j1] == 0
                        # Looks for j2 in 1:n, 
                        # where indices[i1,j2] + indices[i2,j2] == 0
                        for j2 in findall(.~(indices[i1,(j1+1):end] .| indices[i2,(j1+1):end]))
                            push!(Shor_constraints_indexes, (i1, i2, j1, j1+j2))
                        end
                    end
                end
            end
        end
    end

    return Shor_constraints_indexes
end

function generate_violated_Shor_minors(
    X::Array{Float64, 3},
    indices::BitMatrix,
    Shor_valid_inequalities_noisy_rank1_num_entries_present::Vector{Int},
    Shor_constraints_indexes::Vector{NTuple{4, Int}},
    n_minors::Int,
)
    (k, n, m) = size(X)

    minors_indexes = generate_rank1_matrix_completion_Shor_constraints_indexes(
        indices, Shor_valid_inequalities_noisy_rank1_num_entries_present,
    )
    setdiff!(minors_indexes, Shor_constraints_indexes)
    minors = vec(collect(
        (
            sum(abs.(X[:,i1,j1] .* X[:,i2,j2] .- X[:,i1,j2] .* X[:,i2,j1])),
            (i1, i2, j1, j2)
        )
        for (i1, i2, j1, j2) in minors_indexes
    ))
    if length(minors) < n_minors
        return sort(minors, rev = true)
    else
        partialsort!(minors, 1:n_minors, rev=true)
        return minors[1:n_minors]
    end
end

end
