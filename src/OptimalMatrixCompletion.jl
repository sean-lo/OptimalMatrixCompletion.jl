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

export branchandbound_frob_matrixcomp
export rankk_presolve
export BBNode

@with_kw mutable struct BBNode
    node_id::Int
    parent_id::Int
    U_lower::Matrix{Float64}
    U_upper::Matrix{Float64}
    matrix_cuts::Vector{Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}} = Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}[]
    LB::Float64
    depth::Int
    linear_coupling_constraints_indexes::Vector{Tuple{Vector{Int}, Vector{Int}, String, Int}} = Tuple{Vector{Int}, Vector{Int}, String, Int}[]
    Shor_constraints_indexes::Vector{NTuple{4, Int}} = NTuple{4, Int}[]
    Shor_SOC_constraints_indexes::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[]
    master_feasible::Bool = false
end

@doc raw"""
    branchandbound_frob_matrixcomp(
        k::Int,
        A::Array{Float64, 2},
        indices::BitMatrix,
        γ::Float64,
        λ::Float64,
        noise::Bool,
        ;
        <keyword arguments>
    )

Complete matrix `A` with observed indices in `indices` with rank-`k` matrix `X`.

In the noisy case (`noise == true`), solves (with regularization parameter `γ`):
```math
\min_{\mathbf{X}}
\frac{1}{2} \sum_{(i,j) \in \mathcal{I}} (X_{i,j} - A_{i,j})^2 
+ \frac{1}{2 \gamma} \|\mathbf{X}\|_F^2
\quad 
\text{s.t. } \text{Rank}(\mathbf{X}) \leq k
```
In the noiseless case (`noise == false`), solves:

```math
\min_{\mathbf{X}}
\|\mathbf{X}\|_F^2
\quad 
\text{s.t. } X_{i,j} = A_{i,j} \ \forall (i,j) \in \mathcal{I}, 
\ \text{Rank}(\mathbf{X}) \leq k
```

# Arguments
- `branching_type::Union{String, Nothing} = nothing`: in the situation with `use_disjunctive_cuts = false`, determining which coordinate to branch on: either "lexicographic" or "bounds" or "gradient";
- `branch_point::Union{String, Nothing} = nothing`: in the situation with `use_disjunctive_cuts = false`, determine which value to branch on: either "midpoint" or "current_point"
- `node_selection::String = "breadthfirst"`: the node selection strategy to use: either "breadthfirst" or "bestfirst" or "depthfirst" or "bestfirst_depthfirst";
- `bestfirst_depthfirst_cutoff::Int = 10000`: in the situation with `node_selection = "bestfirst_depthfirst"`, the number of nodes in the queue before the algorithm switches from `"bestfirst"` to `"depthfirst"`;
- `gap::Float64 = 1e-4`: relative optimality gap for branch-and-bound algorithm;
- `use_disjunctive_cuts::Bool = true`: whether to use eigenvector disjunctions, highly recommended to be `true`;
- `disjunctive_cuts_type::Union{String, Nothing} = nothing`: number of pieces in piecewise linear upper-approximation; either "linear" (2 pieces) or "linear2" (3 pieces) or "linear3" (4 pieces);
- `disjunctive_cuts_breakpoints::Union{String, Nothing} = nothing`: number of eigenvectors to use in constructing separation oracle, either "smallest_1_eigvec" (most negative eigenvector) or "smallest_2_eigvec" (combination of first and second most negative eigenvectors);
- `presolve::Bool = false`: in the noiseless setting (`noise = false`), whether to perform presolve, highly recommended to be `true`;
- `add_basis_pursuit_valid_inequalities::Bool = false`: in the noiseless setting (`noise = false`), whether to impose valid inequalities from determinant minors;
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
- `use_max_steps::Bool = false`: whether to terminate the algorithm based on the number of branch-and-bound nodes explored;
- `max_steps::Int = 1000000`: if `use_max_steps` is true, the upper limit on number of branch-and-bound nodes explored;
- `time_limit::Int = 3600`: time limit in seconds
- `update_step::Int = 1000`: number of branch-and-bound nodes explored per printed update

"""
function branchandbound_frob_matrixcomp(
    k::Int,
    A::Array{Float64,2}, # This is a rank-k matrix (optionally, with noise)
    indices::BitMatrix,
    γ::Float64,
    λ::Float64,
    noise::Bool,
    ;
    branching_type::Union{String, Nothing} = nothing, # determining which coordinate to branch on: either "lexicographic" or "bounds" or "gradient"
    branch_point::Union{String, Nothing} = nothing, # determine which value to branch on: either "midpoint" or "current_point"
    node_selection::String = "breadthfirst", # determining which node selection strategy to use: either "breadthfirst" or "bestfirst" or "depthfirst" or "bestfirst_depthfirst"
    bestfirst_depthfirst_cutoff::Int = 10000,
    gap::Float64 = 1e-4, # optimality gap for algorithm (proportion)
    use_disjunctive_cuts::Bool = true,
    disjunctive_cuts_type::Union{String, Nothing} = nothing,
    disjunctive_cuts_breakpoints::Union{String, Nothing} = nothing, # either "smallest_1_eigvec" or "smallest_2_eigvec"
    presolve::Bool = false,
    add_basis_pursuit_valid_inequalities::Bool = false,
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
    use_max_steps::Bool = false,
    max_steps::Int = 1000000,
    time_limit::Int = 3600, # time limit in seconds
    update_step::Int = 1000,
)
    function add_message!(
        printlist::Vector{String}, 
        message_list::Vector{String},
    )
        for message in message_list
            print(stdout, message)
            push!(printlist, message)
        end
        return
    end

    function add_update!(
        printlist::Vector{String}, 
        instance::Dict{String, Any}, 
        nodes_explored::Int,
        counter::Int, 
        lower::Float64, 
        upper::Float64, 
        start_time::Float64,
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
        add_message!(printlist, String[message])
        push!(
            instance["run_log"],
            (nodes_explored, counter, lower, upper, now_gap, current_time_elapsed)
        )
        return now_gap
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
    else
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

    if noise && altmin_flag
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

    (n, m) = size(A)
    printlist = String[]

    add_message!(printlist, String[
        Dates.format(log_time, "e, dd u yyyy HH:MM:SS"), 
        "\n",
        (noise ? 
        "Starting branch-and-bound on a (noisy) matrix completion problem.\n" :
        "Starting branch-and-bound on a (noiseless) basis pursuit problem.\n"),
        Printf.@sprintf("k:                                              %15d\n", k),
        Printf.@sprintf("m:                                              %15d\n", m),
        Printf.@sprintf("n:                                              %15d\n", n),
        Printf.@sprintf("num_indices:                                    %15d\n", sum(indices)),
        (noise ? 
        Printf.@sprintf("(Noisy) γ:                                      %15g\n", γ) : ""),
        Printf.@sprintf("λ:                                              %15g\n", λ),
        "\n",
        Printf.@sprintf("Node selection:                                 %15s\n", node_selection),
        (node_selection == "bestfirst_depthfirst" ?
        Printf.@sprintf("Bestfirst-depthfirst cutoff:                    %15s\n", bestfirst_depthfirst_cutoff) : ""),
        Printf.@sprintf("Optimality gap:                                 %15g\n", gap),
        Printf.@sprintf("Only solve root node?:                          %15s\n", root_only),
        (!root_only && noise ?
        Printf.@sprintf("(Noisy) Do altmin at child nodes?:              %15s\n", altmin_flag) : ""),
        (!root_only && noise && altmin_flag ? 
        Printf.@sprintf("(Noisy) Max altmin probability:                 %15s\n", max_altmin_probability) : ""),
        (!root_only && noise && altmin_flag ? 
        Printf.@sprintf("(Noisy) Min altmin probability:                 %15s\n", min_altmin_probability) : ""),
        (!root_only && noise && altmin_flag ? 
        Printf.@sprintf("(Noisy) Altmin probability decay rate:          %15s\n", altmin_probability_decay_rate) : ""),
        Printf.@sprintf("Cap on nodes?                                   %15s\n", use_max_steps),
        (use_max_steps ?
        Printf.@sprintf("Maximum nodes:                                  %15d\n", max_steps) : ""),
        Printf.@sprintf("Time limit (s):                                 %15d\n", time_limit),
        "\n",
    ])
    if use_disjunctive_cuts
        add_message!(printlist, String[
            Printf.@sprintf("Use disjunctive cuts?:                          %15s\n", use_disjunctive_cuts),
            Printf.@sprintf("Disjunctive cuts type:                          %15s\n", disjunctive_cuts_type),
            Printf.@sprintf("Disjunction breakpoints:                        %15s\n", disjunctive_cuts_breakpoints),
        ])
    else
        add_message!(printlist, String[
            Printf.@sprintf("Use disjunctive cuts?:                          %15s\n", use_disjunctive_cuts),
            Printf.@sprintf("Branching type:                                 %15s\n", branching_type),
            Printf.@sprintf("Branching point:                                %15s\n", branch_point),
        ])
    end
    if !noise
        add_message!(printlist, String[
            (k == 1 ? 
            Printf.@sprintf("(Noiseless) Apply presolve?:                    %15s\n", presolve) : ""),
            (k == 1 && presolve ? 
            Printf.@sprintf("(Noiseless) Apply valid inequalities?:          %15s\n", add_basis_pursuit_valid_inequalities) : ""),
        ])
    end
    if use_disjunctive_cuts
        add_message!(printlist, String[
            Printf.@sprintf("Use Shor LMI inequalities?:                     %15s\n", add_Shor_valid_inequalities),
            (add_Shor_valid_inequalities ?
            Printf.@sprintf("Proportion of Shor LMI inequalities?:           %15f\n", add_Shor_valid_inequalities_fraction) : ""),
            (noise && add_Shor_valid_inequalities ? 
            Printf.@sprintf("(Noisy) (rank-1) Apply Shor LMI with            %15s entries.\n", Shor_valid_inequalities_noisy_rank1_num_entries_present) : ""),
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
        lower = Float64[],
        upper = Float64[],
        gap = Float64[],
        runtime = Float64[],
    )
    instance["run_details"] = OrderedDict(
        "noise" => noise,
        "k" => k,
        "m" => m,
        "n" => n,
        "A" => A,
        "indices" => indices,
        "num_indices" => convert(Int, round(sum(indices))),
        "γ" => γ,
        "λ" => λ,
        "node_selection" => node_selection,
        "bestfirst_depthfirst_cutoff" => bestfirst_depthfirst_cutoff,
        "optimality_gap" => gap,
        "root_only" => root_only,
        "altmin_flag" => altmin_flag,
        "max_altmin_probability" => max_altmin_probability,
        "min_altmin_probability" => min_altmin_probability,
        "altmin_probability_decay_rate" => altmin_probability_decay_rate,
        "use_max_steps" => use_max_steps,
        "max_steps" => max_steps,
        "time_limit" => time_limit,
        "use_disjunctive_cuts" => use_disjunctive_cuts,
        "disjunctive_cuts_type" => disjunctive_cuts_type,
        "disjunctive_cuts_breakpoints" => disjunctive_cuts_breakpoints,
        "presolve" => presolve,
        "add_basis_pursuit_valid_inequalities" => add_basis_pursuit_valid_inequalities,
        "add_Shor_valid_inequalities" => add_Shor_valid_inequalities,
        "add_Shor_valid_inequalities_fraction" => add_Shor_valid_inequalities_fraction,
        "add_Shor_valid_inequalities_iterative" => add_Shor_valid_inequalities_iterative,
        "max_update_Shor_indices_probability" => max_update_Shor_indices_probability,
        "min_update_Shor_indices_probability" => min_update_Shor_indices_probability,
        "update_Shor_indices_probability_decay_rate" => update_Shor_indices_probability_decay_rate,
        "update_Shor_indices_n_minors" => update_Shor_indices_n_minors,
        "Shor_valid_inequalities_noisy_rank1_num_entries_present" => Shor_valid_inequalities_noisy_rank1_num_entries_present,
        "branching_type" => branching_type,
        "branch_point" => branch_point,
        "log_time" => log_time,
        "start_time" => start_time,
        "end_time" => start_time,
        "time_taken" => 0.0,
        "entries_presolved" => sum(indices),
        "solve_time_altmin" => solve_time_altmin,
        "dict_solve_times_altmin" => dict_solve_times_altmin,
        "dict_num_iterations_altmin" => dict_num_iterations_altmin,
        "solve_time_relaxation_feasibility" => solve_time_relaxation_feasibility,
        "solve_time_relaxation" => solve_time_relaxation,
        "dict_solve_times_relaxation" => dict_solve_times_relaxation,
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

    if !noise && presolve
        indices_presolved, X_presolved = rankk_presolve(indices, A, k)
        instance["run_details"]["entries_presolved"] = sum(indices_presolved)
        X_initial = X_presolved
        U_initial = svd(X_initial).U[:,1:k]
        Y_initial = U_initial * U_initial'
        objective_initial = evaluate_objective(
            X_initial, A, indices, U_initial, γ, λ, noise
        )
        MSE_in_initial = compute_MSE(X_initial, A, indices, kind = "in")
        MSE_out_initial = compute_MSE(X_initial, A, indices, kind = "out")
        MSE_all_initial = compute_MSE(X_initial, A, indices, kind = "all")
        solution = Dict(
            "objective_initial" => objective_initial,
            "MSE_in_initial" => MSE_in_initial,
            "MSE_out_initial" => MSE_out_initial,
            "MSE_all_initial" => MSE_all_initial,
            "Y_initial" => Y_initial,
            "U_initial" => U_initial,
            "X_initial" => X_initial,
            "objective" => objective_initial,
            "MSE_in" => MSE_in_initial,
            "MSE_out" => MSE_out_initial,
            "MSE_all" => MSE_all_initial,
            "Y" => Y_initial,
            "U" => U_initial,
            "X" => X_initial,
        )
        if sum(indices_presolved) == m * n
            add_message!(printlist, String[
                "Solved in presolve stage.\n",
            ])            
            end_time = time()
            time_taken = end_time - start_time
            instance["run_details"]["end_time"] = end_time
            instance["run_details"]["time_taken"] = time_taken
            return solution, printlist, instance
        end
    else
        indices_presolved = indices
    end

    if noise
        altmin_A_initial = zeros(n, m)
        altmin_A_initial[indices] = A[indices]
        altmin_U_initial = svd(altmin_A_initial).U[:,1:k]

        altmin_results = @suppress alternating_minimization(
            A, n, k, indices, γ, λ, use_disjunctive_cuts,
            ;
            disjunctive_cuts_type = disjunctive_cuts_type,
            U_initial = altmin_U_initial,
            time_limit = time_limit,
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
        X_initial = altmin_results["U"] * altmin_results["V"]
        U_initial = svd(X_initial).U[:,1:k]
        Y_initial = U_initial * U_initial'
        objective_initial = evaluate_objective(
            X_initial, A, indices, U_initial, γ, λ, noise
        )
        MSE_in_initial = compute_MSE(X_initial, A, indices, kind = "in")
        MSE_out_initial = compute_MSE(X_initial, A, indices, kind = "out")
        MSE_all_initial = compute_MSE(X_initial, A, indices, kind = "all")
    else
        objective_initial = Inf
        MSE_in_initial = Inf
        MSE_out_initial = Inf
        MSE_all_initial = Inf
        Y_initial = zeros(Float64, (n, n))
        U_initial = zeros(Float64, (n, k))
        X_initial = zeros(Float64, (n, m))
    end
    
    solution = Dict(
        "objective_initial" => objective_initial,
        "MSE_in_initial" => MSE_in_initial,
        "MSE_out_initial" => MSE_out_initial,
        "MSE_all_initial" => MSE_all_initial,
        "Y_initial" => Y_initial,
        "U_initial" => U_initial,
        "X_initial" => X_initial,
        "objective" => objective_initial,
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
    upper = objective_initial
    lower = -Inf
    U_lower_initial = -ones(n, k)
    # Symmetry-breaking constraints
    for i in 1:k
        U_lower_initial[n-k+i:n,i] .= 0.0
    end
    U_upper_initial = ones(n, k)
    initial_node = BBNode(
        U_lower = U_lower_initial, 
        U_upper = U_upper_initial, 
        LB = lower,
        depth = 0,
        node_id = 1,
        parent_id = 0,
    )

    if !noise
        if add_basis_pursuit_valid_inequalities
            if presolve
                # compute indices for valid inequalities
                if k == 1
                    # Optimized version
                    initial_node.linear_coupling_constraints_indexes = generate_rank1_basis_pursuit_linear_coupling_constraints_indexes(indices_presolved)
                elseif k == 2
                    # Optimized version
                    initial_node.linear_coupling_constraints_indexes = generate_rank2_basis_pursuit_linear_coupling_constraints_indexes(indices_presolved)
                else
                    initial_node.linear_coupling_constraints_indexes = generate_rankk_basis_pursuit_linear_coupling_constraints_indexes(indices_presolved, k)
                end
            else
                error("""
                `add_basis_pursuit_valid_inequalities` only currently implemented together with `presolve`.
                """)
            end
        else
            initial_node.linear_coupling_constraints_indexes = Tuple{Vector{Int}, Vector{Int}, String, Int}[]
        end
    end

    if (add_Shor_valid_inequalities && !add_Shor_valid_inequalities_iterative)
        if !noise
            # Assuming presolve is done:
            initial_node.Shor_constraints_indexes = generate_rank1_basis_pursuit_Shor_constraints_indexes(indices_presolved, 1)
            initial_node.Shor_constraints_indexes = randsubseq(
                initial_node.Shor_constraints_indexes,
                add_Shor_valid_inequalities_fraction,
            )
            Shor_non_SOC_constraints_indexes = unique(vcat(
                [
                    [(i1,j1), (i1,j2), (i2,j1), (i2,j2)]
                    for (i1, i2, j1, j2) in initial_node.Shor_constraints_indexes
                ]...
            ))
            initial_node.Shor_SOC_constraints_indexes = setdiff(
                vec(collect(Iterators.product(1:n, 1:m))), 
                Shor_non_SOC_constraints_indexes,
                [(x[1], x[2]) for x in findall(indices_presolved)]
            )
        else
            initial_node.Shor_constraints_indexes = generate_rank1_matrix_completion_Shor_constraints_indexes(
                indices_presolved, 
                Shor_valid_inequalities_noisy_rank1_num_entries_present
            )
            initial_node.Shor_constraints_indexes = randsubseq(
                initial_node.Shor_constraints_indexes,
                add_Shor_valid_inequalities_fraction,
            )
            Shor_non_SOC_constraints_indexes = unique(vcat(
                [
                    [(i1,j1), (i1,j2), (i2,j1), (i2,j2)]
                    for (i1, i2, j1, j2) in initial_node.Shor_constraints_indexes
                ]...
            ))
            initial_node.Shor_SOC_constraints_indexes = setdiff(
                vec(collect(Iterators.product(1:n, 1:m))), 
                Shor_non_SOC_constraints_indexes
            )
        end
    elseif (add_Shor_valid_inequalities && add_Shor_valid_inequalities_iterative)
        initial_node.Shor_constraints_indexes = NTuple{4, Int}[]
        Shor_non_SOC_constraints_indexes = Tuple{Int, Int}[]
        initial_node.Shor_SOC_constraints_indexes = vec(collect(Iterators.product(1:n, 1:m)))
    elseif (!add_Shor_valid_inequalities && add_Shor_valid_inequalities_iterative)
        error("""
        Setting `add_Shor_valid_inequalities_iterative` to `true`
        requires `add_Shor_valid_inequalities` to be `true`.
        """)
    else
        initial_node.Shor_constraints_indexes = NTuple{4, Int}[]
        Shor_non_SOC_constraints_indexes = Tuple{Int, Int}[]
        initial_node.Shor_SOC_constraints_indexes = Tuple{Int, Int}[]
    end

    nodes[1] = initial_node

    add_message!(printlist, String[
        "-----------------------------------------------------------------------------------\n",
        "|   Explored |      Total |      Lower |      Upper |        Gap |    Runtime (s) |\n",
        "-----------------------------------------------------------------------------------\n",
    ])

    # leaves' mapping from node_id to lower_bound
    lower_bounds = PriorityQueue([1=>Inf])
    node_ids = [1]
    minimum_lower_bounds = Inf
    ancestry = Dict{Int, Vector{Int}}()

    while (
        now_gap > gap 
        && !(use_max_steps && (counter ≥ max_steps))
        && time() - start_time ≤ time_limit
    )
        if length(nodes) != 0
            if node_selection == "bestfirst_depthfirst"
                if length(nodes) > bestfirst_depthfirst_cutoff
                    node_selection_here = "depthfirst"
                else
                    node_selection_here = "bestfirst"
                end
            else
                node_selection_here = node_selection
            end

            if node_selection_here == "breadthfirst"
                id = popfirst!(node_ids)
                delete!(lower_bounds, id)
                current_node = pop!(nodes, id)
            elseif node_selection_here == "bestfirst"
                id = dequeue!(lower_bounds)
                deleteat!(node_ids, findfirst(isequal(id), node_ids))
                current_node = pop!(nodes, id)
            elseif node_selection_here == "depthfirst" # NOTE: may not work well
                id = pop!(node_ids)
                delete!(lower_bounds, id)
                current_node = pop!(nodes, id)
            end
            nodes_explored += 1
        else
            break
        end

        split_flag = true

        # possible, since we may not explore tree breadth-first
        # (should not be possible for breadth-first search)
        if current_node.LB > solution["objective"]
            split_flag = false
            nodes_dominated += 1
        end

        if !use_disjunctive_cuts && split_flag
            relax_feasibility_result = @suppress relax_feasibility_frob_matrixcomp(
                n, k, A, indices, noise;
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
                if !noise
                    relax_result = @suppress relax_frob_matrixcomp(
                        n, k, A, indices, indices_presolved, γ, λ, noise, use_disjunctive_cuts;
                        disjunctive_cuts_type = disjunctive_cuts_type,
                        add_basis_pursuit_valid_inequalities = add_basis_pursuit_valid_inequalities,
                        linear_coupling_constraints_indexes = current_node.linear_coupling_constraints_indexes,
                        add_Shor_valid_inequalities = add_Shor_valid_inequalities,
                        Shor_constraints_indexes = current_node.Shor_constraints_indexes,
                        Shor_SOC_constraints_indexes = current_node.Shor_SOC_constraints_indexes,
                        matrix_cuts = current_node.matrix_cuts,
                        time_limit = time_limit,
                    )
                else
                    relax_result = @suppress relax_frob_matrixcomp(
                        n, k, A, indices, indices_presolved, γ, λ, noise, use_disjunctive_cuts;
                        disjunctive_cuts_type = disjunctive_cuts_type,
                        add_basis_pursuit_valid_inequalities = false,
                        add_Shor_valid_inequalities = add_Shor_valid_inequalities,
                        Shor_constraints_indexes = current_node.Shor_constraints_indexes,
                        Shor_SOC_constraints_indexes = current_node.Shor_SOC_constraints_indexes,
                        matrix_cuts = current_node.matrix_cuts,
                        time_limit = time_limit,
                    )
                end
            else
                if noise 
                    add_basis_pursuit_valid_inequalities = false
                end
                relax_result = @suppress relax_frob_matrixcomp(
                    n, k, A, indices, indices_presolved, γ, λ, noise, false; 
                    U_lower = current_node.U_lower, 
                    U_upper = current_node.U_upper,
                    add_basis_pursuit_valid_inequalities = add_basis_pursuit_valid_inequalities,
                    time_limit = time_limit,
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
            if relax_result["feasible"] == false # should not happen, since this should be checked by relax_feasibility_frob_matrixcomp
                nodes_relax_infeasible += 1
                split_flag = false
            elseif relax_result["termination_status"] in [
                MOI.OPTIMAL,
                MOI.LOCALLY_SOLVED, # TODO: investigate this
                MOI.SLOW_PROGRESS, # TODO: investigate this
                MOI.TIME_LIMIT,
            ]
                nodes_relax_feasible += 1
                objective_relax = relax_result["objective"]
                Y_relax = relax_result["Y"]
                U_relax = relax_result["U"]
                X_relax = relax_result["X"]
                Θ_relax = relax_result["Θ"]
                α_relax = relax_result["α"]
                if current_node.node_id == 1
                    lower = objective_relax
                end
                # if solution for relax_result has higher objective than best found so far: prune the node
                if objective_relax > solution["objective"]
                    nodes_relax_feasible_pruned += 1
                    split_flag = false            
                end
            end
        end

        # if solution for relax_result is feasible for original problem:
        # prune this node;
        # if it is the best found so far, update solution
        if split_flag
            if master_problem_frob_matrixcomp_feasible(Y_relax, U_relax, X_relax, Θ_relax, use_disjunctive_cuts)
                current_node.master_feasible = true
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
        if altmin_flag && noise # only perform alternating minimization in the noisy setting
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
                    altmin_results_BB = @suppress alternating_minimization(
                        A, n, k, indices, γ, λ, use_disjunctive_cuts;
                        disjunctive_cuts_type = disjunctive_cuts_type,
                        U_initial = Matrix(U_rounded),
                        matrix_cuts = current_node.matrix_cuts,
                        time_limit = time_limit,
                    )
                else
                    altmin_results_BB = @suppress alternating_minimization(
                        A, n, k, indices, γ, λ, use_disjunctive_cuts;
                        U_initial = Matrix(U_rounded),
                        U_lower = current_node.U_lower,
                        U_upper = current_node.U_upper,
                        time_limit = time_limit,
                    )
                end
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
                        X_local, A, indices, U_local, γ, λ, noise
                    )
                    if objective_local < solution["objective"]
                        nodes_relax_feasible_split_altmin_improvement += 1
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
                    Y = Y_relax, 
                    U = U_relax,
                    X = X_relax,
                    indices = indices_presolved,
                    counter = counter,
                    objective_relax = objective_relax,
                    update_Shor_indices_flag = update_Shor_indices_flag_now,
                    Shor_valid_inequalities_noisy_rank1_num_entries_present = Shor_valid_inequalities_noisy_rank1_num_entries_present,
                    n_minors = update_Shor_indices_n_minors,
                )
                merge!(
                    nodes, 
                    Dict(
                        (counter + i) => node
                        for (i, node) in enumerate(matrix_cut_child_nodes)
                    )
                )
                new_node_ids = collect(counter+1:counter+length(matrix_cut_child_nodes))
                append!(node_ids, new_node_ids)
                for id in new_node_ids
                    enqueue!(lower_bounds, id => objective_relax)
                end
                ancestry[current_node.node_id] = new_node_ids
                counter += length(matrix_cut_child_nodes)
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
                    # initialize a node's LB with the objective of relaxation of parent
                    LB = objective_relax,
                    depth = current_node.depth + 1,
                    node_id = counter + 1,
                    parent_id = current_node.node_id,
                )
                right_child_node = BBNode(
                    U_lower = U_lower_right,
                    U_upper = U_upper_right,
                    # initialize a node's LB with the objective of relaxation of parent
                    LB = objective_relax,
                    node_id = counter + 2,
                    depth = current_node.depth + 1,
                    parent_id = current_node.node_id,
                )
                merge!(
                    nodes, 
                    Dict(
                        counter + 1 => left_child_node,
                        counter + 2 => right_child_node,
                    )
                )
                append!(node_ids, [counter + 1, counter + 2])
                enqueue!(lower_bounds, counter + 1 => objective_relax)
                enqueue!(lower_bounds, counter + 2 => objective_relax)
                ancestry[current_node.node_id] = [counter + 1, counter + 2]
                counter += 2
            end
        end

        # cleanup actions - to be done regardless of whether split_flag was true or false
        if current_node.node_id != 1
            ancestry[current_node.parent_id] = setdiff(ancestry[current_node.parent_id], [current_node.node_id])
            if length(ancestry[current_node.parent_id]) == 0
                delete!(ancestry, current_node.parent_id)
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
            _, minimum_lower_bounds = peek(lower_bounds)
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
                || (use_max_steps && counter ≥ max_steps)
                || time() - start_time > time_limit
            )
                now_gap = add_update!(
                    printlist, instance, nodes_explored, counter, 
                    lower, upper, start_time,
                )
                last_updated_counter = counter

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

    instance["run_details"]["nodes_explored"] = nodes_explored
    instance["run_details"]["nodes_total"] = counter
    instance["run_details"]["nodes_dominated"] = nodes_dominated
    instance["run_details"]["nodes_relax_infeasible"] = nodes_relax_infeasible
    instance["run_details"]["nodes_relax_feasible"] = nodes_relax_feasible
    instance["run_details"]["nodes_relax_feasible_pruned"] = nodes_relax_feasible_pruned
    instance["run_details"]["nodes_master_feasible"] = nodes_master_feasible
    instance["run_details"]["nodes_master_feasible_improvement"] = nodes_master_feasible_improvement
    instance["run_details"]["nodes_relax_feasible_split"] = nodes_relax_feasible_split
    instance["run_details"]["nodes_relax_feasible_split_altmin"] = nodes_relax_feasible_split_altmin
    instance["run_details"]["nodes_relax_feasible_split_altmin_improvement"] = nodes_relax_feasible_split_altmin_improvement

    add_message!(printlist, String["\n\nRun details:\n"])
    add_message!(printlist, String[
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
    ])
    if !use_disjunctive_cuts
        for item in ranges
            add_message!(printlist, String[
                Printf.@sprintf("\n\nnode_id: %10d\n", item[1]),
                "\nU_lower:\n",
                sprint(show, "text/plain", item[2]),
                "\nU_upper:\n",
                sprint(show, "text/plain", item[3]),
            ])               
        end
    end
    add_message!(printlist, String[
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
    ])

    return solution, printlist, instance
end

"""
    master_problem_frob_matrixcomp_feasible(
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
function master_problem_frob_matrixcomp_feasible(
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

function relax_feasibility_frob_matrixcomp( # this is the version without matrix_cuts
    n::Int,
    k::Int,
    A::Array{Float64, 2},
    indices::BitMatrix,
    noise::Bool,
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

function relax_frob_matrixcomp(
    n::Int,
    k::Int,
    A::Array{Float64,2},
    indices::BitMatrix, # for objective computation
    indices_presolved::BitMatrix, # for coupling constraints, in the noiseless case
    γ::Float64,
    λ::Float64,
    noise::Bool,
    use_disjunctive_cuts::Bool,
    ;
    disjunctive_cuts_type::Union{String, Nothing} = nothing,
    add_basis_pursuit_valid_inequalities::Bool = false,
    linear_coupling_constraints_indexes::Vector{Tuple{Vector{Int}, Vector{Int}, String, Int}} = Tuple{Vector{Int}, Vector{Int}, String, Int}[],
    add_Shor_valid_inequalities::Bool = false,
    Shor_constraints_indexes::Vector{NTuple{4, Int}} = NTuple{4, Int}[],
    Shor_SOC_constraints_indexes::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[],
    U_lower::Array{Float64,2} = begin
        U_lower = -ones(n,k)
        for i in 1:k
            U_lower[n-k+i:n,i] .= 0.0
        end
        U_lower
    end,
    U_upper::Array{Float64,2} = ones(n,k),
    matrix_cuts::Vector{Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}} = Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}[],
    orthogonality_tolerance::Float64 = 0.0,
    solver_output::Int = 0,
    time_limit::Int = 3600, # forces Mosek to not use a time limit
)
    # TODO: only for noisy case: remove?
    function compute_α(Y, γ, A, indices)
        (n, m) = size(A)
        α = zeros(size(A))
        for j in 1:m
            for i in 1:n
                if indices[i,j]
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
        && size(A, 1) == size(indices, 1) == size(indices_presolved, 1) == n
        && size(A, 2) == size(indices, 2) == size(indices_presolved, 2) 
    )
        error("""
        Dimension mismatch. 
        Input matrix U_lower must have size (n, k); 
        Input matrix U_upper must have size (n, k); 
        Input matrix A must have size (n, m);
        Input matrix indices must have size (n, m);
        Input matrix indices_presolved must have size (n, m).""")
    end

    (n, k) = size(U_lower)
    (n, m) = size(A)

    model = Model(Mosek.Optimizer)
    if solver_output == 0
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
    end
    set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
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
                for (i1, i2, j1, j2) in Shor_constraints_indexes
            ]...
        ))
        Shor_constraints_coords_i = sort(unique([x[1] for x in Shor_constraints_coords]))
        Shor_constraints_coords_j = sort(unique([x[2] for x in Shor_constraints_coords]))
        if k == 1
            @variable(model, V1[
                Shor_constraints_coords_i, 
                [tuple(x...) for x in combinations(Shor_constraints_coords_j, 2)]
            ])
            @variable(model, V2[
                [tuple(x...) for x in combinations(Shor_constraints_coords_i, 2)], 
                Shor_constraints_coords_j
            ])
            @variable(model, V3[
                [tuple(x...) for x in combinations(Shor_constraints_coords_i, 2)], 
                [tuple(x...) for x in combinations(Shor_constraints_coords_j, 2)]
            ])
        else
            # Generate only some Wt, H, V
            @variable(model, Wt[
                1:k, 
                Shor_constraints_coords
            ] ≥ 0)
            @variable(model, H[
                [tuple(x...) for x in combinations(1:k, 2)], 
                Shor_constraints_coords
            ])
            @variable(model, V1[
                1:k, 
                Shor_constraints_coords_i, 
                [tuple(x...) for x in combinations(Shor_constraints_coords_j, 2)]
            ])
            @variable(model, V2[
                1:k, 
                [tuple(x...) for x in combinations(Shor_constraints_coords_i, 2)], 
                Shor_constraints_coords_j
            ])
            @variable(model, V3[
                1:k, 
                [tuple(x...) for x in combinations(Shor_constraints_coords_i, 2)], 
                [tuple(x...) for x in combinations(Shor_constraints_coords_j, 2)]
            ])
        end
    end

    # If noiseless, coupling constraints between X and A
    if !noise
        if k == 1
            for i in 1:n, j in 1:m
                if indices_presolved[i,j]
                    @constraint(model, X[i,j] == A[i,j])
                end
            end
        else
            for i in 1:n, j in 1:m
                if indices_presolved[i,j]
                    @constraint(model, X[i,j] == A[i,j])
                end
            end
        end
    end

    @constraint(model, LinearAlgebra.Symmetric([Y X; X' Θ]) in PSDCone())
    @constraint(model, LinearAlgebra.Symmetric([Y U; U' I]) in PSDCone())
    @constraint(model, LinearAlgebra.Symmetric(I - Y) in PSDCone())
    # Trace constraint on Y
    @constraint(model, sum(Y[i,i] for i in 1:n) <= k)

    # Lower bounds and upper bounds on U
    @constraint(model, [i=1:n, j=1:k], U_lower[i,j] ≤ U[i,j] ≤ U_upper[i,j])

    # matrix cuts on U, if supplied
    if use_disjunctive_cuts
        if length(matrix_cuts) > 0
            L = length(matrix_cuts)
            if disjunctive_cuts_type in ["linear", "linear2", "linear3"]
                @expression(
                    model,
                    matrix_cut[l=1:L],
                    zero(AffExpr),
                ) # stores the RHS of the linear inequality in U and Y
                @variable(model, v[1:L, 1:k]) # stores each U[:,t]' * x[:,t]
                v̂ = zeros(L, k) # stores fitted version: each Ű[:,t]' * x[:,t]
                
                # Constraints linking v (or w) to previous fitted Us and breakpoint vectors
                for (l, (breakpoint_vec, Û, _)) in enumerate(matrix_cuts)
                    @constraint(
                        model, 
                        v[l,:] .== U' * breakpoint_vec,
                    )
                    v̂[l,:] = Û' * breakpoint_vec
                end
                
                # Constraints linking v to breakpoints
                expressions = Dict()
                if disjunctive_cuts_type == "linear"
                    for (l, (_, _, directions)) in enumerate(matrix_cuts)
                        for j in 1:k
                            if directions[j] == "left"
                                @constraint(model, -1 ≤ v[l,j])
                                @constraint(model, v[l,j] ≤ v̂[l,j])
                                expressions[(l,j)] = @expression(model, - v[l,j] + v̂[l,j] * v[l,j] + v̂[l,j])
                            elseif directions[j] == "right"
                                @constraint(model, v̂[l,j] ≤ v[l,j])
                                @constraint(model, v[l,j] ≤ 1)
                                expressions[(l,j)] = @expression(model, + v[l,j] + v̂[l,j] * v[l,j] - v̂[l,j])
                            end
                        end
                    end
                elseif disjunctive_cuts_type == "linear2"
                    for (l, (_, _, directions)) in enumerate(matrix_cuts)
                        for j in 1:k
                            if directions[j] == "left"
                                @constraint(model, -1 ≤ v[l,j])
                                @constraint(model, v[l,j] ≤ - abs(v̂[l,j]))
                                expressions[(l,j)] = @expression(model, - v[l,j] - abs(v̂[l,j]) * v[l,j] - abs(v̂[l,j]))
                            elseif directions[j] == "middle"
                                @constraint(model, - abs(v̂[l,j]) ≤ v[l,j])
                                @constraint(model, v[l,j] ≤ abs(v̂[l,j]))
                                expressions[(l,j)] = @expression(model, (v̂[l,j])^2)
                            elseif directions[j] == "right"
                                @constraint(model, abs(v̂[l,j]) ≤ v[l,j])
                                @constraint(model, v[l,j] ≤ 1)
                                expressions[(l,j)] = @expression(model, + v[l,j] + abs(v̂[l,j]) * v[l,j] - abs(v̂[l,j]))
                            end
                        end
                    end
                elseif disjunctive_cuts_type == "linear3"
                    for (l, (_, _, directions)) in enumerate(matrix_cuts)
                        for j in 1:k
                            if directions[j] == "left"
                                @constraint(model, -1 ≤ v[l,j])
                                @constraint(model, v[l,j] ≤ - abs(v̂[l,j]))
                                expressions[(l,j)] = @expression(model, - v[l,j] - abs(v̂[l,j]) * v[l,j] - abs(v̂[l,j]))
                            elseif directions[j] == "inner_left"
                                @constraint(model, - abs(v̂[l,j]) ≤ v[l,j])
                                @constraint(model, v[l,j] ≤ 0)
                                expressions[(l,j)] = @expression(model, - abs(v̂[l,j]) * v[l,j])
                            elseif directions[j] == "inner_right"
                                @constraint(model, 0 ≤ v[l,j])
                                @constraint(model, v[l,j] ≤ abs(v̂[l,j]))
                                expressions[(l,j)] = @expression(model, abs(v̂[l,j]) * v[l,j])
                            elseif directions[j] == "right"
                                @constraint(model, abs(v̂[l,j]) ≤ v[l,j])
                                @constraint(model, v[l,j] ≤ 1)
                                expressions[(l,j)] = @expression(model, abs(v̂[l,j]) * v[l,j])
                            end
                        end
                    end
                end
                for (l, (breakpoint_vec, _, _)) in enumerate(matrix_cuts)
                    for j in 1:k
                        add_to_expression!(matrix_cut[l], expressions[(l,j)])
                    end
                    # aggregated constraint
                    @constraint(
                        model,
                        matrix_cut[l] ≥ LinearAlgebra.dot(Y, breakpoint_vec * breakpoint_vec'),
                    )
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

    if !noise && add_basis_pursuit_valid_inequalities
        for (R, C, type, ind) in linear_coupling_constraints_indexes
            if type == "col"
                @constraint(
                    model, 
                    sum(
                        (-1)^i * X[R[i],ind] * det(A[setdiff(R,R[i]),setdiff(C,ind)])
                        for i in 1:(k+1)
                    ) == 0
                )
            elseif type == "row"
                @constraint(
                    model, 
                    sum(
                        (-1)^i * X[ind,C[i]] * det(A[setdiff(R,ind),setdiff(C,C[i])])
                        for i in 1:(k+1)
                    ) == 0
                )
            end
        end
    end

    if add_Shor_valid_inequalities
        if k == 1
            for (i,j) in Shor_SOC_constraints_indexes
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
            for (i1, i2, j1, j2) in Shor_constraints_indexes
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
            for (i,j) in Shor_SOC_constraints_indexes
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
            for (i1, i2, j1, j2) in Shor_constraints_indexes
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
                for (t1, t2) in combinations(1:k, 2)
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
        [j = 1:k],
        [
            1;
            U[:,j]
        ] in SecondOrderCone()
    )

    if !noise
        @objective(
            model,
            Min,
            (1 / (2 * γ)) * sum(Θ[i, i] for i = 1:m)
            + λ * sum(Y[i, i] for i = 1:n)
        )
    else
        if add_Shor_valid_inequalities
            @objective(
                model,
                Min,
                (1 / 2) * sum(
                    (A[i,j]^2 - 2 * A[i,j] * X[i,j] + W[i,j]) * indices[i, j] 
                    for i = 1:n, j = 1:m
                ) 
                + (1 / (2 * γ)) * sum(Θ[i, i] for i = 1:m) 
                + λ * sum(Y[i, i] for i = 1:n)
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
                + λ * sum(Y[i, i] for i = 1:n)
            )
        end
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
        if !noise
            results["objective"] = (
                (1 / (2 * γ)) * sum(value.(Θ)[i, i] for i = 1:m)
                + λ * sum(value.(Y)[i, i] for i = 1:n)
            )
        else
            if add_Shor_valid_inequalities
                results["objective"] = (
                    (1 / 2) * sum(
                        (A[i,j]^2 - 2 * A[i,j] * value.(X)[i,j] + value.(W)[i,j]) * indices[i, j] 
                        for i = 1:n, j = 1:m
                    ) 
                    + (1 / (2 * γ)) * sum(value.(Θ)[i, i] for i = 1:m) 
                    + λ * sum(value.(Y)[i, i] for i = 1:n)
                )
            else
                results["objective"] = (
                    (1 / 2) * sum(
                        (A[i,j] - value.(X)[i,j])^2 * indices[i, j] 
                        for i = 1:n, j = 1:m
                    ) 
                    + (1 / (2 * γ)) * sum(value.(Θ)[i, i] for i = 1:m) 
                    + λ * sum(value.(Y)[i, i] for i = 1:n)
                )
            end
        end
        results["α"] = compute_α(value.(Y), γ, A, indices)
        results["Y"] = value.(Y)
        results["U"] = value.(U)
        results["X"] = value.(X)
        results["Θ"] = value.(Θ)
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
        if (
            use_disjunctive_cuts 
            && length(matrix_cuts) > 0
            && disjunctive_cuts_type in ["linear", "linear2", "linear3"]
        )
            results["v"] = value.(v)
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

function alternating_minimization(
    A::Array{Float64,2},
    n::Int,
    k::Int,
    indices::BitMatrix,
    γ::Float64,
    λ::Float64,
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
    matrix_cuts::Vector{Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}} = Tuple{Vector{Float64}, Matrix{Float64}, Vector{String}}[],
    ϵ::Float64 = 1e-5,
    orthogonality_tolerance::Float64 = 1e-8,
    max_iters::Int = 100,
    time_limit::Int = 3600,
)
    # Note: only used in the noisy case
    altmin_start_time = time()

    (n, m) = size(A)
        
    U_current = U_initial

    counter = 0
    objective_current = 1e10

    model_U = Model(Mosek.Optimizer)
    set_silent(model_U)
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
        if length(matrix_cuts) > 0
            L = length(matrix_cuts)
            if disjunctive_cuts_type in ["linear", "linear2", "linear3"]
                @variable(model_U, v[1:L, 1:k]) # stores each U[:,t]' * x[:,t]
                v̂ = zeros(L, k) # stores fitted version: each Ű[:,t]' * x[:,t]

                # Constraints linking v (or w) to previous fitted Us and breakpoint vectors
                for (l, (breakpoint_vec, Û, _)) in enumerate(matrix_cuts)
                    @constraint(
                        model_U, 
                        v[l,:] .== U' * breakpoint_vec,
                    )
                    v̂[l,:] = Û' * breakpoint_vec
                end

                # Constraints linking v to breakpoints
                if disjunctive_cuts_type == "linear"   
                    for (l, (_, _, directions)) in enumerate(matrix_cuts)
                        for j in 1:k
                            if directions[j] == "left"
                                @constraint(model_U, -1 ≤ v[l,j])
                                @constraint(model_U, v[l,j] ≤ v̂[l,j])
                            elseif directions[j] == "right"
                                @constraint(model_U, v̂[l,j] ≤ v[l,j])
                                @constraint(model_U, v[l,j] ≤ 1)
                            end
                        end
                    end
                elseif disjunctive_cuts_type == "linear2"
                    for (l, (_, _, directions)) in enumerate(matrix_cuts)
                        for j in 1:k
                            if directions[j] == "left"
                                @constraint(model_U, -1 ≤ v[l,j])
                                @constraint(model_U, v[l,j] ≤ - abs(v̂[l,j]))
                            elseif directions[j] == "middle"
                                @constraint(model_U, - abs(v̂[l,j]) ≤ v[l,j])
                                @constraint(model_U, v[l,j] ≤ abs(v̂[l,j]))
                            elseif directions[j] == "right"
                                @constraint(model_U, abs(v̂[l,j]) ≤ v[l,j])
                                @constraint(model_U, v[l,j] ≤ 1)
                            end
                        end
                    end
                elseif disjunctive_cuts_type == "linear3"
                    for (l, (_, _, directions)) in enumerate(matrix_cuts)
                        for j in 1:k
                            if directions[j] == "left"
                                @constraint(model_U, -1 ≤ v[l,j])
                                @constraint(model_U, v[l,j] ≤ - abs(v̂[l,j]))
                            elseif directions[j] == "inner_left"
                                @constraint(model_U, - abs(v̂[l,j]) ≤ v[l,j])
                                @constraint(model_U, v[l,j] ≤ 0)
                            elseif directions[j] == "inner_right"
                                @constraint(model_U, 0 ≤ v[l,j])
                                @constraint(model_U, v[l,j] ≤ abs(v̂[l,j]))
                            elseif directions[j] == "right"
                                @constraint(model_U, abs(v̂[l,j]) ≤ v[l,j])
                                @constraint(model_U, v[l,j] ≤ 1)
                            end
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
    set_silent(model_V)
    @variable(model_V, V[1:k, 1:m])

    objectives = []
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
            objective_new = objective_value(model_U)
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
    )
end

function evaluate_objective(
    X::Array{Float64,2},
    A::Array{Float64,2},
    indices::BitMatrix,
    U::Array{Float64,2},
    γ::Float64,
    λ::Float64,
    noise::Bool,
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
    if !noise
        return (
            (1 / (2 * γ)) * sum(X.^2)
            + λ * sum(U.^2)
        )
    else
        return (
            (1 / 2) * sum(
                (X[i,j] - A[i,j])^2 * indices[i,j]
                for i = 1:n, j = 1:m
            )
            + (1 / (2 * γ)) * sum(X.^2)
            + λ * sum(U.^2)
        )
    end
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
    Y::Union{Matrix{Float64}, Nothing} = nothing,
    U::Matrix{Float64},
    X::Matrix{Float64},
    indices::BitMatrix,
    counter::Int,
    objective_relax::Float64,
    update_Shor_indices_flag::Bool = false,
    Shor_valid_inequalities_noisy_rank1_num_entries_present::Vector{Int} = [4],
    n_minors::Union{Int, Nothing} = 100,
)
    # matrix_cuts is a vector of tuples: each containing:
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
        if update_Shor_indices_flag
            minors = generate_violated_Shor_minors(
                reshape(X, (1, n, m)), 
                indices,
                Shor_valid_inequalities_noisy_rank1_num_entries_present,
                node.Shor_constraints_indexes,
                n_minors,
            )
            Shor_constraints_indexes = union(
                node.Shor_constraints_indexes,
                [
                    m[2] for m in minors
                ]
            )
            Shor_non_SOC_constraints_indexes = unique(vcat(
                [
                    [(i1,j1), (i1,j2), (i2,j1), (i2,j2)]
                    for (i1, i2, j1, j2) in Shor_constraints_indexes
                ]...
            ))
            Shor_SOC_constraints_indexes = setdiff(
                node.Shor_SOC_constraints_indexes, 
                Shor_non_SOC_constraints_indexes
            )
        else
            Shor_constraints_indexes = node.Shor_constraints_indexes
            Shor_SOC_constraints_indexes = node.Shor_SOC_constraints_indexes
        end
        return (
            BBNode(
                U_lower = node.U_lower,
                U_upper = node.U_upper,
                matrix_cuts = vcat(node.matrix_cuts, [(breakpoint_vec, U, collect(directions))]),
                # initialize a node's LB with the objective of relaxation of parent
                LB = objective_relax,
                depth = node.depth + 1,
                node_id = counter + ind,
                parent_id = node.node_id,
                linear_coupling_constraints_indexes = node.linear_coupling_constraints_indexes,
                Shor_constraints_indexes = Shor_constraints_indexes,
                Shor_SOC_constraints_indexes = Shor_SOC_constraints_indexes,
            )
            for (ind, directions) in directions_list
        )
    end
end

function rank1_presolve(
    indices::BitMatrix,
    A::Array{Float64, 2},
)
    indices_presolved = copy(indices)
    (n, m) = size(indices)
    X_presolved = zeros(Float64, (n, m))
    X_presolved[indices] = A[indices]

    for j0 in 1:m
        selected_rows = findall(indices_presolved[:,j0])
        if length(selected_rows) > 1
            rows_or = any(indices_presolved[selected_rows, :], dims=1)
            for j in findall(vec(rows_or))
                if j == j0
                    continue
                end
                i0 = selected_rows[findfirst(indices_presolved[selected_rows, j])]
                for i in selected_rows
                    if i == i0
                        continue
                    end
                    if !indices_presolved[i,j]
                        X_presolved[i,j] = X_presolved[i,j0] * X_presolved[i0,j] / X_presolved[i0,j0]
                        indices_presolved[i,j] = true
                    end
                end
            end
        end
    end

    return indices_presolved, X_presolved
end

function rank2_presolve(
    indices::BitMatrix,
    A::Array{Float64, 2},
)
    indices_presolved = copy(indices)
    (n, m) = size(indices)
    X_presolved = zeros(Float64, (n, m))
    X_presolved[indices] = A[indices]
    current_sum = sum(indices_presolved)

    while true
        for (j0, j1) in combinations(1:m, 2)
            selected_rows = findall(indices_presolved[:,j0] .& indices_presolved[:,j1])
            if length(selected_rows) ≤ 2
                continue
            end
            for j2 in setdiff(1:m, [j0, j1])
                filled_rows = selected_rows[findall(indices_presolved[selected_rows, j2])]
                if !(2 ≤ length(filled_rows) < length(selected_rows))
                    continue
                end
                (i0, i1) = filled_rows[1:2]
                for i2 in setdiff(selected_rows, filled_rows)
                    X_presolved[i2,j2] = (
                        X_presolved[i0,j0] * X_presolved[i1,j2] * X_presolved[i2,j1] 
                        + X_presolved[i0,j2] * X_presolved[i1,j1] * X_presolved[i2,j0] 
                        - X_presolved[i0,j1] * X_presolved[i1,j2] * X_presolved[i2,j0] 
                        - X_presolved[i0,j2] * X_presolved[i1,j0] * X_presolved[i2,j1] 
                    ) / (
                        X_presolved[i0,j0] * X_presolved[i1,j1] 
                        - X_presolved[i0,j1] * X_presolved[i1,j0]
                    )
                    indices_presolved[i2,j2] = true
                end
            end
        end
        if current_sum == sum(indices_presolved)
            break
        else
            current_sum = sum(indices_presolved)
        end
    end
    return indices_presolved, X_presolved
end

function rankk_presolve(
    indices::BitMatrix,
    A::Array{Float64, 2},
    k::Int,
)
    indices_presolved = copy(indices)
    (n, m) = size(indices)
    X_presolved = zeros(Float64, (n, m))
    X_presolved[indices] = A[indices]
    current_sum = sum(indices_presolved)

    while true
        for C in combinations(1:m, k)
            selected_rows = findall(vec(all(indices_presolved[:,C], dims=2)))
            if length(selected_rows) ≤ k
                continue
            end
            for j in setdiff(1:m, C)
                filled_rows = selected_rows[
                    findall(indices_presolved[selected_rows, j])
                ]
                if !(k ≤ length(filled_rows) < length(selected_rows))
                    continue
                end
                R = filled_rows[1:k]
                for i in setdiff(selected_rows, filled_rows)
                    r_ind = searchsortedfirst(R, i) - 1
                    R_top = R[1:r_ind]
                    R_bot = R[r_ind+1:end]
                    cumsum = 0
                    for ind in 1:r_ind
                        cumsum += (-1)^(ind-1) * X_presolved[R[r_ind+1-ind],j] * det(X_presolved[union(setdiff(R_top, R_top[r_ind+1-ind]), i, R_bot),collect(C)])
                    end
                    for ind in 1:k-r_ind
                        cumsum += (-1)^(ind-1) * X_presolved[R[r_ind+ind],j] * det(X_presolved[union(R_top, i, setdiff(R_bot, R_bot[ind]), i),collect(C)])
                    end
                    X_presolved[i,j] = cumsum / det(X_presolved[R,collect(C)])
                    indices_presolved[i,j] = true
                end
            end
        end
        if current_sum == sum(indices_presolved)
            break
        else
            current_sum = sum(indices_presolved)
        end
    end
    return indices_presolved, X_presolved
end

function generate_rank1_basis_pursuit_linear_coupling_constraints_indexes(
    indices_presolved::BitMatrix, # for coupling constraints, in the noiseless case
)
    # Used in basis pursuit: linear inequalities in rank-1
    (n, m) = size(indices_presolved)

    rowp = sortperm([indices_presolved[i,:] for i in 1:n], rev = true)
    colp = sortperm([indices_presolved[rowp,j] for j in 1:m], rev = true)
    rowind = unique([findfirst(indices_presolved[rowp, j]) for j in colp])
    colind = unique([findfirst(indices_presolved[i, colp]) for i in rowp])

    linear_coupling_constraints_indexes = Tuple{Vector{Int}, Vector{Int}, String, Int}[]
    for block_i in 1:length(rowind), block_j in 1:length(colind)
        if block_i == block_j
            continue
        end
        i_start = rowind[block_i]
        i_end = (
            block_i == length(rowind)
            ? n
            : rowind[block_i+1]-1
        )
        j_start = colind[block_j]
        j_end = (
            block_j == length(colind)
            ? m
            : colind[block_j+1]-1
        )
        j_ref = colind[block_i]
        i_ref = rowind[block_j]
        # (rowp[i_ref], colp[j_start:j_end]) are filled
        # (rowp[i_start:i_end], colp[j_ref]) are filled
        if (i_end > i_start)
            for i in (i_start+1):i_end, j in j_start:j_end
                if j_ref < j
                    push!(linear_coupling_constraints_indexes, (
                        [rowp[i_start], rowp[i]],
                        [colp[j_ref], colp[j]],
                        "col",
                        colp[j],
                    ))
                else
                    push!(linear_coupling_constraints_indexes, (
                        [rowp[i_start], rowp[i]],
                        [colp[j], colp[j_ref]],
                        "col",
                        colp[j],
                    ))
                end
            end
        end
        if (j_end > j_start)
            for j in (j_start+1):j_end
                if i_ref < i_start
                    push!(linear_coupling_constraints_indexes, (
                        [rowp[i_ref], rowp[i_start]],
                        [colp[j_start], colp[j]],
                        "row",
                        rowp[i_start],
                    ))
                else
                    push!(linear_coupling_constraints_indexes, (
                        [rowp[i_start], rowp[i_ref]],
                        [colp[j_start], colp[j]],
                        "row",
                        rowp[i_start],
                    ))
                end
            end
        end
    end

    return linear_coupling_constraints_indexes
end


function generate_rank2_basis_pursuit_linear_coupling_constraints_indexes(
    indices_presolved::BitMatrix, # for coupling constraints, in the noiseless case
)
    (n, m) = size(indices_presolved)

    constrained_vars_R = [
        Vector{Int}[] for i in 1:n
    ]
    constrained_vars_C = [
        Vector{Int}[] for j in 1:m
    ]
    linear_coupling_constraints_indexes = Tuple{Vector{Int}, Vector{Int}, String, Int}[]
    for (j0, j1) in combinations(1:m, 2)
        selected_rows = findall(indices[:,j0] .& indices[:,j1])
        if length(selected_rows) ≤ 2
            continue
        end
        for j in setdiff(1:m, [j0, j1])
            C_new = [j0, j1, j]
            colp = sortperm(C_new)
            i_choices = findall(indices[:,j0] .& indices[:,j1] .& indices[:,j])
            if length(i_choices) == 0
                R_range = [
                    collect(x) 
                    for x in Iterators.product(
                        selected_rows[1:1],
                        selected_rows[2:2],
                        selected_rows[3:end],
                    )
                ]
            elseif length(i_choices) == 1
                i = i_choices[1]
                R_range = [
                    sort(collect(x))
                    for x in Iterators.product(
                        [i],
                        setdiff(selected_rows, i)[1:1],
                        setdiff(selected_rows, i)[2:end],
                    )
                ]
            else
                continue
            end
            for R in R_range
                if R in constrained_vars_C[j]
                    continue
                end
                push!(constrained_vars_C[j], R)
                push!(linear_coupling_constraints_indexes, (
                    R, 
                    C_new[colp],
                    "col",
                    j,
                ))
            end
        end
    end

    for (i0, i1) in combinations(1:m, 2)
        selected_cols = findall(indices[:,i0] .& indices[:,i1])
        if length(selected_cols) ≤ 2
            continue
        end
        for i in setdiff(1:n, [i0, i1])
            R_new = [i0, i1, i]
            rowp = sortperm(R_new)
            j_choices = findall(indices[i0,:] .& indices[i1,:] .& indices[i,:])
            if length(j_choices) == 0
                C_range = [
                    collect(x) 
                    for x in Iterators.product(
                        selected_cols[1:1],
                        selected_cols[2:2],
                        selected_cols[3:end],
                    )
                ]
            elseif length(j_choices) == 1
                j = j_choices[1]
                C_range = [
                    sort(collect(x))
                    for x in Iterators.product(
                        [j],
                        setdiff(selected_cols, j)[1:1],
                        setdiff(selected_cols, j)[2:end],
                    )
                ]
            else
                continue
            end
            for C in C_range
                if C in constrained_vars_R[i]
                    continue
                end
                push!(constrained_vars_R[i], C)
                push!(linear_coupling_constraints_indexes, (
                    R_new[rowp],
                    C,
                    "row",
                    i,
                ))
            end
        end
    end

    return linear_coupling_constraints_indexes
end

function generate_rankk_basis_pursuit_linear_coupling_constraints_indexes(
    indices_presolved::BitMatrix, # for coupling constraints, in the noiseless case
    k::Int,
)
    (n, m) = size(indices_presolved)

    constrained_vars_R = [
        Vector{Int}[] for i in 1:n
    ]
    constrained_vars_C = [
        Vector{Int}[] for j in 1:m
    ]
    linear_coupling_constraints_indexes = Tuple{Vector{Int}, Vector{Int}, String, Int}[]
    for C in combinations(1:m, k)
        selected_rows = findall(vec(all(indices_presolved[:,C], dims=2)))
        if length(selected_rows) ≤ k
            continue
        end
        for j in setdiff(1:m, C)
            C_new = vcat(C, j)
            colp = sortperm(C_new)
            i_choices = findall(vec(all(indices_presolved[:,C_new], dims=2)))
            if length(i_choices) ≥ k
                continue
            end
            h = length(i_choices)
            remaining_rows = setdiff(selected_rows, i_choices)
            R_range = [
                sort(collect(x))
                for x in Iterators.product(
                    [[x] for x in i_choices]..., # h
                    [remaining_rows[ind:ind] for ind in 1:(k-h)]..., # k-h
                    remaining_rows[k-h+1:end],
                )
            ]
            for R in R_range
                if R in constrained_vars_C[j]
                    continue
                end
                push!(constrained_vars_C[j], R)
                push!(linear_coupling_constraints_indexes, (
                    R, 
                    C_new[colp],
                    "col",
                    j,
                ))
            end
        end
    end

    for R in combinations(1:n, k)
        selected_cols = findall(vec(all(indices_presolved[R,:], dims=1)))
        if length(selected_cols) ≤ k
            continue
        end
        for i in setdiff(1:n, R)
            R_new = vcat(R, i)
            rowp = sortperm(R_new)
            j_choices = findall(vec(all(indices_presolved[R_new,:], dims=1)))
            if length(j_choices) ≥ k
                continue
            end
            h = length(j_choices)
            remaining_cols = setdiff(selected_cols, j_choices)
            C_range = [
                sort(collect(x))
                for x in Iterators.product(
                    [[x] for x in j_choices]...,
                    [remaining_cols[ind:ind] for ind in 1:k-h]...,
                    remaining_cols[k-h+1:end],
                )
            ]
            for C in C_range
                if C in constrained_vars_R[i]
                    continue
                end
                push!(constrained_vars_R[i], C)
                push!(linear_coupling_constraints_indexes, (
                    R_new[rowp],
                    C,
                    "row",
                    i,
                ))
            end
        end
    end

    return linear_coupling_constraints_indexes
end

function generate_rank1_basis_pursuit_Shor_constraints_indexes(
    indices_presolved::BitMatrix, # for Shor indexes, in the noiseless case
    num_entries_present::Int = 1 # Only makes sense to have num_entries_present == 1
)
    # Used in basis pursuit: Shor LMIs in rank-1
    (n, m) = size(indices_presolved)

    rowp = sortperm([indices_presolved[i,:] for i in 1:n], rev = true)
    colp = sortperm([indices_presolved[rowp,j] for j in 1:m], rev = true)
    rowind = unique([findfirst(indices_presolved[rowp, j]) for j in colp])
    colind = unique([findfirst(indices_presolved[i, colp]) for i in rowp])

    Shor_constraints_indexes = NTuple{4, Int}[]
    # One entry present
    if num_entries_present == 1
        for block_i1 in 1:(length(rowind)-1), block_i2 in (block_i1+1):length(rowind)
            i1_start = rowind[block_i1]
            i1_end = rowind[block_i1+1]-1
            j_i1_start = colind[block_i1]
            j_i1_end = colind[block_i1+1]-1
            i2_start = rowind[block_i2]
            i2_end = (
                block_i2 == length(rowind) 
                ? n
                : rowind[block_i2+1]-1 
            )
            j_i2_start = colind[block_i2]
            j_i2_end = (
                block_i2 == length(rowind)
                ? m
                : colind[block_i2+1]-1
            )        
            for block_j in 1:length(colind)
                # Ensures that block_j is not equal to either block_i1 or block_i2
                if block_j in [block_i1, block_i2]
                    continue
                end
                j1_start = colind[block_j]
                j1_end = (
                    block_j == length(colind) 
                    ? m
                    : colind[block_j+1]-1
                )
                append!(
                    Shor_constraints_indexes,
                    [
                        (sort([rowp[i1], rowp[i2]])..., sort([colp[j1], colp[j2]])...)
                        for i1 in i1_start:i1_end, i2 in i2_start:i2_end, j1 in j1_start:j1_end, j2 in vcat(j_i1_start:j_i1_end, j_i2_start:j_i2_end)
                    ]
                )
            end
        end
    end
    return Shor_constraints_indexes
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
            for (i1, i2) in combinations(1:n, 2)
                for j1 in 1:(m-1)
                    if indices[i1,j1] == 1 && indices[i2,j1] == 1
                        # only looks for j2 after j1, 
                        # where indices[i1,j2] == indices[i2,j2] == 1
                        for j2 in findall(indices[i1,(j1+1):end] .& indices[i2,(j1+1):end])
                            push!(Shor_constraints_indexes, (i1, i2, j1, j1+j2))
                        end
                    end
                end
            end
        elseif num_entries_present == 3
            for (i1, i2) in combinations(1:n, 2)
                for j1 in 1:m
                    if indices[i1,j1] == 1 && indices[i2,j1] == 1
                        # Looks for j2 in 1:n, 
                        # where indices[i1,j2] + indices[i2,j2] == 1
                        for j2 in findall(indices[i1,:] .⊻ indices[i2,:])
                            push!(Shor_constraints_indexes, (i1, i2, sort([j1, j2])...))
                        end
                    end
                end
            end
        elseif num_entries_present == 2
            # (a): [1 0; 1 0] and [0 1; 0 1]
            for (i1, i2) in combinations(1:n, 2)
                for j1 in 1:m
                    if indices[i1,j1] == 1 && indices[i2,j1] == 1
                        for j2 in findall(.~(indices[i1,:] .| indices[i2,:]))
                            push!(Shor_constraints_indexes, (i1, i2, sort([j1, j2])...))
                        end
                    end
                end
            end
            # (b): all other cases
            for (i1, i2) in combinations(1:n, 2)
                for (j1, j2) in combinations(findall(indices[i1,:] .⊻ indices[i2,:]), 2)
                    push!(Shor_constraints_indexes, (i1, i2, j1, j2))
                end
            end
        elseif num_entries_present == 1
            for (i1, i2) in combinations(1:n, 2)
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
            for (i1, i2) in combinations(1:n, 2)
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

    # minors_indexes = vec(collect(
    #     (i1, i2, j1, j2)
    #     for (i1, i2) in combinations(1:n, 2), 
    #         (j1, j2) in combinations(1:m, 2)
    # ))
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