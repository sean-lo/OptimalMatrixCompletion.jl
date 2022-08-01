using CSV
using DataFrames
using StatsBase

include("matrix_completion.jl")
include("utils.jl")

function test_branchandbound_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 0.0,
    noise::Bool = false,
    ϵ::Float64 = 0.01,
    branching_region::String = "angular",
    node_selection::String = "breadthfirst",
    bestfirst_depthfirst_cutoff::Int = 10000,
    use_matrix_cuts::Bool = true,
    root_only::Bool = false,
    altmin_flag::Bool = true,
    use_max_steps::Bool = false,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
    update_step::Int = 1000,
    with_log::Bool = true,
)
    if !(branching_region in ["box", "angular", "polyhedral", "hybrid"])
        error("""
        Invalid input for branching region.
        Branching region must be either "box" or "angular" or "polyhedral" or "hybrid"; $branching_region supplied instead.
        """)
    end
    if !(node_selection in ["breadthfirst", "bestfirst", "depthfirst", "bestfirst_depthfirst"])
        error("""
        Invalid input for node selection.
        Node selection must be either "breadthfirst" or "bestfirst" or "depthfirst" or "bestfirst_depthfirst"; $node_selection supplied instead.
        """)
    end
    (A, indices) = generate_matrixcomp_data(
        k, m, n, n_indices, seed; 
        noise = noise, ϵ = ϵ,
    )

    log_time = Dates.now()
    solution, printlist, instance = branchandbound_frob_matrixcomp(
        k,
        A,
        indices,
        γ,
        λ,
        ;
        branching_region = branching_region,
        node_selection = node_selection,
        bestfirst_depthfirst_cutoff = bestfirst_depthfirst_cutoff,
        use_matrix_cuts = use_matrix_cuts,
        root_only = root_only,
        altmin_flag = altmin_flag,
        use_max_steps = use_max_steps,
        max_steps = max_steps,
        time_limit = time_limit,
        update_step = update_step,
    )

    if with_log
        time_string = Dates.format(log_time, "yyyymmdd_HHMMSS")
        outfile = "logs/" * time_string * ".txt"
        open(outfile, "a+") do f
            for note in printlist
                print(f, note)
            end
        end
    end

    return solution, printlist, instance
end


r = test_branchandbound_frob_matrixcomp(
    1,5,5,12,0, 
    ;
    γ = 40.0, λ = 0.0,
    branching_region = "angular",
    use_matrix_cuts = false,
    time_limit = 60,
    with_log = false,
)
println("Compilation complete.")

### 
k = 1
λ = 0.0
###

args_df = DataFrame(CSV.File("args.csv"))

# Multi-job version
task_index = parse(Int, ARGS[1])
n_runs = parse(Int, ARGS[2])
time_limit = parse(Int, ARGS[3])

for ind in 1:n_runs
    row_index = (task_index-1) * n_runs + ind
    if row_index > size(args_df, 1)
        continue
    end 
    n = args_df[row_index, :n]
    p = args_df[row_index, :p]
    γ_num = args_df[row_index, :γ_num]
    noise = args_df[row_index, :noise]
    node_selection = String(args_df[row_index, :node_selection])
    seed = args_df[row_index, :seed]

    γ = γ_num / p
    print("""

    n: $n   γ: $γ   p: $p   
    noise: $noise
    node_selection: $node_selection
    seed: $seed
    """)
    num_indices = Int(round(n*n*p))
    # first, check if number of indices are sufficient
    if !((n + n) * k ≤ num_indices ≤ n * n)
        # num_indices outside allowable range; exit quietly
        continue
    end
    local r = @suppress test_branchandbound_frob_matrixcomp(
        k, n, n, num_indices, seed,
        ;
        γ = γ, λ = λ,
        noise = true, ϵ = noise,
        branching_region = "box", 
        node_selection = node_selection,
        use_matrix_cuts = true,
        time_limit = time_limit,
        with_log = false,
    )
    records = [
        (
            k = k,
            m = n,
            n = n,
            p = p,
            num_indices = num_indices,
            noise = noise,
            γ = γ,
            λ = λ,
            node_selection = node_selection,
            use_matrix_cuts = r[3]["run_details"]["use_matrix_cuts"],
            optimality_gap = r[3]["run_details"]["optimality_gap"],
            use_max_steps = r[3]["run_details"]["use_max_steps"],
            max_steps = r[3]["run_details"]["max_steps"],
            time_limit = time_limit,
            altmin_probability = r[3]["run_details"]["altmin_probability"],
            seed = seed,
            # results: time
            time_taken = r[3]["run_details"]["time_taken"],
            solve_time_altmin = r[3]["run_details"]["solve_time_altmin"],
            solve_time_altmin_root_node = filter(
                row -> (row.node_id == 0), 
                r[3]["run_details"]["dict_solve_times_altmin"]
            )[1,:solve_time],
            solve_time_relaxation_feasibility = r[3]["run_details"]["solve_time_relaxation_feasibility"],
            solve_time_relaxation = r[3]["run_details"]["solve_time_relaxation"],
            average_solve_time_relaxation = mean(
                r[3]["run_details"]["dict_solve_times_relaxation"][!,:solve_time]
            ),
            average_solve_time_altmin = mean(
                r[3]["run_details"]["dict_solve_times_altmin"][!,:solve_time]
            ),
            # results: nodes
            nodes_explored = r[3]["run_details"]["nodes_explored"],
            nodes_total = r[3]["run_details"]["nodes_total"],
            nodes_dominated = r[3]["run_details"]["nodes_dominated"],
            nodes_relax_infeasible = r[3]["run_details"]["nodes_relax_infeasible"],
            nodes_relax_feasible = r[3]["run_details"]["nodes_relax_feasible"],
            nodes_relax_feasible_pruned = r[3]["run_details"]["nodes_relax_feasible_pruned"],
            nodes_master_feasible = r[3]["run_details"]["nodes_master_feasible"],
            nodes_master_feasible_improvement = r[3]["run_details"]["nodes_master_feasible_improvement"],
            nodes_relax_feasible_split = r[3]["run_details"]["nodes_relax_feasible_split"],
            nodes_relax_feasible_split_altmin = r[3]["run_details"]["nodes_relax_feasible_split_altmin"],
            nodes_relax_feasible_split_altmin_improvement = r[3]["run_details"]["nodes_relax_feasible_split_altmin_improvement"],
            # results: bound gap
            lower_bound_root_node = r[3]["run_log"][1,:lower],
            upper_bound_root_node = r[3]["run_log"][1,:upper],
            relative_gap_root_node = r[3]["run_log"][1,:gap],
            lower_bound = r[3]["run_log"][end,:lower],
            upper_bound = r[3]["run_log"][end,:upper],
            relative_gap = r[3]["run_log"][end,:gap],
            # results: MSE
            MSE_in_initial = r[1]["MSE_in_initial"],
            MSE_out_initial = r[1]["MSE_out_initial"],
            MSE_all_initial = r[1]["MSE_all_initial"],
            MSE_in = r[1]["MSE_in"],
            MSE_out = r[1]["MSE_out"],
            MSE_all = r[1]["MSE_all"],
        )
    ]
    CSV.write("records/$(task_index)_$(ind).csv", DataFrame(records))
end 
