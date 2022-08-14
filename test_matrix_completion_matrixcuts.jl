include("matrix_completion.jl")
include("utils.jl")

using Plots
using StatsBase
using Suppressor
using CSV

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
    branching_type::String = "lexicographic",
    branch_point::String = "midpoint",
    node_selection::String = "breadthfirst",
    bestfirst_depthfirst_cutoff::Int = 10000,
    use_disjunctive_cuts::Bool = true,
    disjunctive_cuts_type::String = "linear",
    root_only::Bool = false,
    altmin_flag::Bool = true,
    use_max_steps::Bool = true,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
    update_step::Int = 1000,
    with_log::Bool = true,
)
    if use_disjunctive_cuts
        if !(disjunctive_cuts_type in ["linear", "semidefinite"])
            error("""
            Invalid input for disjunctive cuts type.
            Disjunctive cuts type must be either "linear" or "semidefinite";
            $disjunctive_cuts_type supplied instead.
            """)
        end
    else
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
        branching_type = branching_type,
        branch_point = branch_point,
        node_selection = node_selection,
        bestfirst_depthfirst_cutoff = bestfirst_depthfirst_cutoff,
        use_disjunctive_cuts = use_disjunctive_cuts,
        disjunctive_cuts_type = disjunctive_cuts_type,
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

# Experiment 1: Matrix cuts perform worse because later iterations are costly 
# also; great performance of alternating minimization heuristic
r_1a = test_branchandbound_frob_matrixcomp(
    1,5,5,12,0, 
    ;
    γ = 40.0, λ = 0.0,
    branching_region = "angular",
    use_disjunctive_cuts = false,
    time_limit = 60,
)
println("# explored nodes: $(r_1a[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_1a[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_1a[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_1a[3]["run_log"][end,:upper])")
println("Gap:              $(r_1a[3]["run_log"][end,:gap])")
println("Runtime:          $(r_1a[3]["run_details"]["time_taken"])")

r_1h = test_branchandbound_frob_matrixcomp(
    1,5,5,12,0, 
    ;
    γ = 40.0, λ = 0.0,
    branching_region = "hybrid",
    use_disjunctive_cuts = false,
    time_limit = 60,
);
println("# explored nodes: $(r_1h[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_1h[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_1h[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_1h[3]["run_log"][end,:upper])")
println("Gap:              $(r_1h[3]["run_log"][end,:gap])")
println("Runtime:          $(r_1h[3]["run_details"]["time_taken"])")

r_1b = test_branchandbound_frob_matrixcomp(
    1,5,5,12,0, 
    ;
    γ = 40.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = false,
    time_limit = 60,
);
println("# explored nodes: $(r_1b[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_1b[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_1b[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_1b[3]["run_log"][end,:upper])")
println("Gap:              $(r_1b[3]["run_log"][end,:gap])")
println("Runtime:          $(r_1b[3]["run_details"]["time_taken"])")

r_1m = test_branchandbound_frob_matrixcomp(
    1,5,5,12,0, 
    ;
    γ = 40.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 60,
)
println("# explored nodes: $(r_1m[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_1m[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_1m[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_1m[3]["run_log"][end,:upper])")
println("Gap:              $(r_1m[3]["run_log"][end,:gap])")
println("Runtime:          $(r_1m[3]["run_details"]["time_taken"])")

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 1: (k, m, n, i, seed, γ) = (1,5,5,12,0,40)"
)
plot!(
    r_1a[3]["run_log"][!,:runtime],
    r_1a[3]["run_log"][!,:gap],
    label = "Angular branching",
    color = :orange
)
plot!(
    r_1h[3]["run_log"][!,:runtime],
    r_1h[3]["run_log"][!,:gap],
    label = "Hybrid branching", 
    color = :red
)
plot!(
    r_1b[3]["run_log"][!,:runtime],
    r_1b[3]["run_log"][!,:gap],
    label = "Box branching",
    color = :blue
)
plot!(
    r_1m[3]["run_log"][!,:runtime],
    r_1m[3]["run_log"][!,:gap],
    label = "Matrix cuts",
    color = :green
)

# Experiment 2: Vanilla box branching does not improve objective, but matrix cuts do -- still worse than angular branching though
r_2a = test_branchandbound_frob_matrixcomp(
    1,5,5,12,9, 
    ;
    γ = 10.0, λ = 0.0,
    branching_region = "angular",
    use_disjunctive_cuts = false,
    time_limit = 60,
);
println("# explored nodes: $(r_2a[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_2a[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_2a[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_2a[3]["run_log"][end,:upper])")
println("Gap:              $(r_2a[3]["run_log"][end,:gap])")
println("Runtime:          $(r_2a[3]["run_details"]["time_taken"])")

r_2h = test_branchandbound_frob_matrixcomp(
    1,5,5,12,9,
    ;
    γ = 10.0, λ = 0.0,
    branching_region = "hybrid",
    use_disjunctive_cuts = false,
    time_limit = 60,
);
println("# explored nodes: $(r_2h[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_2h[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_2h[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_2h[3]["run_log"][end,:upper])")
println("Gap:              $(r_2h[3]["run_log"][end,:gap])")
println("Runtime:          $(r_2h[3]["run_details"]["time_taken"])")

r_2b = test_branchandbound_frob_matrixcomp(
    1,5,5,12,9, 
    ;
    γ = 10.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = false,
    time_limit = 60,
);
println("# explored nodes: $(r_2b[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_2b[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_2b[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_2b[3]["run_log"][end,:upper])")
println("Gap:              $(r_2b[3]["run_log"][end,:gap])")
println("Runtime:          $(r_2b[3]["run_details"]["time_taken"])")

r_2m = test_branchandbound_frob_matrixcomp(
    1,5,5,12,9, 
    ;
    γ = 10.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 60,
);
println("# explored nodes: $(r_2m[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_2m[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_2m[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_2m[3]["run_log"][end,:upper])")
println("Gap:              $(r_2m[3]["run_log"][end,:gap])")
println("Runtime:          $(r_2m[3]["run_details"]["time_taken"])")

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 2: (k, m, n, i, seed, γ) = (1,5,5,12,9,10)"
)
plot!(
    r_2a[3]["run_log"][!,:runtime],
    r_2a[3]["run_log"][!,:gap],
    label = "Angular branching",
    color = :orange
)
plot!(
    r_2h[3]["run_log"][!,:runtime],
    r_2h[3]["run_log"][!,:gap],
    label = "Hybrid branching", 
    color = :red
)
plot!(
    r_2b[3]["run_log"][!,:runtime],
    r_2b[3]["run_log"][!,:gap],
    label = "Box branching",
    color = :blue
)
plot!(
    r_2m[3]["run_log"][!,:runtime],
    r_2m[3]["run_log"][!,:gap],
    label = "Matrix cuts",
    color = :green
)

# Experiment 3: Angular fails to make progress
r_3a = test_branchandbound_frob_matrixcomp(
    1,8,8,24,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "angular",
    use_disjunctive_cuts = false,
    time_limit = 60,
);

r_3h = test_branchandbound_frob_matrixcomp(
    1,8,8,24,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "hybrid",
    use_disjunctive_cuts = false,
    time_limit = 60,
);

r_3b = test_branchandbound_frob_matrixcomp(
    1,8,8,24,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = false,
    time_limit = 60,
);

r_3m = test_branchandbound_frob_matrixcomp(
    1,8,8,24,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 60,
);

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 3: (k, m, n, i, seed, γ) = (1,8,8,24,0,20)"
)
plot!(
    r_3a[3]["run_log"][!,:runtime],
    r_3a[3]["run_log"][!,:gap],
    label = "Angular branching",
    color = :orange
)
# plot!(
#     r_3h[3]["run_log"][!,:runtime],
#     r_3h[3]["run_log"][!,:gap],
#     label = "Hybrid branching", 
#     color = :red
# )
plot!(
    r_3b[3]["run_log"][!,:runtime],
    r_3b[3]["run_log"][!,:gap],
    label = "Box branching",
    color = :blue
)
plot!(
    r_3m[3]["run_log"][!,:runtime],
    r_3m[3]["run_log"][!,:gap],
    label = "Matrix cuts",
    color = :green
)


r_4a = test_branchandbound_frob_matrixcomp(
    1,7,7,21,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "angular",
    use_disjunctive_cuts = false,
    time_limit = 60,
);
r_4h = test_branchandbound_frob_matrixcomp(
    1,7,7,21,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "hybrid",
    use_disjunctive_cuts = false,
    time_limit = 60,
);
r_4b = test_branchandbound_frob_matrixcomp(
    1,7,7,21,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = false,
    time_limit = 60,
);
r_4m = test_branchandbound_frob_matrixcomp(
    1,7,7,21,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 60,
);

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 4: (k, m, n, i, seed, γ) = (1,7,7,21,0,20)"
)
plot!(
    r_4a[3]["run_log"][!,:runtime],
    r_4a[3]["run_log"][!,:gap],
    label = "Angular branching",
    color = :orange
)
plot!(
    r_4h[3]["run_log"][!,:runtime],
    r_4h[3]["run_log"][!,:gap],
    label = "Hybrid branching", 
    color = :red
)
plot!(
    r_4b[3]["run_log"][!,:runtime],
    r_4b[3]["run_log"][!,:gap],
    label = "Box branching",
    color = :blue
)
plot!(
    r_4m[3]["run_log"][!,:runtime],
    r_4m[3]["run_log"][!,:gap],
    label = "Matrix cuts",
    color = :green
)


r = test_branchandbound_frob_matrixcomp(
    1,10,10,30,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 300,
);
solve_times = r[3]["run_details"]["dict_solve_times_relaxation"]

maximum(solve_times[!,:depth])
plot(
    1:maximum(solve_times[!,:depth]),
    [
        StatsBase.mean(
            filter(
                [:depth, :solve_time] => (
                    (x , y) -> (
                        x == k 
                        && y > 0.0
                    )
                ), 
                solve_times
            )[!, :solve_time]
        )
        for k in 1:maximum(solve_times[!,:depth])
    ]
)

filter(
    [:depth, :solve_time] => (
        (x , y) -> (
            x == 5 
            && y > 0.0
        )
    ), 
    solve_times
)[!, :solve_time]


outfile = "logs/runtimes/runtimes_v2.csv"
headers_table = DataFrame(
    k = Int[],
    m = Int[],
    n = Int[],
    p = Float64[],
    num_indices = Int[],
    noise = Float64[],
    γ = Float64[],
    λ = Float64[],
    node_selection = String[],
    use_disjunctive_cuts = Bool[],
    optimality_gap = Float64[],
    use_max_steps = Bool[],
    max_steps = Int[],
    time_limit = Int[],
    altmin_probability = Float64[],
    seed = Int[],
    # results: time
    time_taken = Float64[],
    solve_time_altmin = Float64[],
    solve_time_altmin_root_node = Float64[],
    solve_time_relaxation_feasibility = Float64[],
    solve_time_relaxation = Float64[],
    average_solve_time_relaxation = Float64[],
    average_solve_time_altmin = Float64[],
    # results: nodes
    nodes_explored = Int[],
    nodes_total = Int[],
    nodes_dominated = Int[],
    nodes_relax_infeasible = Int[],
    nodes_relax_feasible = Int[],
    nodes_relax_feasible_pruned = Int[],
    nodes_master_feasible = Int[],
    nodes_master_feasible_improvement = Int[],
    nodes_relax_feasible_split = Int[],
    nodes_relax_feasible_split_altmin = Int[],
    nodes_relax_feasible_split_altmin_improvement = Int[],
    # results: bound gap
    lower_bound_root_node = Float64[],
    upper_bound_root_node = Float64[],
    relative_gap_root_node = Float64[],
    lower_bound = Float64[],
    upper_bound = Float64[],
    relative_gap = Float64[],
    # results: MSE
    MSE_in_initial = Float64[],
    MSE_out_initial = Float64[],
    MSE_all_initial = Float64[],
    MSE_in = Float64[],
    MSE_out = Float64[],
    MSE_all = Float64[],
)
CSV.write(outfile, headers_table)

### edit these
k = 1
λ = 0.0
time_limit = 1000
n_values = [10, 15, 20, 25, 30, 40, 50]
γ_values = [5.0, 10.0, 20.0] # [5.0, 10.0, 20.0, 40.0, 80.0]
p_values = [0.3] # [0.3, 0.1]
noise_values = [0.001, 0.01, 0.1]
node_selection_values = ["breadthfirst", "bestfirst_depthfirst"] # ["breadthfirst", "bestfirst_depthfirst"]
seeds = collect(1:20)
###


params = []
runtimes_df = DataFrame(CSV.File(outfile))
sort!(runtimes_df, :time_limit, rev = true)
for (n, γ, p, noise, node_selection, seed) in Iterators.product(
    n_values, γ_values, p_values, noise_values,
    node_selection_values,
    seeds,
)
    prev_record = filter(
        row -> (
            row.n == n 
            && row.γ == γ
            && row.p == p
            && row.noise == noise
            && row.node_selection == node_selection
            && row.seed == seed
        ), 
        runtimes_df,
    )
    if nrow(prev_record) == 0
        push!(params, (n, γ, p, noise, node_selection, seed))
    elseif prev_record[1,:time_taken] > prev_record[1,:time_limit]
        push!(params, (n, γ, p, noise, node_selection, seed))
    end
end

for (n, γ, p, noise, node_selection, seed) in params
    print("""

    n: $n   γ: $γ   p: $p   noise: $noise
    node_selection: $node_selection
    """)

    num_indices = Int(round(n*n*p))
    r = @suppress test_branchandbound_frob_matrixcomp(
        k, n, n, num_indices, seed,
        ;
        γ = γ, λ = λ,
        noise = true, ϵ = noise,
        branching_region = "box", 
        node_selection = node_selection,
        use_disjunctive_cuts = true,
        time_limit = time_limit,
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
            use_disjunctive_cuts = r[3]["run_details"]["use_disjunctive_cuts"],
            optimality_gap = r[3]["run_details"]["optimality_gap"],
            use_max_steps = r[3]["run_details"]["use_max_steps"],
            max_steps = r[3]["run_details"]["max_steps"],
            time_limit = time_limit,
            altmin_probability = r[3]["run_details"]["altmin_probability"],
            seed = seed,
            # results: time
            time_taken = r[3]["run_details"]["time_taken"],
            solve_time_altmin = r[3]["run_details"]["solve_time_altmin"],
            solve_time_altmin_root_node = filter(row -> (row.node_id == 0), r[3]["run_details"]["dict_solve_times_altmin"])[1,:solve_time],
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
    CSV.write(outfile, DataFrame(records), append=true)
    message = Printf.@sprintf(
        "Run %2d:    %6.3f\n",
        seed, r[3]["run_details"]["time_taken"]
    )
    print(stdout, message)
end 

# node selection strategy tests

r_n1 = test_branchandbound_frob_matrixcomp(
    1,10,10,30,1, 
    ;
    γ = 10.0, λ = 0.0,
    node_selection = "breadthfirst",
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 100,
    max_steps = 100000,
)
println("# explored nodes: $(r_n1[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_n1[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_n1[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_n1[3]["run_log"][end,:upper])")
println("Gap:              $(r_n1[3]["run_log"][end,:gap])")
println("Runtime:          $(r_n1[3]["run_details"]["time_taken"])")


r_n1bd_10000 = test_branchandbound_frob_matrixcomp(
    1,10,10,30,1, 
    ;
    γ = 10.0, λ = 0.0,
    node_selection = "bestfirst_depthfirst",
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 100,
    max_steps = 100000,
)
println("# explored nodes: $(r_n1bd_10000[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_n1bd_10000[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_n1bd_10000[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_n1bd_10000[3]["run_log"][end,:upper])")
println("Gap:              $(r_n1bd_10000[3]["run_log"][end,:gap])")
println("Runtime:          $(r_n1bd_10000[3]["run_details"]["time_taken"])")

r_n1bd_1000 = test_branchandbound_frob_matrixcomp(
    1,10,10,30,1, 
    ;
    γ = 10.0, λ = 0.0,
    node_selection = "bestfirst_depthfirst",
    bestfirst_depthfirst_cutoff = 1000,
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 100,
    max_steps = 100000,
)
println("# explored nodes: $(r_n1bd_1000[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_n1bd_1000[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_n1bd_1000[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_n1bd_1000[3]["run_log"][end,:upper])")
println("Gap:              $(r_n1bd_1000[3]["run_log"][end,:gap])")
println("Runtime:          $(r_n1bd_1000[3]["run_details"]["time_taken"])")

r_n1bd_100 = test_branchandbound_frob_matrixcomp(
    1,10,10,30,1, 
    ;
    γ = 10.0, λ = 0.0,
    node_selection = "bestfirst_depthfirst",
    bestfirst_depthfirst_cutoff = 100,
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 100,
    max_steps = 100000,
)
println("# explored nodes: $(r_n1bd_100[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_n1bd_100[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_n1bd_100[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_n1bd_100[3]["run_log"][end,:upper])")
println("Gap:              $(r_n1bd_100[3]["run_log"][end,:gap])")
println("Runtime:          $(r_n1bd_100[3]["run_details"]["time_taken"])")

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment: (k, m, n, i, seed, γ) = (1,10,10,30,1,10.0)"
)
plot!(
    r_n1[3]["run_log"][!,:runtime],
    r_n1[3]["run_log"][!,:gap],
    label = "breadthfirst",
    color = :black
)
plot!(
    r_n1bd_10000[3]["run_log"][!,:runtime],
    r_n1bd_10000[3]["run_log"][!,:gap],
    label = "bestfirst-depthfirst 10000", 
    color = :green
)
plot!(
    r_n1bd_1000[3]["run_log"][!,:runtime],
    r_n1bd_1000[3]["run_log"][!,:gap],
    label = "bestfirst-depthfirst 1000",
    color = :blue
)
plot!(
    r_n1bd_100[3]["run_log"][!,:runtime],
    r_n1bd_100[3]["run_log"][!,:gap],
    label = "bestfirst-depthfirst 100",
    color = :purple
)


# Rank-2 tests

test_branchandbound_frob_matrixcomp(
    2,12,12,120,1, 
    ;
    γ = 10.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = false,
    time_limit = 300,
)

# GOOD: alternating minimization can indeed reach optimality
test_branchandbound_frob_matrixcomp(
    2,15,15,150,1, 
    ;
    γ = 5.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 60,
)

# BAD: no improvement after alternating minimization heuristic
test_branchandbound_frob_matrixcomp(
    2,15,15,150,1, 
    ;
    γ = 10.0, λ = 0.0,
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 60,
)

# BAD: no improvement after alternating minimization heuristic
test_branchandbound_frob_matrixcomp(
    2,15,15,150,1, 
    ;
    γ = 10.0, λ = 0.0,
    node_selection = "bestfirst_depthfirst",
    branching_region = "box",
    use_disjunctive_cuts = true,
    time_limit = 60,
)


# Disjunctive cut type tests
test_branchandbound_frob_matrixcomp(
    2,15,15,150,1, 
    ;
    use_disjunctive_cuts = true,
    γ = 5.0, λ = 0.0,
    disjunctive_cuts_type = "linear",
    node_selection = "breadthfirst",
    time_limit = 60,
) # Good: converges
test_branchandbound_frob_matrixcomp(
    2,15,15,150,1, 
    ;
    use_disjunctive_cuts = true,
    γ = 5.0, λ = 0.0,
    disjunctive_cuts_type = "semidefinite",
    node_selection = "breadthfirst",
    time_limit = 60,
) # Bad: explores entire tree, does not close gap, IF does not encounter alternating minimization
 

test_branchandbound_frob_matrixcomp(
    2,15,15,150,1, 
    ;
    use_disjunctive_cuts = true,
    γ = 10.0, λ = 0.0,
    disjunctive_cuts_type = "linear",
    node_selection = "breadthfirst",
    time_limit = 60,
) # Bad: does not converge after altmin
test_branchandbound_frob_matrixcomp(
    2,15,15,150,1, 
    ;
    use_disjunctive_cuts = true,
    γ = 10.0, λ = 0.0,
    disjunctive_cuts_type = "semidefinite",
    node_selection = "breadthfirst",
    time_limit = 60,
) # Bad: if altmin, best lower bound > altmin solution
