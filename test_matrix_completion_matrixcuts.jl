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
    disjunctive_cuts_type::Union{String, Nothing} = nothing,
    disjunctive_cuts_breakpoints::Union{String, Nothing} = nothing,
    presolve::Bool = false,
    root_only::Bool = false,
    altmin_flag::Bool = true,
    use_max_steps::Bool = true,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
    update_step::Int = 1000,
    with_log::Bool = true,
)
    if use_disjunctive_cuts
        if !(disjunctive_cuts_type in ["linear", "linear2", "linear3", "linear_all"])
            error("""
            Invalid input for disjunctive cuts type.
            Disjunctive cuts type must be either "linear" or "linear2" or "linear3" or "linear_all";
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
        noise,
        ;
        branching_region = branching_region,
        branching_type = branching_type,
        branch_point = branch_point,
        node_selection = node_selection,
        bestfirst_depthfirst_cutoff = bestfirst_depthfirst_cutoff,
        use_disjunctive_cuts = use_disjunctive_cuts,
        disjunctive_cuts_type = disjunctive_cuts_type,
        disjunctive_cuts_breakpoints = disjunctive_cuts_breakpoints,
        presolve = presolve,
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

# Alternative cuts for rank-1
r_5_1 = test_branchandbound_frob_matrixcomp(
    1,10,10,30,0
    ;
    γ = 9.0, λ = 0.0,
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    use_disjunctive_cuts = true,
    time_limit = 60,
);
r_5_2 = test_branchandbound_frob_matrixcomp(
    1,10,10,30,0
    ;
    γ = 9.0, λ = 0.0,
    disjunctive_cuts_type = "linear2",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    use_disjunctive_cuts = true,
    time_limit = 60,
); # Benefit of using linear2: faster convergence
plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 5: (k, m, n, i, seed, γ) = (1,10,10,30,0,9)"
)
plot!(
    r_5_1[3]["run_log"][!,:runtime],
    r_5_1[3]["run_log"][!,:gap],
    label = "Linear (1 breakpoint)",
    color = :orange
)
plot!(
    r_5_2[3]["run_log"][!,:runtime],
    r_5_2[3]["run_log"][!,:gap],
    label = "Linear (2 breakpoints)", 
    color = :red
)



# Alternative cuts for rank-2
r_6_11b = test_branchandbound_frob_matrixcomp(
    2, 15, 15, 150, 1,
    ;
    γ = 8.0, λ = 0.0,
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    node_selection = "breadthfirst",
    use_disjunctive_cuts = true,
    time_limit = 60,
);
println("# explored nodes: $(r_6_11b[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_6_11b[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_6_11b[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_6_11b[3]["run_log"][end,:upper])")
println("Gap:              $(r_6_11b[3]["run_log"][end,:gap])")
println("Runtime:          $(r_6_11b[3]["run_details"]["time_taken"])")
r_6_12b = test_branchandbound_frob_matrixcomp(
    2, 15, 15, 150, 1,
    ;
    γ = 8.0, λ = 0.0,
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    node_selection = "breadthfirst",
    use_disjunctive_cuts = true,
    time_limit = 60,
);
println("# explored nodes: $(r_6_12b[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_6_12b[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_6_12b[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_6_12b[3]["run_log"][end,:upper])")
println("Gap:              $(r_6_12b[3]["run_log"][end,:gap])")
println("Runtime:          $(r_6_12b[3]["run_details"]["time_taken"])") # Bad: does not help convergence


r_6_21b = test_branchandbound_frob_matrixcomp(
    2, 15, 15, 150, 1,
    ;
    γ = 8.0, λ = 0.0,
    disjunctive_cuts_type = "linear2",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    node_selection = "breadthfirst",
    use_disjunctive_cuts = true,
    time_limit = 60,
)
println("# explored nodes: $(r_6_11b[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_6_11b[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_6_11b[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_6_11b[3]["run_log"][end,:upper])")
println("Gap:              $(r_6_11b[3]["run_log"][end,:gap])")
println("Runtime:          $(r_6_11b[3]["run_details"]["time_taken"])")
r_6_22b = test_branchandbound_frob_matrixcomp(
    2, 15, 15, 150, 1,
    ;
    γ = 8.0, λ = 0.0,
    disjunctive_cuts_type = "linear2",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    node_selection = "breadthfirst",
    use_disjunctive_cuts = true,
    time_limit = 60,
);
println("# explored nodes: $(r_6_12b[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_6_12b[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_6_12b[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_6_12b[3]["run_log"][end,:upper])")
println("Gap:              $(r_6_12b[3]["run_log"][end,:gap])")
println("Runtime:          $(r_6_12b[3]["run_details"]["time_taken"])") # Bad: does not help convergence

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
    disjunctive_cuts_type = "linear2",
    node_selection = "breadthfirst",
    time_limit = 60,
) # Good: converges

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
    disjunctive_cuts_type = "linear2",
    node_selection = "breadthfirst",
    time_limit = 60,
) # Bad: does not converge after altmin


# Joint linearization tests
r = test_branchandbound_frob_matrixcomp(
    1, 15, 15, 45, 0,
    ;
    use_disjunctive_cuts = true,
    γ = 40.0, λ = 0.0,
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    time_limit = 60,
)

r = test_branchandbound_frob_matrixcomp(
    1, 15, 15, 45, 0,
    ;
    use_disjunctive_cuts = true,
    γ = 40.0, λ = 0.0,
    disjunctive_cuts_type = "linear2",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    time_limit = 60,
)

test_branchandbound_frob_matrixcomp(
    1, 15, 15, 45, 0,
    ;
    use_disjunctive_cuts = true,
    γ = 40.0, λ = 0.0,
    disjunctive_cuts_type = "linear_all",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    time_limit = 60,
)

r = test_branchandbound_frob_matrixcomp(
    2, 15, 15, 125, 0,
    ;
    use_disjunctive_cuts = true,
    γ = 40.0, λ = 0.0,
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    time_limit = 60,
)

r = test_branchandbound_frob_matrixcomp(
    2, 15, 15, 125, 0,
    ;
    use_disjunctive_cuts = true,
    γ = 40.0, λ = 0.0,
    disjunctive_cuts_type = "linear2",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    time_limit = 60,
)

r = test_branchandbound_frob_matrixcomp(
    2, 15, 15, 125, 0,
    ;
    use_disjunctive_cuts = true,
    γ = 40.0, λ = 0.0,
    disjunctive_cuts_type = "linear_all",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    time_limit = 60,
)