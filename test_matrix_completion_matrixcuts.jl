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
    use_matrix_cuts = false,
    time_limit = 60,
);
println("# explored nodes: $(r_1a[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_1a[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_1a[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_1a[3]["run_log"][end,:upper])")
println("Gap:              $(r_1a[3]["run_log"][end,:gap])")
println("Runtime:          $(r_1a[3]["run_details"]["time_taken"])")

r_1b = test_branchandbound_frob_matrixcomp(
    1,5,5,12,0, 
    ;
    γ = 40.0, λ = 0.0,
    branching_region = "box",
    use_matrix_cuts = false,
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
    use_matrix_cuts = true,
    time_limit = 60,
);
println("# explored nodes: $(r_1m[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_1m[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_1m[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_1m[3]["run_log"][end,:upper])")
println("Gap:              $(r_1m[3]["run_log"][end,:gap])")
println("Runtime:          $(r_1m[3]["run_details"]["time_taken"])")

# Experiment 2: Vanilla box branching does not improve objective, but matrix cuts do -- still worse than angular branching though
r_2a = test_branchandbound_frob_matrixcomp(
    1,5,5,12,9, 
    ;
    γ = 10.0, λ = 0.0,
    branching_region = "angular",
    use_matrix_cuts = false,
    time_limit = 60,
);
println("# explored nodes: $(r_2a[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_2a[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_2a[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_2a[3]["run_log"][end,:upper])")
println("Gap:              $(r_2a[3]["run_log"][end,:gap])")
println("Runtime:          $(r_2a[3]["run_details"]["time_taken"])")

r_2b = test_branchandbound_frob_matrixcomp(
    1,5,5,12,9, 
    ;
    γ = 10.0, λ = 0.0,
    branching_region = "box",
    use_matrix_cuts = false,
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
    use_matrix_cuts = true,
    time_limit = 60,
);
println("# explored nodes: $(r_2m[3]["run_details"]["nodes_explored"])")
println("# total nodes:    $(r_2m[3]["run_details"]["nodes_total"])")
println("Lower bound:      $(r_2m[3]["run_log"][end,:lower])")
println("Upper bound:      $(r_2m[3]["run_log"][end,:upper])")
println("Gap:              $(r_2m[3]["run_log"][end,:gap])")
println("Runtime:          $(r_2m[3]["run_details"]["time_taken"])")


test_branchandbound_frob_matrixcomp(
    1,8,8,24,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "angular",
    use_matrix_cuts = false,
    time_limit = 60,
)

test_branchandbound_frob_matrixcomp(
    1,8,8,24,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "box",
    use_matrix_cuts = false,
    time_limit = 60,
)

test_branchandbound_frob_matrixcomp(
    1,8,8,24,0
    ;
    γ = 20.0, λ = 0.0,
    branching_region = "box",
    use_matrix_cuts = true,
    time_limit = 60,
)


test_branchandbound_frob_matrixcomp(
    2,12,12,54,1, 
    ;
    γ = 50.0, λ = 0.0,
    branching_region = "box",
    use_matrix_cuts = false,
    time_limit = 60,
)



test_branchandbound_frob_matrixcomp(
    2,12,12,54,1, 
    ;
    γ = 50.0, λ = 0.0,
    branching_region = "box",
    use_matrix_cuts = false,
    time_limit = 60,
)

test_branchandbound_frob_matrixcomp(
    2,12,12,54,1, 
    ;
    γ = 50.0, λ = 0.0,
    branching_region = "box",
    use_matrix_cuts = true,
    time_limit = 60,
)

