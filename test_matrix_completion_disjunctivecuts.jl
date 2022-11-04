include("matrix_completion.jl")
include("utils.jl")

using Plots
using StatsBase
using Suppressor
using CSV

function test_matrix_completion_disjunctivecuts(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ϵ::Float64, 
    γ::Float64,
    ;
    λ::Float64 = 0.0,
    node_selection::String = "breadthfirst",
    bestfirst_depthfirst_cutoff::Int = 10000,
    disjunctive_cuts_type::Union{String, Nothing} = nothing,
    disjunctive_cuts_breakpoints::Union{String, Nothing} = nothing,
    disjunctive_sorting::Bool = false,
    add_Shor_valid_inequalities::Bool = false,
    add_Shor_valid_inequalities_iterative::Bool = false,
    Shor_valid_inequalities_noisy_rank1_num_entries_present::Vector{Int} = [1,2,3,4],
    max_update_Shor_indices_probability::Float64 = 1.0, # TODO
    min_update_Shor_indices_probability::Float64 = 0.1, # TODO
    update_Shor_indices_probability_decay_rate::Float64 = 1.1, # TODO
    update_Shor_indices_n_minors::Int = 100,
    root_only::Bool = false,
    altmin_flag::Bool = true,
    max_altmin_probability::Float64 = 1.0,
    min_altmin_probability::Float64 = 0.005,
    altmin_probability_decay_rate::Float64 = 1.1,
    use_max_steps::Bool = true,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
    update_step::Int = 1000,
    with_log::Bool = true,
)
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
    if !(node_selection in ["breadthfirst", "bestfirst", "depthfirst", "bestfirst_depthfirst"])
        error("""
        Invalid input for node selection.
        Node selection must be either "breadthfirst" or "bestfirst" or "depthfirst" or "bestfirst_depthfirst"; $node_selection supplied instead.
        """)
    end
    (A, indices) = generate_matrixcomp_data(
        k, m, n, n_indices, seed; 
        noise = true, ϵ = ϵ,
    )

    log_time = Dates.now()
    r = branchandbound_frob_matrixcomp(
        k,
        A,
        indices,
        γ,
        λ,
        true, # noise
        ;
        node_selection = node_selection,
        bestfirst_depthfirst_cutoff = bestfirst_depthfirst_cutoff,
        use_disjunctive_cuts = true,
        disjunctive_cuts_type = disjunctive_cuts_type,
        disjunctive_cuts_breakpoints = disjunctive_cuts_breakpoints,
        disjunctive_sorting = disjunctive_sorting,
        presolve = false, # does not apply in noisy case
        add_Shor_valid_inequalities = add_Shor_valid_inequalities,
        add_Shor_valid_inequalities_iterative = add_Shor_valid_inequalities_iterative,
        Shor_valid_inequalities_noisy_rank1_num_entries_present = Shor_valid_inequalities_noisy_rank1_num_entries_present,
        max_update_Shor_indices_probability = max_update_Shor_indices_probability,
        min_update_Shor_indices_probability = min_update_Shor_indices_probability,
        update_Shor_indices_probability_decay_rate = update_Shor_indices_probability_decay_rate,
        update_Shor_indices_n_minors = update_Shor_indices_n_minors,
        root_only = root_only,
        altmin_flag = altmin_flag,
        use_max_steps = use_max_steps,
        max_steps = max_steps,
        time_limit = time_limit,
        update_step = update_step,
    )
    solution, printlist, instance = r[1], r[2], r[3]

    if with_log
        time_string = Dates.format(log_time, "yyyymmdd_HHMMSS")
        outfile = "logs/" * time_string * ".txt"
        open(outfile, "a+") do f
            for note in printlist
                print(f, note)
            end
        end
    end

    return r
end

# Experiment 1: number of eigenvectors
r_1_1 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_1_2 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)

runtimes_1_1 = []
gaps_1_1 = []
runtimes_1_2 = []
gaps_1_2 = []
for seed in 1:20
    r_1_1 = test_matrix_completion_disjunctivecuts(
        1, 10, 10, 30, seed, 0.01, 20.0;
        node_selection = "bestfirst",
        disjunctive_cuts_type = "linear",
        disjunctive_cuts_breakpoints = "smallest_1_eigvec",
        add_Shor_valid_inequalities = false,
        time_limit = 60,
    )
    push!(runtimes_1_1, r_1_1[3]["run_log"][!,:runtime])
    push!(gaps_1_1, r_1_1[3]["run_log"][!,:gap])

    r_1_2 = test_matrix_completion_disjunctivecuts(
        1, 10, 10, 30, seed, 0.01, 20.0;
        node_selection = "bestfirst",
        disjunctive_cuts_type = "linear",
        disjunctive_cuts_breakpoints = "smallest_2_eigvec",
        add_Shor_valid_inequalities = false,
        time_limit = 60,
    )
    push!(runtimes_1_2, r_1_2[3]["run_log"][!,:runtime])
    push!(gaps_1_2, r_1_2[3]["run_log"][!,:gap])
end

seed = 20
plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 1: (k, m, n, i, seed, γ) = (1,10,10,30,$seed,20)"
)
plot!(
    runtimes_1_1[seed],
    gaps_1_1[seed],
    label = "1 eigenvector",
    color = :orange
)
plot!(
    runtimes_1_2[seed],
    gaps_1_2[seed],
    label = "2 eigenvectors",
    color = :blue
)

# Experiment 2: number of breakpoints
r_2_1 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_2_2 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear2",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_2_3 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear3",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_2_a = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear_all",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)


plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 1: (k, m, n, i, seed, γ) = (1,10,10,30,0,20)"
)
plot!(
    r_2_1[3]["run_log"][!,:runtime],
    r_2_1[3]["run_log"][!,:gap],
    label = "2 pieces",
    color = :orange
)
plot!(
    r_2_2[3]["run_log"][!,:runtime],
    r_2_2[3]["run_log"][!,:gap],
    label = "3 pieces",
    color = :blue
)
plot!(
    r_2_3[3]["run_log"][!,:runtime],
    r_2_3[3]["run_log"][!,:gap],
    label = "4 pieces",
    color = :green
)
plot!(
    r_2_a[3]["run_log"][!,:runtime],
    r_2_a[3]["run_log"][!,:gap],
    label = "joint linearization",
    color = :red
)


# Experiment 2: number of breakpoints (rank-2)
r_3_1 = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 240,
)
r_3_2 = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear2",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 240,
)
r_3_3 = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear3",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 240,
)
r_3_a = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear_all",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 240,
)


plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 3: (k, m, n, i, seed, γ) = (2,10,10,60,0,20)"
)
plot!(
    r_3_1[3]["run_log"][!,:runtime],
    r_3_1[3]["run_log"][!,:gap],
    label = "2 pieces",
    color = :orange
)
plot!(
    r_3_2[3]["run_log"][!,:runtime],
    r_3_2[3]["run_log"][!,:gap],
    label = "3 pieces",
    color = :blue
)
plot!(
    r_3_3[3]["run_log"][!,:runtime],
    r_3_3[3]["run_log"][!,:gap],
    label = "4 pieces",
    color = :green
)
plot!(
    r_3_a[3]["run_log"][!,:runtime],
    r_3_a[3]["run_log"][!,:gap],
    label = "joint linearization",
    color = :red
)


# Experiment 4: majorization inequalities
r_4_1t = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    disjunctive_sorting = true,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
    use_max_steps = false,
)
r_4_1f = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
    use_max_steps = false,
)

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 4: (k, m, n, i, seed, γ) = (1,10,10,30,0,20)"
)
plot!(
    r_4_1t[3]["run_log"][!,:runtime],
    r_4_1t[3]["run_log"][!,:gap],
    label = "With majorization",
    color = :orange
)
plot!(
    r_4_1f[3]["run_log"][!,:runtime],
    r_4_1f[3]["run_log"][!,:gap],
    label = "Without majorization",
    color = :blue
)

r_4_2t_noaltmin = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = true,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
    use_max_steps = false,
    altmin_flag = false,
)


r_4_2t = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.001, 10.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = true,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
    use_max_steps = false,
)
r_4_2f = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.001, 10.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
    use_max_steps = false,
)

r_4_2t_noaltmin[3]["run_details"]["dict_solve_times_altmin"]
r_4_2t_noaltmin[3]["run_details"]["solve_time_altmin"]
r_4_2t_noaltmin[3]["run_details"]["solve_time_relaxation"]
r_4_2t_noaltmin[3]["run_details"]["time_taken"]
r_4_2t_noaltmin[3]["run_details"]["dict_num_interations_altmin"]
r_4_2t_noaltmin[3]["run_details"]["dict_solve_times_relaxation"]
show(r_4_2t_noaltmin[3]["run_details"]["dict_solve_times_relaxation"][:,:], allrows=true)

r_4_2t[3]["run_details"]["dict_solve_times_altmin"]
r_4_2t[3]["run_details"]["solve_time_altmin"]
r_4_2t[3]["run_details"]["solve_time_relaxation"]
r_4_2t[3]["run_details"]["time_taken"]
r_4_2t[3]["run_details"]["dict_num_interations_altmin"]
show(r_4_2t[3]["run_details"]["dict_solve_times_relaxation"][:,:], allrows=true)

r_4_2f[3]["run_details"]["dict_solve_times_altmin"]
r_4_2f[3]["run_details"]["solve_time_altmin"]
r_4_2f[3]["run_details"]["solve_time_relaxation"]
r_4_2f[3]["run_details"]["time_taken"]
r_4_2f[3]["run_details"]["dict_num_interations_altmin"]
r_4_2f[3]["run_details"]["dict_solve_times_relaxation"]

r_4_2_br_t = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.01, 20.0;
    node_selection = "breadthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    disjunctive_sorting = true,
    add_Shor_valid_inequalities = false,
    time_limit = 240,
    use_max_steps = false,
)
r_4_2_br_f = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.01, 20.0;
    node_selection = "breadthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 240,
    use_max_steps = false,
)

r_4_2_br_t[3]["run_details"]["dict_solve_times_altmin"]
r_4_2_br_t[3]["run_details"]["solve_time_altmin"]
r_4_2_br_t[3]["run_details"]["solve_time_relaxation"]
r_4_2_br_t[3]["run_details"]["time_taken"]
r_4_2_br_t[3]["run_details"]["dict_num_interations_altmin"]

r_4_2_br_f[3]["run_details"]["dict_solve_times_altmin"]
r_4_2_br_f[3]["run_details"]["solve_time_altmin"]
r_4_2_br_f[3]["run_details"]["solve_time_relaxation"]
r_4_2_br_f[3]["run_details"]["time_taken"]
r_4_2_br_f[3]["run_details"]["dict_num_interations_altmin"]

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 4: (k, m, n, i, seed, γ) = (2,10,10,60,0,20)"
)
plot!(
    r_4_2t[3]["run_log"][!,:runtime],
    r_4_2t[3]["run_log"][!,:gap],
    label = "With majorization",
    color = :orange
)
plot!(
    r_4_2f[3]["run_log"][!,:runtime],
    r_4_2f[3]["run_log"][!,:gap],
    label = "Without majorization",
    color = :blue
)

r_4_2at = test_matrix_completion_disjunctivecuts(
    2, 15, 15, 125, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    disjunctive_sorting = true,
    add_Shor_valid_inequalities = false,
    time_limit = 3600,
    use_max_steps = false,
)
r_4_2af = test_matrix_completion_disjunctivecuts(
    2, 15, 15, 125, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_2_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 3600,
    use_max_steps = false,
)

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 4: (k, m, n, i, seed, γ) = (2,10,10,60,0,20)"
)
plot!(
    r_4_2at[3]["run_log"][!,:runtime],
    r_4_2at[3]["run_log"][!,:gap],
    label = "With majorization",
    color = :orange
)
plot!(
    r_4_2af[3]["run_log"][!,:runtime],
    r_4_2af[3]["run_log"][!,:gap],
    label = "Without majorization",
    color = :blue
)