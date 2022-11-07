module TestMatrixCompletionDisjunctiveCuts

include("matrix_completion.jl")
include("utils.jl")

export test_matrix_completion_disjunctivecuts

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
    node_selection::String = "bestfirst",
    bestfirst_depthfirst_cutoff::Int = 10000,
    disjunctive_cuts_type::Union{String, Nothing} = "linear",
    disjunctive_cuts_breakpoints::Union{String, Nothing} = "smallest_1_eigvec",
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
        max_altmin_probability = max_altmin_probability,
        min_altmin_probability = min_altmin_probability,
        altmin_probability_decay_rate = altmin_probability_decay_rate,
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

end
