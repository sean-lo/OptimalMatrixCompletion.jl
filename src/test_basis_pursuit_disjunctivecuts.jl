module TestBasisPursuitDisjunctiveCuts

include("matrix_completion.jl")
include("utils.jl")

using .MCBnB

export test_basis_pursuit_disjunctivecuts

function test_basis_pursuit_disjunctivecuts(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 0.0,
    node_selection::String = "breadthfirst",
    bestfirst_depthfirst_cutoff::Int = 10000,
    disjunctive_cuts_type::Union{String, Nothing} = "linear",
    disjunctive_cuts_breakpoints::Union{String, Nothing} = "smallest_1_eigvec",
    presolve::Bool = false,
    add_basis_pursuit_valid_inequalities::Bool = false,
    add_Shor_valid_inequalities::Bool = false,
    add_Shor_valid_inequalities_fraction::Float64 = 1.0,
    max_update_Shor_indices_probability::Float64 = 1.0, # TODO
    min_update_Shor_indices_probability::Float64 = 0.1, # TODO
    update_Shor_indices_probability_decay_rate::Float64 = 1.1, # TODO
    update_Shor_indices_n_minors::Int = 100,
    root_only::Bool = false,
    use_max_steps::Bool = true,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
    update_step::Int = 1000,
    with_log::Bool = true,
)
    (A, indices) = generate_matrixcomp_data(
        k, m, n, n_indices, seed; 
        noise = false, ϵ = 0.0,
    )

    log_time = Dates.now()
    r = branchandbound_frob_matrixcomp(
        k,
        A,
        indices,
        γ,
        λ,
        false, # noise
        ;
        node_selection = node_selection,
        bestfirst_depthfirst_cutoff = bestfirst_depthfirst_cutoff,
        use_disjunctive_cuts = true,
        disjunctive_cuts_type = disjunctive_cuts_type,
        disjunctive_cuts_breakpoints = disjunctive_cuts_breakpoints,
        presolve = presolve,
        add_basis_pursuit_valid_inequalities = add_basis_pursuit_valid_inequalities,
        add_Shor_valid_inequalities = add_Shor_valid_inequalities,
        add_Shor_valid_inequalities_fraction = add_Shor_valid_inequalities_fraction,
        max_update_Shor_indices_probability = max_update_Shor_indices_probability,
        min_update_Shor_indices_probability = min_update_Shor_indices_probability,
        update_Shor_indices_probability_decay_rate = update_Shor_indices_probability_decay_rate,
        update_Shor_indices_n_minors = update_Shor_indices_n_minors,
        root_only = root_only,
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

module TestBasisPursuitNonDisjunctiveCuts

include("matrix_completion.jl")
include("utils.jl")

export test_basis_pursuit_nondisjunctivecuts

function test_basis_pursuit_nondisjunctivecuts(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 0.0,
    branching_region::String = "box",
    branching_type::String = "lexicographic",
    branch_point::String = "midpoint",
    node_selection::String = "breadthfirst",
    bestfirst_depthfirst_cutoff::Int = 10000,
    presolve::Bool = false,
    add_basis_pursuit_valid_inequalities::Bool = false,
    root_only::Bool = false,
    use_max_steps::Bool = true,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
    update_step::Int = 1000,
    with_log::Bool = true,
)
    (A, indices) = generate_matrixcomp_data(
        k, m, n, n_indices, seed; 
        noise = false, ϵ = 0.0,
    )

    log_time = Dates.now()
    r = branchandbound_frob_matrixcomp(
        k,
        A,
        indices,
        γ,
        λ,
        false, # noise
        ;
        branching_region = branching_region,
        branching_type = branching_type,
        branch_point = branch_point,
        node_selection = node_selection,
        bestfirst_depthfirst_cutoff = bestfirst_depthfirst_cutoff,
        use_disjunctive_cuts = false,
        presolve = presolve,
        add_basis_pursuit_valid_inequalities = add_basis_pursuit_valid_inequalities,
        root_only = root_only,
        use_max_steps = use_max_steps,
        max_steps = max_steps,
        time_limit = time_limit,
        update_step = update_step,
    )
    solution, printlist, instance = r[1], r[2], r[3]

    if with_log
        time_string = lexicographicDates.format(log_time, "yyyymmdd_HHMMSS")
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
