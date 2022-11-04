include("matrix_completion.jl")
include("utils.jl")

using Plots
using StatsBase
using Suppressor
using CSV

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
    disjunctive_cuts_type::Union{String, Nothing} = nothing,
    disjunctive_cuts_breakpoints::Union{String, Nothing} = nothing,
    disjunctive_sorting::Bool = false,
    presolve::Bool = false,
    add_basis_pursuit_valid_inequalities::Bool = false,
    add_Shor_valid_inequalities::Bool = false,
    max_update_Shor_indices_probability::Float64 = 1.0, # TODO
    min_update_Shor_indices_probability::Float64 = 0.1, # TODO
    update_Shor_indices_probability_decay_rate::Float64 = 1.1, # TODO
    update_Shor_indices_n_minors::Int = 100,
    root_only::Bool = false,
    altmin_flag::Bool = true,
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
        noise = false, ϵ = 0.0,
    )

    log_time = Dates.now()
    solution, printlist, instance = branchandbound_frob_matrixcomp(
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
        disjunctive_sorting = disjunctive_sorting,
        presolve = presolve,
        add_basis_pursuit_valid_inequalities = add_basis_pursuit_valid_inequalities,
        add_Shor_valid_inequalities = add_Shor_valid_inequalities,
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

# Presolve is required
test_basis_pursuit_disjunctivecuts(
    1, 10, 10, 20, 0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    presolve = false,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
test_basis_pursuit_disjunctivecuts(
    1, 10, 10, 20, 0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    presolve = true,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)

# Experiment 1: number of eigenvectors
runtimes_1 = Dict()
for n in [20, 30, 40, 50]
    runtimes_1[n] = Dict(
        "smallest_1_eigvec" => [],
        "smallest_2_eigvec" => [],
    )
    for seed in 1:20
        r_1_1 = test_basis_pursuit_disjunctivecuts(
            1, n, n, Int(round(2 * n * log10(n))), seed;
            node_selection = "bestfirst",
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_1_eigvec",
            presolve = true,
            time_limit = 600,
        )
        push!(runtimes_1[n]["smallest_1_eigvec"], r_1_1[3]["run_details"]["time_taken"])
        r_1_2 = test_basis_pursuit_disjunctivecuts(
            1, n, n, Int(round(2 * n * log10(n))), seed;
            node_selection = "bestfirst",
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_2_eigvec",
            presolve = true,
            time_limit = 600,
        )
        push!(runtimes_1[n]["smallest_2_eigvec"], r_1_2[3]["run_details"]["time_taken"])
    end
end
