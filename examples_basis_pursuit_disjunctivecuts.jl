include("test_basis_pursuit_disjunctivecuts.jl")

using .TestBasisPursuitDisjunctiveCuts
using Plots
using StatsBase
using Suppressor
using CSV
using JLD

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
r = test_basis_pursuit_disjunctivecuts(
    1, 10, 10, 20, 0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    presolve = true,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)

println(keys(r[1]))
r[1]["objective"]
r[1]["objective_initial"]



# Experiment 0: Presolve
results_1 = Dict()
results_1 = load("results/oct_22/bp1_presolve.jld")["r"]
for n in [10, 20, 30, 40, 50], presolve in [true]
    # if (n, presolve) in keys(results_1)
    #     continue
    # end
    println("""
    Experiment 0:
    Presolve?       $presolve     
    n:              $n

    """)
    d = Dict(
        "time_taken" => [], 
        "root_node_lower_bound" => [],
        "eventual_lower_bound" => [],
        "incumbent" => [],
        "nodes_explored" => [],
        "entries_presolved" => [],
    )
    if !((n, presolve) in keys(results_1))
        results_1[(n, presolve)] = d
    end
    for seed in 1:20
        if length(results_1[(n, presolve)]["time_taken"]) ≥ seed
            continue
        end
        r = @suppress test_basis_pursuit_disjunctivecuts(
            1, n, n, Int(round(2 * n * log10(n))), seed;
            node_selection = "bestfirst", 
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_1_eigvec",
            presolve = presolve,
            time_limit = Int(round(2 * n * n))
        )
        sleep(0.01)
        @printf("Run %2d:      %5f\n", seed, r[3]["run_details"]["time_taken"])
        push!(results_1[(n, presolve)]["time_taken"], r[3]["run_details"]["time_taken"])
        push!(results_1[(n, presolve)]["entries_presolved"], r[3]["run_details"]["entries_presolved"])
        push!(results_1[(n, presolve)]["incumbent"], r[1]["objective"])
        if r[3]["run_details"]["entries_presolved"] == n * n
            push!(results_1[(n, presolve)]["root_node_lower_bound"], r[1]["objective"])
            push!(results_1[(n, presolve)]["eventual_lower_bound"], r[1]["objective"])
            push!(results_1[(n, presolve)]["nodes_explored"], 0)
        else 
            push!(results_1[(n, presolve)]["root_node_lower_bound"], r[3]["run_log"][1,"lower"])
            push!(results_1[(n, presolve)]["eventual_lower_bound"], r[3]["run_log"][end,"lower"])
            push!(results_1[(n, presolve)]["nodes_explored"], r[3]["run_log"][end,"explored"])
        end
    end
end
save("results/oct_22/bp1_presolve.jld", "r", results_1)

results_1[(10,true)]

results_1
bp1_presolve_df = DataFrame(
    "n" => Int[], 
    "presolve" => Bool[],
    "seed" => Int[],
    "entries_presolved" => Int[],
    "time_taken" => Float64[],
    "nodes_explored" => Int[],
    "root_node_lower_bound" => Float64[],
    "eventual_lower_bound" => Float64[],
    "incumbent" => Float64[],
)
for (ind, ((n, presolve), v)) in enumerate(pairs(results_1))
    push!(bp1_presolve_df, 
        [
            (
                n, presolve, i,
                v["entries_presolved"][i], 
                v["time_taken"][i], v["nodes_explored"][i], 
                v["root_node_lower_bound"][i],
                v["eventual_lower_bound"][i], 
                v["incumbent"][i]
            )
            for i in 1:length(v["time_taken"])
        ]...
    )
end
transform!(
    bp1_presolve_df, 
    (
        [:entries_presolved, :n]
        => ((x1, x2) -> x1 .== x2 .* x2)
        => :solved_in_presolve
    ),
    (
        [:root_node_lower_bound, :incumbent] 
        => ((x1, x2) -> abs.(x1.-x2) ./ x1) 
        => :root_node_rel_gap
    ),
    (
        [:eventual_lower_bound, :incumbent] 
        => ((x1, x2) -> abs.(x1.-x2) ./ x1) 
        => :eventual_rel_gap
    ),
)
bp1_presolved_y_df = filter(
    r -> (
        r.solved_in_presolve
        && !(r.n in [40, 50]) # TODO remove;
    ), 
    bp1_presolve_df
)
bp1_presolved_n_df = filter(
    r -> (
        !r.presolve
        && !(r.n in [40, 50]) # TODO remove;
        && count(r2 -> (r2.n == r.n && r2.seed == r.seed), eachrow(bp1_presolved_y_df)) > 0
    ), 
    bp1_presolve_df
)
bp1_presolved_df = vcat(
    bp1_presolved_y_df,
    bp1_presolved_n_df,
)
bp1_not_presolved_df = filter(
    r -> (
        !(r.n in [40, 50]) # TODO remove;
    ), 
    antijoin(
        bp1_presolve_df,
        bp1_presolved_df,
        on = [:n, :seed],
    )
)

gdf_bp1_presolved = groupby(bp1_presolved_df, [:presolve, :n])
summary_bp1_presolved = combine(
    gdf_bp1_presolved,
    nrow,
    :entries_presolved => mean,
    :time_taken => mean,
    :nodes_explored => mean,
    :root_node_rel_gap => geomean,
    :eventual_rel_gap => geomean,
)

gdf_bp1_not_presolved = groupby(bp1_not_presolved_df, [:presolve, :n])
summary_bp1_not_presolved = combine(
    gdf_bp1_not_presolved,
    nrow,
    :entries_presolved => mean,
    :time_taken => mean,
    :nodes_explored => mean,
    :root_node_rel_gap => geomean,
    :eventual_rel_gap => geomean,
)



CSV.write("results/oct_22/bp1_presolve_df.csv", bp1_presolve_df)



# Experiment 1: number of eigenvectors
entries_after_presolveruntimes_1 = Dict()
for n in [10, 20, 30]
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

runtimes_1

runtimes_1_df = DataFrame(
    "smallest_1_eigvec" => [
        mean(runtimes_1[n]["smallest_1_eigvec"])
        for n in [10, 20, 30]
    ],
    "smallest_2_eigvec" => [
        mean(runtimes_1[n]["smallest_2_eigvec"])
        for n in [10, 20, 30]
    ],
)
display(runtimes_1_df)


# Experiment 2: Big-n basis pursuit
runtimes_2 = []
for seed in 1:20
    r_2 = test_basis_pursuit_disjunctivecuts(
        1, 100, 100, Int(round(1.5*100*log10(100))), seed;
        node_selection = "bestfirst",
        disjunctive_cuts_type = "linear",
        disjunctive_cuts_breakpoints = "smallest_1_eigvec",
        presolve = true,
        add_Shor_valid_inequalities = false,
        time_limit = 3600,
    )
    push!(runtimes_2, r_2[3]["run_details"]["time_taken"])
end

runtimes_2

# Experiment 3: majorization inequalities
r_3_1t = test_basis_pursuit_disjunctivecuts(
    1, 10, 10, 20, 0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = true,
    presolve = true,
    add_Shor_valid_inequalities = false,
    time_limit = 600,
    use_max_steps = false,
)
r_3_1f = test_basis_pursuit_disjunctivecuts(
    1, 10, 10, 20, 0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    presolve = true,
    add_Shor_valid_inequalities = false,
    time_limit = 600,
    use_max_steps = false,
)

# Experiment 3: majorization inequalities
r_3_2t = test_basis_pursuit_disjunctivecuts(
    2, 10, 10, 60, 0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = true,
    presolve = true,
    add_Shor_valid_inequalities = false,
    time_limit = 120,
    use_max_steps = false,
)
r_3_2t[3]["run_log"]
r_3_2t[3]["run_details"]["dict_solve_times_relaxation"]


r_3_2f = test_basis_pursuit_disjunctivecuts(
    2, 10, 10, 60, 0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    presolve = true,
    add_Shor_valid_inequalities = false,
    time_limit = 120,
    use_max_steps = false,
)

r_3_2f[3]["run_log"]
r_3_2f[3]["run_details"]["dict_solve_times_relaxation"]


plot(
    fmt=:png,
    ylabel = "Lower bound", xlabel = "Runtime (s)",
    title = "Experiment 3b: (Basis pursuit) (2, 10, 10, 60, 0)"
)
plot!(
    r_3_2t[3]["run_log"][!, :runtime],
    r_3_2t[3]["run_log"][!, :lower],
    label = "With majorization"
)
plot!(
    r_3_2f[3]["run_log"][!, :runtime],
    r_3_2f[3]["run_log"][!, :lower],
    label = "Without majorization"
)

# Experiment 4: Valid inequalities
r_4_1ff = test_basis_pursuit_disjunctivecuts(
    1, 20, 20, 52, 1;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    presolve = true,
    add_basis_pursuit_valid_inequalities = false,
    add_Shor_valid_inequalities = false,
    time_limit = 120,
    use_max_steps = false,
)
r_4_1tf = test_basis_pursuit_disjunctivecuts(
    1, 20, 20, 52, 1;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    presolve = true,
    add_basis_pursuit_valid_inequalities = true,
    add_Shor_valid_inequalities = false,
    time_limit = 120,
    use_max_steps = false,
)
r_4_1tt = test_basis_pursuit_disjunctivecuts(
    1, 20, 20, 52, 1;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    presolve = true,
    add_basis_pursuit_valid_inequalities = true,
    add_Shor_valid_inequalities = true,
    time_limit = 120,
    use_max_steps = false,
)

r_4_1ff[3]["run_details"]["time_taken"]
r_4_1tf[3]["run_details"]["time_taken"]
r_4_1tt[3]["run_details"]["time_taken"]

plot(
    fmt=:png,
    ylabel = "Lower bound", xlabel = "Runtime (s)",
    title = "Experiment 4a: (Basis pursuit) (1, 20, 20, 52, 1)"
)
plot!(
    r_4_1ff[3]["run_log"][!, :runtime],
    r_4_1ff[3]["run_log"][!, :lower],
    label = "None"
)
plot!(
    r_4_1tf[3]["run_log"][!, :runtime],
    r_4_1tf[3]["run_log"][!, :lower],
    label = "With linear constraints"
)
plot!(
    r_4_1tt[3]["run_log"][!, :runtime],
    r_4_1tt[3]["run_log"][!, :lower],
    label = "With linear and SDP constraints"
)

runtimes_4 = Dict()
indices_4 = Dict()
for n in [10, 20, 30, 40]
    runtimes_4[n] = Dict(
        (false, false) => [],
        (true, false) => [],
        (true, true) => [],
    )
    indices_4[n] = Dict(
        "indices_initial" => [],
        "indices_presolved" => [],
        "Shor_constraints_indexes" => [],
        "Shor_SOC_constraints_indexes" => [],
    )
    for seed in 1:20
        m = n
        n_indices = Int(round(n * log10(n) * 2))
        k = 1
        (A, indices) = generate_matrixcomp_data(
            k, m, n, n_indices, seed; 
            noise = false, ϵ = 0.0,
        )
        indices_presolved, X_presolved = rank1_presolve(indices, A)
        Shor_constraints_indexes = generate_rank1_basis_pursuit_Shor_constraints_indexes(indices_presolved, 1)
        Shor_non_SOC_constraints_indexes = unique(vcat(
            [
                [(i1,j1), (i1,j2), (i2,j1), (i2,j2)]
                for (i1, i2, j1, j2) in Shor_constraints_indexes
            ]...
        ))
        Shor_SOC_constraints_indexes = setdiff!(
            vec(collect(Iterators.product(1:n, 1:m))), Shor_non_SOC_constraints_indexes,
            [[x[1], x[2]] for x in findall(indices_presolved)]
        )
        push!(indices_4[n]["indices_initial"], indices)
        push!(indices_4[n]["indices_presolved"], indices_presolved)
        push!(indices_4[n]["Shor_constraints_indexes"], Shor_constraints_indexes)
        push!(indices_4[n]["Shor_SOC_constraints_indexes"], Shor_SOC_constraints_indexes)
        println("Run $seed:")
        println("Indices (before presolve): $n_indices")
        # println("$indices")
        println("Indices (after presolve): $(sum(indices_presolved))")
        display(indices_presolved)
        println("Shor_constraints_indexes: $(length(Shor_constraints_indexes))")
        println("Shor_SOC_constraints_indexes: $(length(Shor_SOC_constraints_indexes))")
        
        r_4_1_ff = @suppress test_basis_pursuit_disjunctivecuts(
            k, m, n, n_indices, seed;
            node_selection = "bestfirst",
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_1_eigvec",
            disjunctive_sorting = false,
            presolve = true,
            add_basis_pursuit_valid_inequalities = false,
            add_Shor_valid_inequalities = false,
            time_limit = 600,
            use_max_steps = false,
        )
        println("Time taken (F, F): ", r_4_1ff[3]["run_details"]["time_taken"])
        r_4_1tf = @suppress test_basis_pursuit_disjunctivecuts(
            k, m, n, n_indices, seed;
            node_selection = "bestfirst",
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_1_eigvec",
            disjunctive_sorting = false,
            presolve = true,
            add_basis_pursuit_valid_inequalities = true,
            add_Shor_valid_inequalities = false,
            time_limit = 600,
            use_max_steps = false,
        )
        println("Time taken (T, F): ", r_4_1tf[3]["run_details"]["time_taken"])
        r_4_1tt = @suppress test_basis_pursuit_disjunctivecuts(
            k, m, n, n_indices, seed;
            node_selection = "bestfirst",
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_1_eigvec",
            disjunctive_sorting = false,
            presolve = true,
            add_basis_pursuit_valid_inequalities = true,
            add_Shor_valid_inequalities = true,
            time_limit = 600,
            use_max_steps = false,
        )
        println("Time taken (T, T): ", r_4_1tt[3]["run_details"]["time_taken"])
        push!(runtimes_4[n][(false, false)], r_4_1ff[3])
        push!(runtimes_4[n][(true, false)], r_4_1tf[3])
        push!(runtimes_4[n][(true, true)], r_4_1tt[3])
    end
end
runtimes_4_df = DataFrame(
    n = Int[], 
    add_basis_pursuit_valid_inequalities = Bool[],
    add_Shor_valid_inequalities = Bool[],
    seed = Int[],
    time_taken = Float64[],
    nodes_explored = Int[],
    nodes_total = Int[],
)
for n in [10, 20, 30, 40]
    for param in [(false, false), (true, false), (true, true)]
        for (seed, d) in enumerate(runtimes_4[n][param])
            push!(
                runtimes_4_df, 
                (
                    n, 
                    param[1],
                    param[2],
                    seed,
                    d["run_details"]["time_taken"],
                    d["run_details"]["nodes_explored"],
                    d["run_details"]["nodes_total"]
                )
            )
        end
    end
end
# Including all examples
table_4 = combine(
    groupby(
        runtimes_4_df, 
        [
            :n, 
            :add_basis_pursuit_valid_inequalities, 
            :add_Shor_valid_inequalities
        ]
    ),
    :time_taken => mean,
    :nodes_explored => mean,
)
# Excluding examples solved in presolve
table_4_no_presolve = combine(
    groupby(
        filter(
            :nodes_explored => (x -> x > 0),
            runtimes_4_df
        ), 
        [
            :n, 
            :add_basis_pursuit_valid_inequalities, 
            :add_Shor_valid_inequalities
        ]
    ),
    :time_taken => mean,
    :nodes_explored => mean,
    nrow,
)

unstack(
    table_4_no_presolve, 
    [
        :add_basis_pursuit_valid_inequalities, 
        :add_Shor_valid_inequalities
    ], 
    :n, 
    :time_taken_mean,
) 