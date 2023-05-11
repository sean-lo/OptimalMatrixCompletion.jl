include("test_matrix_completion_disjunctivecuts.jl")
include("utils.jl")
include("matrix_completion.jl")

using .TestMatrixCompletionDisjunctiveCuts
using .TestMatrixCompletionNonDisjunctiveCuts
using Plots
using StatsBase
using Suppressor
using CSV
using JLD
using DataFrames


# Experiment 1: number of eigenvectors
r_1_1 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_1_1[3]["run_details"]["dict_num_iterations_altmin"]
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
    time_limit = 180,
    use_max_steps = false,
    altmin_flag = false,
)

r_4_2t = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = true,
    add_Shor_valid_inequalities = false,
    time_limit = 800,
    use_max_steps = false,
)
r_4_2f = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 200,
    use_max_steps = false,
)

r_4_2t_noaltmin[3]["run_details"]["dict_solve_times_altmin"]
r_4_2t_noaltmin[3]["run_details"]["solve_time_altmin"]
r_4_2t_noaltmin[3]["run_details"]["solve_time_relaxation"]
r_4_2t_noaltmin[3]["run_details"]["time_taken"]
r_4_2t_noaltmin[3]["run_details"]["dict_num_iterations_altmin"]
r_4_2t_noaltmin[3]["run_details"]["dict_solve_times_relaxation"]
show(r_4_2t_noaltmin[3]["run_details"]["dict_solve_times_relaxation"][:,:], allrows=true)

r_4_2t[3]["run_details"]["nodes_explored"]
r_4_2t[3]["run_details"]["dict_solve_times_altmin"]
r_4_2t[3]["run_details"]["solve_time_altmin"]
r_4_2t[3]["run_details"]["solve_time_relaxation"]
r_4_2t[3]["run_details"]["time_taken"]
r_4_2t[3]["run_details"]["dict_num_iterations_altmin"]
show(r_4_2t[3]["run_details"]["dict_solve_times_relaxation"][:,:], allrows=true)

r_4_2f[3]["run_details"]["nodes_explored"]
r_4_2f[3]["run_details"]["dict_solve_times_altmin"]
r_4_2f[3]["run_details"]["solve_time_altmin"]
r_4_2f[3]["run_details"]["solve_time_relaxation"]
r_4_2f[3]["run_details"]["time_taken"]
r_4_2f[3]["run_details"]["dict_num_iterations_altmin"]
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

# Experiment 5: strength of root node relaxation,
# depending on when you apply Shor LMI inequalities

noise = 0.01
γ = 20.0
root_node_lower_bounds_df = DataFrame(
    type = String[],
    n = Int[],
    p = Int[],
    seed = Int[],
    noise = Float64[],
    γ = Float64[],
    lower_bound = Float64[],
    incumbent = Float64[],
    time_taken = Float64[],
)
for type in [
    "none", 
    "4", 
    "43", 
    # "432", 
    # "4321", 
    # "43210",
]
    if type == "none"
        add_Shor_valid_inequalities = false
        Shor_valid_inequalities_noisy_rank1_num_entries_present = Int[]
    else
        add_Shor_valid_inequalities = true
        if type == "4"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4]
        elseif type == "43"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3]
        elseif type == "432"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3, 2]
        elseif type == "4321"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3, 2, 1]
        elseif type == "43210"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3, 2, 1, 0]
        end
    end
    for n in [
        10, 
        20, 
        30, 
        40, 
        50,
    ]
        for p in [
            2, 
            # 3, 
            # 4, 
            # 5,
        ]
            for seed in 1:20
                r_5 = test_matrix_completion_disjunctivecuts(
                    1, n, n, Int(round(p * n * log10(n))), seed, noise, γ;
                    add_Shor_valid_inequalities = add_Shor_valid_inequalities,
                    Shor_valid_inequalities_noisy_rank1_num_entries_present = Shor_valid_inequalities_noisy_rank1_num_entries_present,
                    root_only = true,
                )
                push!(root_node_lower_bounds_df, (
                    type, n, p, seed, noise, γ,
                    r_5[3]["run_log"][1,"lower"],
                    r_5[1]["objective"],
                    r_5[3]["run_details"]["time_taken"],
                ))
            end
        end
    end
end

transform!(
    root_node_lower_bounds_df, 
    [:lower_bound, :incumbent] => ((x1, x2) -> abs.(x1.-x2) ./ x1) => :rel_gap
)

gdf_5 = groupby(root_node_lower_bounds_df, [:type, :n, :p])
rootnode_lb_gdf_summary = combine(
    gdf_5, 
    :time_taken => mean,
    :rel_gap => geomean,
)
filter(
    r -> (r.n == 10),
    rootnode_lb_gdf_summary,
)

plot(
    xscale = :log, xaxis = "Time taken (s)",
    yscale = :log, yaxis = "Relative gap",
    size = (900, 600)
)
colors = Dict(10 => :red, 20 => :orange, 30 => :green)
shapes = Dict(3 => :utriangle, 4 => :diamond, 5 => :star5)
for n in [10, 20, 30]
    for p in [3, 4, 5]
        plot!(
            filter(r -> (r.n == n && r.p == p), rootnode_lb_gdf_summary)[!, :time_taken_mean], 
            filter(r -> (r.n == n && r.p == p), rootnode_lb_gdf_summary)[!, :rel_gap_geomean],
            label = "n = $n, $p × $n log10($n) entries",
            color = colors[n], shape = shapes[p], 
            linealpha = 0.2
        )
    end
end
plot!(
    title = "Relative gap at root node against time taken, \nand different extent of Shor LMIs, \nfor matrix completion (k=1)",
    legend = :topright, fmt = :png
)
savefig("plots/root_node_lower_bounds_matrix_completion_Shor_1.png")

test_matrix_completion_disjunctivecuts(
    1, n, n, Int(round(p * n * log10(n))), seed, noise, γ;
    add_Shor_valid_inequalities = add_Shor_valid_inequalities,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = Shor_valid_inequalities_noisy_rank1_num_entries_present,
    root_only = true,
)

r_5_1t = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 60,
    use_max_steps = false,
)
r_5_1f = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
    use_max_steps = false,
)
plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 5: (k, m, n, i, seed, noise, γ) = (1, 10, 10, 30, 0, 0.01, 20)"
)
plot!(
    r_5_1t[3]["run_log"][!,:runtime],
    r_5_1t[3]["run_log"][!,:gap],
    label = "With Shor valid inequalities (4)",
    color = :orange
)
plot!(
    r_5_1f[3]["run_log"][!,:runtime],
    r_5_1f[3]["run_log"][!,:gap],
    label = "Without Shor_valid inequalities",
    color = :blue
)





# Experiment 5.2: rank-2
# strength of root node relaxation,
# depending on when you apply Shor LMI inequalities

noise = 0.01
γ = 20.0
root_node_lower_bounds_df_2 = DataFrame(
    type = String[],
    n = Int[],
    p = Int[],
    seed = Int[],
    noise = Float64[],
    γ = Float64[],
    lower_bound = Float64[],
    incumbent = Float64[],
    time_taken = Float64[],
)


for type in [
    "none", 
    "4", 
    "43", 
    # "432", 
    # "4321", 
    # "43210",
]
    if type == "none"
        add_Shor_valid_inequalities = false
        Shor_valid_inequalities_noisy_rank1_num_entries_present = Int[]
    else
        add_Shor_valid_inequalities = true
        if type == "4"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4]
        elseif type == "43"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3]
        elseif type == "432"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3, 2]
        elseif type == "4321"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3, 2, 1]
        elseif type == "43210"
            Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3, 2, 1, 0]
        end
    end
    for n in [
        10, 
        20, 
        30, 
        40, 
        50,
    ]
        for p in [
            2, 
            # 3, 
            # 4, 
            # 5,
        ]
            for seed in 1:20
                if nrow(filter(
                    r -> (
                        r.n == n
                        && r.p == p
                        && r.seed == seed
                        && r.noise == noise
                        && r.γ == γ
                        && r.type == type
                    ),
                    root_node_lower_bounds_df_2
                )) ≥ 1
                    continue
                end
                r_5 = test_matrix_completion_disjunctivecuts(
                    2, n, n, Int(round(p * 2 * n * log10(n))), seed, noise, γ;
                    add_Shor_valid_inequalities = add_Shor_valid_inequalities,
                    Shor_valid_inequalities_noisy_rank1_num_entries_present = Shor_valid_inequalities_noisy_rank1_num_entries_present,
                    root_only = true,
                )
                push!(root_node_lower_bounds_df_2, (
                    type, n, p, seed, noise, γ,
                    r_5[3]["run_log"][1,"lower"],
                    r_5[1]["objective"],
                    r_5[3]["run_details"]["time_taken"],
                ))
            end
        end
    end
end
CSV.write("results/oct_22/mc2_rootnode.csv", root_node_lower_bounds_df_2)

root_node_lower_bounds_df_2 = CSV.read("results/oct_22/mc2_rootnode.csv", DataFrame)

transform!(
    root_node_lower_bounds_df_2, 
    [:lower_bound, :incumbent] => ((x1, x2) -> abs.(x1.-x2) ./ x1) => :rel_gap
)
gdf_5_2 = groupby(root_node_lower_bounds_df_2, [:n, :type, :p])
rootnode_2_lb_gdf_summary = combine(
    gdf_5_2, 
    :time_taken => mean,
    :rel_gap => geomean,
)
groupby(root_node_lower_bounds_df_2, [:seed, :n, :p])

rootnode_2_lb_gdf_summary

plot(
    xscale = :log, xaxis = "Time taken (s)",
    yscale = :log, yaxis = "Relative gap",
    size = (900, 600)
)
colors = Dict(10 => :red, 20 => :orange, 30 => :green, 40 => :blue, 50 => :black)
shapes = Dict(2 => :cross, :3 => :utriangle, 4 => :diamond, 5 => :star5)
for n in [10, 20, 30, 40, 50]
    for p in [2]
        plot!(
            filter(r -> (r.n == n && r.p == p), rootnode_2_lb_gdf_summary)[!, :time_taken_mean], 
            filter(r -> (r.n == n && r.p == p), rootnode_2_lb_gdf_summary)[!, :rel_gap_geomean],
            label = "n = $n, 2 × $p × $n log10($n) entries",
            color = colors[n], shape = shapes[p], 
            linealpha = 0.2
        )
    end
end
plot!(
    title = """
    Relative gap at root node against time taken,
    and different extent of Shor LMIs,
    for matrix completion (k=2)""",
    legend = :topright, fmt = :png
)
savefig("plots/root_node_lower_bounds_matrix_completion_Shor_2.png")



r_5_2t = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 3, 0.001, 10.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 600,
    use_max_steps = false,
)
r_5_2f = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 3, 0.001, 10.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 600,
    use_max_steps = false,
)

keys(r_5_2t[3]["run_details"])

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 5: (k, m, n, i, seed, noise, γ) = (2,10,10,60,3,0.001,10)"
)
plot!(
    r_5_2t[3]["run_log"][!,:runtime],
    r_5_2t[3]["run_log"][!,:gap],
    label = "With Shor valid inequalities (4)",
    color = :orange
)
plot!(
    r_5_2f[3]["run_log"][!,:runtime],
    r_5_2f[3]["run_log"][!,:gap],
    label = "Without Shor_valid inequalities",
    color = :blue
)

# Experiment 6: Iterative Shor valid inequalities

r_6_1tt43 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    add_Shor_valid_inequalities_iterative = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4,3],
    time_limit = 1000,
    use_max_steps = false,
)

(A, indices) = generate_matrixcomp_data(
    1, 10, 10, 30, 0; noise = true, ϵ = 0.01
)
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [4]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [3]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [2]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [1]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [0]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [4,3,2,1,0]
))

r_6_1 = Dict()
for Shor_valid_inequalities_noisy_rank1_num_entries_present in [
    [4], 
    [4,3], 
    [4,3,2],
]
    for add_Shor_valid_inequalities_iterative in [
        true, 
        false,
    ]
        r_6_1[
            (
                Shor_valid_inequalities_noisy_rank1_num_entries_present,
                add_Shor_valid_inequalities_iterative
            )
        ] = test_matrix_completion_disjunctivecuts(
            1, 10, 10, 30, 0, 0.01, 20.0;
            node_selection = "bestfirst",
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_1_eigvec",
            disjunctive_sorting = false,
            add_Shor_valid_inequalities = true,
            add_Shor_valid_inequalities_iterative = add_Shor_valid_inequalities_iterative,
            Shor_valid_inequalities_noisy_rank1_num_entries_present = Shor_valid_inequalities_noisy_rank1_num_entries_present,
            time_limit = 60,
            use_max_steps = false,
        )
    end
end
r_6_1[false] = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
    use_max_steps = false,
)

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 6: (k, m, n, i, seed, noise, γ) = (1,10,10,30,0,0.01,20)"
)
c = Dict(
    [4] => :red,
    [4,3] => :orange,
    [4,3,2] => :green,
)
s = Dict(
    true => :solid,
    false => :dash,
)
for (
    Shor_valid_inequalities_noisy_rank1_num_entries_present,
    add_Shor_valid_inequalities_iterative
) in [
    ([4], true),
    ([4], false),
    ([4,3], true),
    ([4,3], false),
    ([4,3,2], true),
    ([4,3,2], false),
]
    plot!(
        r_6_1[(
            Shor_valid_inequalities_noisy_rank1_num_entries_present,
            add_Shor_valid_inequalities_iterative
        )][3]["run_log"][!,:runtime],
        abs.(r_6_1[(
            Shor_valid_inequalities_noisy_rank1_num_entries_present,
            add_Shor_valid_inequalities_iterative
        )][3]["run_log"][!,:gap]),
        label = "$Shor_valid_inequalities_noisy_rank1_num_entries_present, $add_Shor_valid_inequalities_iterative",
        color = c[Shor_valid_inequalities_noisy_rank1_num_entries_present],
        style = s[add_Shor_valid_inequalities_iterative],
    )
end
plot!(
    r_6_1[false][3]["run_log"][!,:runtime],
    r_6_1[false][3]["run_log"][!,:gap],
    label = "Without",
    color = :blue
)

r_6_1[([4,3],false)]
r_6_1[([4,3,2],false)][3]["run_log"]
r_6_1[([4,3],false)][3]["run_log"]
r_6_1[([4],false)][3]["run_log"]
r_6_1[([4,3,2],true)][3]["run_log"]
r_6_1[([4,3],true)][3]["run_log"]
r_6_1[([4],true)][3]["run_log"]



(A, indices) = generate_matrixcomp_data(
    1, 30, 30, 66, 4; noise = true, ϵ = 0.01
)
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [4]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [3]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [2]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [1]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [0]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [4,3,2,1,0]
))

r_6_1b = Dict()
for Shor_valid_inequalities_noisy_rank1_num_entries_present in [
    [4], 
    [4,3], 
    # [4,3,2],
]
    for add_Shor_valid_inequalities_iterative in [
        true, 
        false,
    ]
        r_6_1b[
            (
                Shor_valid_inequalities_noisy_rank1_num_entries_present,
                add_Shor_valid_inequalities_iterative
            )
        ] = test_matrix_completion_disjunctivecuts(
            1, 30, 30, 66, 4, 0.01, 20.0;
            node_selection = "bestfirst",
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_1_eigvec",
            disjunctive_sorting = false,
            add_Shor_valid_inequalities = true,
            add_Shor_valid_inequalities_iterative = add_Shor_valid_inequalities_iterative,
            Shor_valid_inequalities_noisy_rank1_num_entries_present = Shor_valid_inequalities_noisy_rank1_num_entries_present,
            time_limit = 200,
            use_max_steps = false,
        )
        println(
            r_6_1b[
                (
                    Shor_valid_inequalities_noisy_rank1_num_entries_present,
                    add_Shor_valid_inequalities_iterative
                )
            ][3]["run_log"]
        )
    end
end
r_6_1b[false] = test_matrix_completion_disjunctivecuts(
    1, 30, 30, 66, 4, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 200,
    use_max_steps = false,
)




r_6_2t432 = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 2, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3, 2],
    time_limit = 60,
    use_max_steps = false,
    root_only = true,
)





r_6_2t43 = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 2, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3],
    time_limit = 60,
    use_max_steps = false,
    root_only = true,
)

r_6_2t4 = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 2, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 60,
    use_max_steps = false,
    root_only = true,
)

r_6_2f = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 60, 2, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 60,
    use_max_steps = false,
    root_only = true,
)

r_6_2a_tt432 = test_matrix_completion_disjunctivecuts(
    2, 20, 20, Int(round(2 * 2 * 20 * log10(20))), 1, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    add_Shor_valid_inequalities_iterative = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3, 2],
    time_limit = 300,
    use_max_steps = false,
    root_only = false,
)
r_6_2a_tt43 = test_matrix_completion_disjunctivecuts(
    2, 20, 20, Int(round(2 * 2 * 20 * log10(20))), 1, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    add_Shor_valid_inequalities_iterative = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3],
    time_limit = 300,
    use_max_steps = false,
    root_only = false,
)

r_6_2a_t4 = test_matrix_completion_disjunctivecuts(
    2, 20, 20, Int(round(2 * 2 * 20 * log10(20))), 1, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 300,
    use_max_steps = false,
    root_only = false,
)
r_6_2a_tt4 = test_matrix_completion_disjunctivecuts(
    2, 20, 20, Int(round(2 * 2 * 20 * log10(20))), 1, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = true,
    add_Shor_valid_inequalities_iterative = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 300,
    use_max_steps = false,
    root_only = false,
)

r_6_2a_f = test_matrix_completion_disjunctivecuts(
    2, 20, 20, Int(round(2 * 2 * 20 * log10(20))), 1, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    disjunctive_sorting = false,
    add_Shor_valid_inequalities = false,
    time_limit = 300,
    use_max_steps = false,
    root_only = false,
)

r_6_2a_t43[3]["run_log"]
r_6_2a_t4[3]["run_log"]
r_6_2a_f[3]["run_log"]


# Experiment 7: Slices of Y
r_7_1f = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    disjunctive_slices = false,
    time_limit = 60,
)
r_7_1t = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    disjunctive_slices = false,
    time_limit = 60,
)

r_7_2 = Dict()
for (disjunctive_slices, (add_Shor_valid_inequalities, Shor_valid_inequalities_noisy_rank1_num_entries_present)) in Iterators.product(
    [false, true],
    [(false, [4]), (true, [4]), (true, [4,3]), (true, [4,3,2]), (true, [4,3,2,1])],
)
    r_7_2[(
        disjunctive_slices, 
        add_Shor_valid_inequalities, 
        Shor_valid_inequalities_noisy_rank1_num_entries_present
    )] = test_matrix_completion_disjunctivecuts(
        2, 10, 10, 80, 1, 0.1, 100.0;
        node_selection = "bestfirst",
        disjunctive_cuts_type = "linear",
        disjunctive_cuts_breakpoints = "smallest_1_eigvec",
        root_only = true,
        add_Shor_valid_inequalities = add_Shor_valid_inequalities,
        disjunctive_slices = disjunctive_slices,
        Shor_valid_inequalities_noisy_rank1_num_entries_present = Shor_valid_inequalities_noisy_rank1_num_entries_present,
        time_limit = 3600,
        use_max_steps = false,
    )
end

r_7_2_df = Dict(
    "disjunctive_slices" => Bool[],
    "add_Shor_valid_inequalities" => Bool[],
    "Shor_valid_inequalities_noisy_rank1_num_entries_present" => Vector{Int}[],
    "gap" => Float64[],
    "time_taken" => Float64[],
)
for (disjunctive_slices, (add_Shor_valid_inequalities, Shor_valid_inequalities_noisy_rank1_num_entries_present)) in Iterators.product(
    [false, true],
    [(false, [4]), (true, [4]), (true, [4,3]), (true, [4,3,2]), (true, [4,3,2,1])],
)
    key = (disjunctive_slices, add_Shor_valid_inequalities, Shor_valid_inequalities_noisy_rank1_num_entries_present)
    println(key)
    push!(r_7_2_df["disjunctive_slices"], disjunctive_slices)
    push!(r_7_2_df["add_Shor_valid_inequalities"], add_Shor_valid_inequalities)
    push!(r_7_2_df["Shor_valid_inequalities_noisy_rank1_num_entries_present"], Shor_valid_inequalities_noisy_rank1_num_entries_present)
    push!(r_7_2_df["gap"], r_7_2[key][3]["run_log"][1,:gap])
    push!(r_7_2_df["time_taken"], r_7_2[key][3]["run_details"]["time_taken"])
end

r_7_2_df = DataFrame(r_7_2_df)

plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 7: (k, m, n, i, seed, noise, γ) = (2,10,10,80,1,0.1,100)"
)
scatter!(
    [r_7_2_df[1,:time_taken],],
    [r_7_2_df[1,:gap],],
    color = :black,
    label = "with slices of Y",
)
scatter!(
    filter(r -> (r.disjunctive_slices), r_7_2_df)[!,:time_taken],
    filter(r -> (r.disjunctive_slices), r_7_2_df)[!,:gap],
    color = :blue,
    label = "with slices of Y",
)
scatter!(
    filter(r -> (!r.disjunctive_slices), r_7_2_df)[!,:time_taken],
    filter(r -> (!r.disjunctive_slices), r_7_2_df)[!,:gap],
    color = :red,
    label = "without slices of Y",
)











plot(
    xaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 7: (k, m, n, i, seed, noise, γ) = (2,10,10,80,1,0.1,100)"
)
c = Dict(
    (false, [4]) => :black,
    (true, [4]) => :red,
    (true, [4,3]) => :orange,
    (true, [4,3,2]) => :green,
    (true, [4,3,2,1]) => :blue,
)
s = Dict(
    true => :circle,
    false => :cross,
)
for (disjunctive_slices, (add_Shor_valid_inequalities, Shor_valid_inequalities_noisy_rank1_num_entries_present)) in Iterators.product(
    [false, true],
    [(false, [4]), (true, [4]), (true, [4,3]), (true, [4,3,2]), (true, [4,3,2,1])],
)
    label = ""
    if disjunctive_slices
        label *= "slices of Y"
    else
        label *= "Y"
    end
    if add_Shor_valid_inequalities
        label *= ", with Shor $(Shor_valid_inequalities_noisy_rank1_num_entries_present)"
    else
        label *= ", without Shor"
    end
    scatter!(
        [r_7_2[(
            disjunctive_slices,
            add_Shor_valid_inequalities,
            Shor_valid_inequalities_noisy_rank1_num_entries_present,
        )][3]["run_log"][!,:runtime],],
        [r_7_2[(
            disjunctive_slices, 
            add_Shor_valid_inequalities, 
            Shor_valid_inequalities_noisy_rank1_num_entries_present,
        )][3]["run_log"][1,:gap],],
        label = label,
        color = c[(add_Shor_valid_inequalities,
        Shor_valid_inequalities_noisy_rank1_num_entries_present)],
        shape = s[disjunctive_slices],
    )
end
plot!(legend = :topright)


r_8_2f = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 40, 1, 0.1, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    root_only = false,
    add_Shor_valid_inequalities = true,
    disjunctive_slices = false,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 3600,
    use_max_steps = false,
)
r_8_2 = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 40, 1, 0.1, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    root_only = false,
    add_Shor_valid_inequalities = true,
    disjunctive_slices = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 3600,
    use_max_steps = false,
)
r_8_2new = test_matrix_completion_disjunctivecuts(
    2, 10, 10, 40, 1, 0.1, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec_same",
    root_only = false,
    add_Shor_valid_inequalities = true,
    disjunctive_slices = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 3600,
    use_max_steps = false,
)
plot(
    yaxis=:log10, ylim=(10^(-2.5), 10^(-2)),
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 8: (k, m, n, i, seed, noise, γ) = (2, 10, 10, 40, 1, 0.1, 100.0)"
)
plot!(
    r_8_2f[3]["run_log"][!,:runtime],
    r_8_2f[3]["run_log"][!,:gap],
    label = "Without slices",
    color = :black
)
plot!(
    r_8_2new[3]["run_log"][!,:runtime],
    r_8_2new[3]["run_log"][!,:gap],
    label = "With a single x",
    color = :orange
)
plot!(
    r_8_2[3]["run_log"][!,:runtime],
    r_8_2[3]["run_log"][!,:gap],
    label = "With different x",
    color = :blue
)
savefig("notes/nov_28/plot_2.png")
r_8_2new[3]["run_log"]
r_8_2[3]["run_log"]


# Experiment 9: root node lower bounds, with fraction of Shor LMIs
n = 50
num_indices = Int(round(2.0 * n * log10(n)))
(A, indices) = generate_matrixcomp_data(
    1, n, n, num_indices, 0; noise = true, ϵ = 0.1
)
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [4]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [4,3]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [4,3,2]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [4,3,2,1]
))
length(generate_rank1_matrix_completion_Shor_constraints_indexes(
    indices, [4,3,2,1,0]
))
r_9_1_pre = Dict()
r_9_1_pre[(false, Int64[], 1.0)] = test_matrix_completion_disjunctivecuts(
    1, n, n, num_indices, 0, 0.1, 80.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    root_only = true,
    add_Shor_valid_inequalities = false,
    add_Shor_valid_inequalities_fraction = 1.0,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 3600,
    use_max_steps = false,
);
for (add_Shor, num_entries_present, fraction) in Iterators.product(
    [true],
    [[4], [4,3], [4,3,2]],
    [0.125],
)
    r_9_1_pre[(add_Shor, num_entries_present, fraction)] = test_matrix_completion_disjunctivecuts(
        1, n, n, num_indices, 0, 0.1, 80.0;
        node_selection = "bestfirst",
        disjunctive_cuts_type = "linear",
        disjunctive_cuts_breakpoints = "smallest_1_eigvec",
        root_only = true,
        add_Shor_valid_inequalities = add_Shor,
        add_Shor_valid_inequalities_fraction = fraction,
        Shor_valid_inequalities_noisy_rank1_num_entries_present = num_entries_present,
        time_limit = 3600,
        use_max_steps = false,
    );
end

r_9_1_pre[(false, Int64[], 1.0)][3]["run_details"]["time_taken"]
r_9_1_pre[(true, [4], 1.0)][3]["run_details"]["time_taken"]
r_9_1_pre[(true, [4,3], 1.0)][3]["run_details"]["time_taken"]
r_9_1_pre[(true, [4], 0.5)][3]["run_details"]["time_taken"]
r_9_1_pre[(true, [4,3], 0.5)][3]["run_details"]["time_taken"]
r_9_1_pre[(true, [4], 0.25)][3]["run_details"]["time_taken"]
r_9_1_pre[(true, [4,3], 0.25)][3]["run_details"]["time_taken"]
r_9_1_pre[(true, [4], 0.125)][3]["run_details"]["time_taken"]
r_9_1_pre[(true, [4,3], 0.125)][3]["run_details"]["time_taken"]

r_9_1_pre[(false, Int64[], 1.0)][3]["run_log"][end, :gap]
r_9_1_pre[(true, [4], 1.0)][3]["run_log"][end, :gap]
r_9_1_pre[(true, [4,3], 1.0)][3]["run_log"][end, :gap]
r_9_1_pre[(true, [4], 0.5)][3]["run_log"][end, :gap]
r_9_1_pre[(true, [4,3], 0.5)][3]["run_log"][end, :gap]
r_9_1_pre[(true, [4], 0.25)][3]["run_log"][end, :gap]
r_9_1_pre[(true, [4,3], 0.25)][3]["run_log"][end, :gap]
r_9_1_pre[(true, [4], 0.125)][3]["run_log"][end, :gap]
r_9_1_pre[(true, [4,3], 0.125)][3]["run_log"][end, :gap]



r_9_1

r_9_1 = Dict()
r_9_1_df = DataFrame(
    "n" => Int64[],
    "p" => Float64[],
    "seed" => Int64[],
    "noise" => Float64[],
    "γ" => Float64[],
    "add_Shor_valid_inequalities" => Bool[],
    "add_Shor_valid_inequalities_fraction" => Float64[],
    "Shor_valid_inequalities_noisy_rank1_num_entries_present" => Vector{Int64}[],
    "time_taken" => Float64[],
    "relative_gap" => Float64[],
)
for seed in 1:20, (n, p, noise, γ) in [
    (50, 2.0, 0.2, 80.0),
]
    num_indices = Int(round(p * n * log10(n)))
    for (add_Shor, num_entries_present, fraction) in vcat(
        vec(collect(Iterators.product(
            [false],
            [Int64[]],
            [1.0],
        ))),
        vec(collect(Iterators.product(
            [true],
            [[4],[4,3]],
            [1.0, 0.5, 0.25, 0.125],
        )))
    )
        r_9_1[(n, p, seed, noise, γ, add_Shor, num_entries_present, fraction)] = @suppress test_matrix_completion_disjunctivecuts(
            1, n, n, num_indices, seed, noise, γ;
            node_selection = "bestfirst",
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_1_eigvec",
            root_only = true,
            add_Shor_valid_inequalities = add_Shor,
            add_Shor_valid_inequalities_fraction = fraction,
            Shor_valid_inequalities_noisy_rank1_num_entries_present = num_entries_present,
            time_limit = 3600,
            use_max_steps = false,
        );
        println(
            """
            n = $(n), p = $(p), seed = $(seed), noise = $(noise), γ = $(γ)
            add_Shor: $(add_Shor), num_entries_present: $(num_entries_present), fraction = $(fraction)

            """
        )
        push!(
            r_9_1_df,
            (
                n, p, seed, noise, γ,
                add_Shor, fraction, num_entries_present,
                r_9_1[(n, p, seed, noise, γ, add_Shor, num_entries_present, fraction)][3]["run_details"]["time_taken"],
                r_9_1[(n, p, seed, noise, γ, add_Shor, num_entries_present, fraction)][3]["run_log"][end,:gap],
            )
        )
    end
end

r_9_1_df

r_9_1_combined_df = r_9_1_df |>
    x -> groupby(x, [:add_Shor_valid_inequalities, :add_Shor_valid_inequalities_fraction, :Shor_valid_inequalities_noisy_rank1_num_entries_present]) |>
    x -> combine(
        x, 
        :time_taken => geomean, 
        :relative_gap => (x -> geomean(abs.(x))) => :relative_gap_geomean,
    )


plot(
    size = (750, 500),
    yaxis=:log10,
    xaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 9: (k, m, n, i, noise, γ) = (1,50,50,170,0.1,20)"
)
scatter!(
    [r_9_1_combined_df[1,:time_taken_geomean]],
    [r_9_1_combined_df[1,:relative_gap_geomean]],
    label = "without",
    color = :black,
)
plot!(
    filter(
        r -> r.Shor_valid_inequalities_noisy_rank1_num_entries_present == [4] && r.add_Shor_valid_inequalities,
        r_9_1_combined_df
    )[!,:time_taken_geomean],
    filter(
        r -> r.Shor_valid_inequalities_noisy_rank1_num_entries_present == [4] && r.add_Shor_valid_inequalities,
        r_9_1_combined_df
    )[!,:relative_gap_geomean],
    label = "Shor-[4]",
    color = :blue,
    shape = :circle,
)
plot!(
    filter(
        r -> r.Shor_valid_inequalities_noisy_rank1_num_entries_present == [4, 3],
        r_9_1_combined_df
    )[!,:time_taken_geomean],
    filter(
        r -> r.Shor_valid_inequalities_noisy_rank1_num_entries_present == [4, 3],
        r_9_1_combined_df
    )[!,:relative_gap_geomean],
    label = "Shor-[4, 3]",
    shape = :circle,
    color = :red,
)


# Experiment 10: branching strategy
r_10_1_b = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_10_1_br = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "breadthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_10_1_d = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "depthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_10_1_bd_100 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst_depthfirst",
    bestfirst_depthfirst_cutoff = 100,
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_10_1_bd_300 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst_depthfirst",
    bestfirst_depthfirst_cutoff = 300,
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)
r_10_1_bd_1000 = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst_depthfirst",
    bestfirst_depthfirst_cutoff = 1000,
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 60,
)


r_10_1_s4_b = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 60,
)
r_10_1_s43_b = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4, 3],
    time_limit = 60,
)
r_10_1_s4_d = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "depthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 60,
)
r_10_1_s4_br = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "breadthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = true,
    Shor_valid_inequalities_noisy_rank1_num_entries_present = [4],
    time_limit = 60,
)


plot(
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 10: (k, m, n, i, seed, γ) = (1,10,10,30,0,20)"
)
plot!(
    r_10_1_b[3]["run_log"][!,:runtime],
    r_10_1_b[3]["run_log"][!,:gap],
    label = "bestfirst",
    color = :orange
)
plot!(
    r_10_1_br[3]["run_log"][!,:runtime],
    r_10_1_br[3]["run_log"][!,:gap],
    label = "breadthfirst",
    color = :blue
)
plot!(
    r_10_1_s4_br[3]["run_log"][!,:runtime],
    r_10_1_s4_br[3]["run_log"][!,:gap],
    label = "breadthfirst ([4])",
    color = :blue,
    style = :dash,
)
plot!(
    r_10_1_d[3]["run_log"][!,:runtime],
    r_10_1_d[3]["run_log"][!,:gap],
    label = "depthfirst",
    color = :black
)
plot!(
    r_10_1_bd_100[3]["run_log"][!,:runtime],
    r_10_1_bd_100[3]["run_log"][!,:gap],
    label = "bestfirst (cutoff 100)",
    color = :orange, 
    style = :dash
)
plot!(
    r_10_1_bd_300[3]["run_log"][!,:runtime],
    r_10_1_bd_300[3]["run_log"][!,:gap],
    label = "bestfirst (cutoff 300)",
    color = :orange, 
    style = :dash
)
plot!(
    r_10_1_bd_1000[3]["run_log"][!,:runtime],
    r_10_1_bd_1000[3]["run_log"][!,:gap],
    label = "bestfirst (cutoff 1000)",
    color = :orange, 
    style = :dash
)
r_10_1_bd_100
r_10_1_bd_300
r_10_1_bd_1000


r_10_1[3]["run_log"]
r_10_1_br[3]["run_log"]

# 11: checking errors in mc1
include("test_matrix_completion_disjunctivecuts.jl")
include("matrix_completion.jl")

using .TestMatrixCompletionDisjunctiveCuts
using .TestMatrixCompletionNonDisjunctiveCuts

mc1_params = [
    (1, 10, 2.0,  7, 0.1, 20.0, false, "depthfirst", true),
    (1, 10, 3.0,  8, 0.1, 20.0, false, "depthfirst", true),
    (1, 10, 3.0, 10, 0.1, 80.0, false, "depthfirst", true),
    (1, 10, 2.0, 11, 0.1, 80.0, false, "depthfirst", true),
    (1, 10, 2.0, 19, 0.1, 80.0, false, "depthfirst", true),
    (1, 10, 3.0, 20, 0.1, 80.0, false, "depthfirst", true),
    (1, 10, 3.0,  2, 0.1, 80.0,  true, "depthfirst", true),
    (1, 10, 2.0,  3, 0.1, 80.0,  true, "depthfirst", true),
    (1, 10, 2.0, 11, 0.1, 80.0,  true, "depthfirst", true),
    (1, 10, 3.0, 12, 0.1, 80.0,  true, "depthfirst", true),
]
for param in mc1_params
    println(param)
    println()
    (k, n, p, seed, noise, γ, use_disjunctive_cuts, node_selection, altmin_flag) = param
    num_indices = Int(ceil(p * k * n * log10(n)))
    time_limit = 120
    if use_disjunctive_cuts
        test_matrix_completion_disjunctivecuts(
            k, n, n, num_indices, seed, noise, γ,
            ;
            node_selection = node_selection,
            disjunctive_cuts_type = "linear",
            disjunctive_cuts_breakpoints = "smallest_1_eigvec",
            add_Shor_valid_inequalities = false,
            time_limit = time_limit,
            root_only = false,
            with_log = false,
            altmin_flag = altmin_flag,
        )
    else
        test_matrix_completion_nondisjunctivecuts(
            k, n, n, num_indices, seed, noise, γ,
            ;
            node_selection = node_selection,
            time_limit = time_limit, 
            root_only = false,
            with_log = false,
            altmin_flag = altmin_flag,
        )
    end
end

test_matrix_completion_nondisjunctivecuts(
    1, 10, 10, Int(ceil(2.0 * 1 * 10 * log10(10))), 7, 0.1, 20.0;
    
)
