include("test_matrix_completion_disjunctivecuts.jl")

using .TestMatrixCompletionDisjunctiveCuts
using Plots
using StatsBase
using Suppressor
using CSV
using JLD


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

r_6_2t[3]["run_log"]
r_6_2f[3]["run_log"]