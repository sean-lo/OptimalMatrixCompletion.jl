include("test_matrix_completion_disjunctivecuts.jl")
include("utils.jl")
include("matrix_completion.jl")

using .TestMatrixCompletionNonDisjunctiveCuts
using .TestMatrixCompletionDisjunctiveCuts
using Plots
using StatsBase
using Suppressor
using CSV
using JLD
using DataFrames

r_1_br = test_matrix_completion_nondisjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "breadthfirst",
    time_limit = 180,
    use_max_steps = false,
)
r_1_b = test_matrix_completion_nondisjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    time_limit = 180,
    use_max_steps = false,
)
r_1_d = test_matrix_completion_nondisjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "depthfirst",
    time_limit = 180,
    use_max_steps = false,
)
r_1_d_br = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "breadthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 180,
    use_max_steps = false,
)
r_1_d_b = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 180,
    use_max_steps = false,
)
r_1_d_d = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "depthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 180,
    use_max_steps = false,
)

plot(
    ylim = (10^(-3), 10^(-0.5)),
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 1: (k, m, n, i, seed, γ) = (1,10,10,30,0,20)",
    legend=:right
)
plot!(
    r_1_br[3]["run_log"][!,:runtime],
    r_1_br[3]["run_log"][!,:gap],
    label = "McCormick (breadthfirst)",
    color = :red,
    style = :dash,
)
plot!(
    r_1_b[3]["run_log"][!,:runtime],
    r_1_b[3]["run_log"][!,:gap],
    label = "McCormick (bestfirst)",
    color = :red,
    style = :solid,
)
plot!(
    r_1_d[3]["run_log"][!,:runtime],
    r_1_d[3]["run_log"][!,:gap],
    label = "McCormick (depthfirst)",
    color = :red,
    style = :dot,
)
plot!(
    r_1_d_br[3]["run_log"][!,:runtime],
    r_1_d_br[3]["run_log"][!,:gap],
    label = "Eigenvalue disjunctions (breadthfirst)",
    color = :green,
    style = :dash,
)
plot!(
    r_1_d_b[3]["run_log"][!,:runtime],
    r_1_d_b[3]["run_log"][!,:gap],
    label = "Eigenvalue disjunctions (bestfirst)",
    color = :green,
    style = :solid,
)
plot!(
    r_1_d_d[3]["run_log"][!,:runtime],
    r_1_d_d[3]["run_log"][!,:gap],
    label = "Eigenvalue disjunctions (depthfirst)",
    color = :green,
    style = :dot,
)

r_1_br_noaltmin = test_matrix_completion_nondisjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "breadthfirst",
    time_limit = 180,
    use_max_steps = false,
    altmin_flag = false,
)
r_1_b_noaltmin = test_matrix_completion_nondisjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    time_limit = 180,
    use_max_steps = false,
    altmin_flag = false,
)
r_1_d_noaltmin = test_matrix_completion_nondisjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "depthfirst",
    time_limit = 180,
    use_max_steps = false,
    altmin_flag = false,
)
r_1_d_br_noaltmin = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "breadthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 180,
    use_max_steps = false,
    altmin_flag = false,
)
r_1_d_b_noaltmin = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "bestfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 180,
    use_max_steps = false,
    altmin_flag = false,
)
r_1_d_d_noaltmin = test_matrix_completion_disjunctivecuts(
    1, 10, 10, 30, 0, 0.01, 20.0;
    node_selection = "depthfirst",
    disjunctive_cuts_type = "linear",
    disjunctive_cuts_breakpoints = "smallest_1_eigvec",
    add_Shor_valid_inequalities = false,
    time_limit = 180,
    use_max_steps = false,
    altmin_flag = false,
)

plot(
    ylim = (10^(-3), 10^(-0.5)),
    yaxis=:log10,
    fmt=:png,
    ylabel="Relative gap", xlabel="Runtime (s)",
    title="Experiment 1: (k, m, n, i, seed, γ) = (1,10,10,30,0,20)\n(no alternating minimization)",
    legend=:bottomright,
)
plot!(
    r_1_br_noaltmin[3]["run_log"][!,:runtime],
    r_1_br_noaltmin[3]["run_log"][!,:gap],
    label = "McCormick (breadthfirst)",
    color = :red,
    style = :dash,
)
plot!(
    r_1_b_noaltmin[3]["run_log"][!,:runtime],
    r_1_b_noaltmin[3]["run_log"][!,:gap],
    label = "McCormick (bestfirst)",
    color = :red,
    style = :solid,
)
plot!(
    r_1_d_noaltmin[3]["run_log"][!,:runtime],
    r_1_d_noaltmin[3]["run_log"][!,:gap],
    label = "McCormick (depthfirst)",
    color = :red,
    style = :dot,
)
plot!(
    r_1_d_br_noaltmin[3]["run_log"][!,:runtime],
    r_1_d_br_noaltmin[3]["run_log"][!,:gap],
    label = "Eigenvalue disjunctions (breadthfirst)",
    color = :green,
    style = :dash,
)
plot!(
    r_1_d_b_noaltmin[3]["run_log"][!,:runtime],
    r_1_d_b_noaltmin[3]["run_log"][!,:gap],
    label = "Eigenvalue disjunctions (bestfirst)",
    color = :green,
    style = :solid,
)
plot!(
    r_1_d_d_noaltmin[3]["run_log"][!,:runtime],
    r_1_d_d_noaltmin[3]["run_log"][!,:gap],
    label = "Eigenvalue disjunctions (depthfirst)",
    color = :green,
    style = :dot,
)

## debugging
k = 1
n = 10
p = 2.0
noise = 0.1
γ = 20.0
use_disjunctive_cuts = false
node_selection = "depthfirst"
altmin_flag = true
seed = 7
num_indices = Int(round(p * n * log10(n)))
time_limit = 200

if use_disjunctive_cuts
    result = @timed test_matrix_completion_disjunctivecuts(
        k, n, n, num_indices, seed, noise, γ;
        node_selection = node_selection,
        disjunctive_cuts_type = "linear",
        disjunctive_cuts_breakpoints = "smallest_1_eigvec",
        add_Shor_valid_inequalities = false,
        time_limit = time_limit,
        root_only = false,
        with_log = true,
        altmin_flag = altmin_flag,
    )
    local r = result.value
else
    result = @timed test_matrix_completion_nondisjunctivecuts(
        k, n, n, num_indices, seed, noise, γ;
        node_selection = node_selection,
        time_limit = time_limit, 
        root_only = false,
        with_log = true,
        altmin_flag = altmin_flag,
    )
    local r = result.value
end
