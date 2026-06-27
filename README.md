# Optimal Low-Rank Matrix Completion via Branch-and-bound

`OptimalMatrixCompletion` is a Julia package which solves low-rank matrix completion problems to certifiable optimality via a custom branch-and-bound scheme. It is the implementation of the following paper: [Disjunctive Branch-and-Bound for Certifiably
Optimal Low-Rank Matrix Completion](https://arxiv.org/abs/2305.12292).

If you use this in your work, please cite this repository as follows:
    
```
@misc{
    OptimalMatrixCompletion,
    author =        {D. Bertsimas and R. Cory-Wright and S. Lo and J. Pauphilet},
    publisher =     {INFORMS Journal on Computing},
    title =         {{Disjunctive Branch-and-Bound for Certifiably
Optimal Low-Rank Matrix Completion}},
    year =          {2026},
    doi =           {10.1287/ijoc.2025.1330.cd},
    url =           {https://github.com/INFORMSJoC/2025.1330},
    note =          {Available for download at https://github.com/INFORMSJoC/2025.1330},
}
```

## Quick start

To install the package:

    julia> Pkg.install("OptimalMatrixCompletion")

To perform matrix completion on the `n`-by-`m` matrix `A` with observed indices `indices`:

    julia> using OptimalMatrixCompletion
    julia> (k, n, m) = (1, 50, 50);
    julia> A = randn(Float64, (n, m)); indices = BitMatrix(rand([0,1], (n, m)));
    julia> γ = 80.0
    julia> noise = true;
    julia> (solution, printlist, instance) = OptimalMatrixCompletion.matrix_completion_branchandbound(
        k, A, indices, γ,
        ;
        node_selection = "bestfirst",
        disjunctive_cuts_type = "linear",
        disjunctive_cuts_breakpoints = "smallest_1_eigvec",
        time_limit = 3600,
    )

Here, `solution` is a dictionary with the following fields:
- `"X_initial"`: the solution obtained after an alternating minimization procedure at the root node.
- `"Y_initial"`, `"U_initial"`: obtained from `X_initial` such that $(Y, U, X)$ are feasible for the master problem.
- `"objective_initial"`: the objective value obtained for `X_initial`.
- `"MSE_in_initial"`: the in-sample MSE of `X_initial` compared to the input matrix `A`
- `"MSE_out_initial"`: the out-of-sample MSE of `X_initial` compared to the input matrix `A`  (this is only relevant if one has access to the unobserved values in `A`)
- `"MSE_all_initial"`: the overall MSE of `X_initial` compared to the input matrix `A`  (this is only relevant if one has access to the unobserved values in `A`)
- `"X"`: the solution obtained after the branch-and-bound algorithm.
- `"Y"`, `"U"`: obtained from `X` such that $(Y, U, X)$ are feasible for the master problem.
- `"objective"`: the objective value obtained for `X`.
- `"MSE_in"`: the in-sample MSE of `X` compared to the input matrix `A`
- `"MSE_out"`: the out-of-sample MSE of `X` compared to the input matrix `A`  (this is only relevant if one has access to the unobserved values in `A`)
- `"MSE_all"`: the overall MSE of `X` compared to the input matrix `A`  (this is only relevant if one has access to the unobserved values in `A`)

`printlist` is a `Vector{String}` that contains the logged output of the algorithm. `instance` is a dictionary with the following fields:
- `"run_log"`: a log of the branch-and-bound algorithm, documenting the number of explored and total nodes, the incumbent lower and upper bounds, the current optimality gap, and the time taken so far.
- `"run_details"`: a dictionary consisting of the parameters to `matrix_completion_branchandbound()`, together with measurements on the time taken for various parts of the algorithm, and number of nodes in branch-and-bound of various categories.

## Parameters

`matrix_completion_branchandbound()` has the following required parameters:

- `k::Int`, the rank constraint on the imputed matrix $X$.
- `A::Array{Float64, 2}`, the observed data matrix $A \in \mathbb{R}^{m \times n}$.
- `indices::BitMatrix`, the observed indices in $A$ as a 0-1 matrix with 1 denoting the positions of observed entries.
- `γ::Float64`, the regularization parameter $\gamma > 0$ on the imputed matrix $X$. A larger value indicates less regularization, while a value closer to 0 indicates more regularization.

See below for a full list of optional paramters.

- `node_selection::String = "breadthfirst"`: the node selection strategy to use: either "breadthfirst" or "bestfirst" or "depthfirst" or "bestfirst_depthfirst".
- `bestfirst_depthfirst_cutoff::Int = 10000`: in the situation with `node_selection = "bestfirst_depthfirst"`, the number of nodes in the queue before the algorithm switches from `"bestfirst"` to `"depthfirst"`.
- `gap::Float64 = 1e-4`: relative optimality gap for branch-and-bound algorithm.
- `use_disjunctive_cuts::Bool = true`: whether to use eigenvector disjunctions, highly recommended to be `true`.
- `disjunctive_cuts_type::Union{String, Nothing} = nothing`: number of pieces in piecewise linear upper-approximation; either "linear" (2 pieces) or "linear2" (3 pieces) or "linear3" (4 pieces).
- `disjunctive_cuts_breakpoints::Union{String, Nothing} = nothing`: number of eigenvectors to use in constructing separation oracle, either "smallest_1_eigvec" (most negative eigenvector) or "smallest_2_eigvec" (combination of first and second most negative eigenvectors).
- `add_Shor_valid_inequalities::Bool = false`: whether to add Shor SDP inequalities to strengthen SDP relaxation at each node.
- `Shor_valid_inequalities_noisy_rank1_num_entries_present::Vector{Int} = [1, 2, 3, 4]`: if `add_Shor_valid_inequalities` is true, the set of 2-by-2 determinant minors to model with Shor SDP inequalities, based on the number of filled entries (should be some subset of `[1, 2, 3, 4]`).
- `add_Shor_valid_inequalities_fraction::Float64 = 1.0`: if `add_Shor_valid_inequalities` is true, the proportion of 2-by-2 determinant minors to model with Shor SDP inequalities.
- `add_Shor_valid_inequalities_iterative::Bool = false`: if `add_Shor_valid_inequalities` is true, whether to add them iteratively from parent node to child node.
- `max_update_Shor_indices_probability::Float64 = 1.0`: if `add_Shor_valid_inequalities_iterative` is true, the maximum probability of adding inequalities at a node.
- `min_update_Shor_indices_probability::Float64 = 0.1`, if `add_Shor_valid_inequalities_iterative` is true, the minimum probability of adding inequalities at a node.
- `update_Shor_indices_probability_decay_rate::Float64 = 1.1`: if `add_Shor_valid_inequalities_iterative` is true, the base of the exponential decay of the probability of adding inequalities at a node, as a function of depth in the tree.
- `update_Shor_indices_n_minors::Int = 100`: if `add_Shor_valid_inequalities_iterative` is true, the number of Shor SDP inequalities to add at a node whenever adding is performed.
- `root_only::Bool = false`: if true, only solves relaxation at root node
- `altmin_flag::Bool = true`: whether to perform alternating minimization at nodes in the branch-and-bound tree, highly recommended to be `true`.
- `max_altmin_probability::Float64 = 1.0`: if `altmin_flag` is true, the maximum probability of performing alternating minimization at a node.
- `min_altmin_probability::Float64 = 0.005`: if `altmin_flag` is true, the minimum probability of performing alternating minimization at a node.
- `altmin_probability_decay_rate::Float64 = 1.1`: if `altmin_flag` is true, the base of the exponential decay of the probability of performing alternating minimization at a node, as a function of depth in the tree.
- `use_max_steps::Bool = false`: whether to terminate the algorithm based on the number of branch-and-bound nodes explored.
- `max_steps::Int = 1000000`: if `use_max_steps` is true, the upper limit on number of branch-and-bound nodes explored.
- `time_limit::Int = 3600`: time limit in seconds.
- `update_step::Int = 1000`: number of branch-and-bound nodes explored per printed update.

## Usage tips

- It is highly recommended to set the parameter `use_disjunctive_cuts` and `altmin_flag` to true (the method in the paper), which implements eigenvector disjunctions and alternating minimization at branch-and-bound nodes respectively.
- The regularization parameter `γ` should be tuned according to some paramter tuning and cross-validation procedure. Bear in mind that large values of `γ`, corresponding to less regularized problems, usually corresponds to longer solution times and more nodes explored.
- Take care in deciding the `add_Shor_valid_inequalities` parameter. If set to true, convex relaxations at each node in general take much longer to solve (especially with a large set in `Shor_valid_inequalities_noisy_rank1_num_entries_present`). This can greatly increase solution time. However, depending on the sparsity and rank regime, judicious choices of `Shor_valid_inequalities_noisy_rank1_num_entries_present` can result in a much stronger relaxation that is tight, resulting in few if any branch-and-bound nodes required.
- Don't forget to set the `time_limit` to a reasonable value!
