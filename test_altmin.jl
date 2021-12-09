using Test
using LinearAlgebra
using DataFrames
using Plots; pyplot() 

include("matrix_completion.jl")

function test_altmin(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 1.0,
)
    if n_indices < (n + m) * k
        error("""
        System is under-determined. 
        n_indices must be at least (n + m) * k.
        """)
    end
    if n_indices > n * m
        error("""
        Cannot generate random indices of length more than the size of matrix A.
        """)
    end
    Random.seed!(seed)
    A = randn(Float64, (n, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    S = svd(A).S

    U_altmin, V_altmin = alternating_minimization(
        A, k, indices, γ, λ,
    )
    X_initial = U_altmin * V_altmin
    U_initial, S_initial, V_initial = svd(X_initial) # TODO: implement truncated SVD
    U_initial = U_initial[:,1:k]
    Y_initial = U_initial * U_initial'
    objective_initial = objective_function(
        X_initial, A, indices, U_initial, γ, λ,
    )

    U_lower_initial = -ones(n, k)
    U_upper_initial = ones(n, k)

    SDP_result = SDP_relax_frob_matrixcomp(U_lower_initial, U_upper_initial, A, indices, γ, λ)
    objective_SDP = SDP_result["objective"]

    return objective_SDP, objective_initial
end

function get_altmin_boundsgap(k, m, n, n_indices, n_seeds)
    results = DataFrame(lower_bound = [], feasible_soln = [], bound_gap = [])
    for seed in 1:n_seeds
        lower_bound, feasible_soln = test_altmin(k,m,n,n_indices,seed)
        push!(results, [lower_bound, feasible_soln, (feasible_soln - lower_bound) / lower_bound])
    end
    return results
end

results


n_seeds = 1000
params = [(1, 8, 10, 60), (2, 8, 10, 60), (3, 8, 10, 60)]
for (k, m, n, n_indices) in params
    results = get_altmin_boundsgap(k, m, n, n_indices, n_seeds)
    histogram(
        results[:,"bound_gap"], 
        bins=10.0 .^ (-5:0), 
        xaxis=(:log10, (0.00001, 10)), 
        fmt = :png,
        title = "Plot of bounds gap (proportion)\n for k = $k, m = $m, n = $n, n_indices =  $n_indices",
        legend = false,
        xlabel = "bounds gap",
        ylabel = "count",
    )
    savefig("plots/altmin_boundsgap_$(k)_$(m)_$(n)_$(n_indices)_$(n_seeds).png")
end