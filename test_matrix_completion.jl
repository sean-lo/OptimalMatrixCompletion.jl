using Test
using Random
using Plots
using CSV
using DataFrames

include("matrix_completion.jl")
include("utils.jl")

function test_relax_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    U_lower::Array{Float64,2},
    U_upper::Array{Float64,2},
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 0.0,
    branching_region::String = "angular",
    solver_output::Int = 1,
)
    if !(branching_region in ["box", "angular", "polyhedral", "hybrid"])
        error("""
        Invalid input for branching region.
        Branching region must be either "box" or "angular" or "polyhedral" or "hybrid"; $branching_region supplied instead.
        """)
    end
    (A, indices) = generate_matrixcomp_data(k, m, n, n_indices, seed)

    return relax_frob_matrixcomp(
        n, k, branching_region,
        A,
        indices,
        γ,
        λ,
        ;
        U_lower = U_lower,
        U_upper = U_upper,
        solver_output = solver_output,
    )
end

U_lower = reshape(
    [-0.11767578125, 
    -0.91943359375, 
    0.29931640625, 
    -0.22509765625, 
    0.0263671875],
    (5,1),
)
U_upper = reshape(
    [-0.1171875, 
    -0.9189453125, 
    0.2998046875, 
    -0.224609375, 
    0.02685546875],
    (5,1),
)

U_lower = reshape(
    [-0.11767578125, 
    -0.91943359375, 
    0.29931640625, 
    -0.22509765625, 
    0.0263671875] .- [0.05, 0.05, 0.05, 0.05, 0.05],
    (5,1),
)
U_upper = reshape(
    [-0.1171875, 
    -0.9189453125, 
    0.2998046875, 
    -0.224609375, 
    0.02685546875] .+ [0.05, 0.05, 0.05, 0.05, 0.05],
    (5,1),
)
U_lower = fill(-1.0, (5,1))
U_upper = fill(1.0, (5,1))

γ = 1.0
(A, indices) = generate_matrixcomp_data(1,4,5,10,0)
r = test_relax_frob_matrixcomp(
    1, 4, 5, 10, 0, 
    U_lower, U_upper,
    γ = γ,
)
master_problem_frob_matrixcomp_feasible(
    r["Y"],
    r["U"],
    r["X"],
    r["Θ"],
)
r["U"]
r["Y"]

r["Y"] * r["Y"] - r["Y"] # Y is a projection matrix
r["U"] * r["U"]' - r["Y"] # Y != U U'

isposdef([r["Y"] r["U"]; r["U"]' I])
eigen(r["Y"])

k = 1
U_rounded = cholesky(r["Y"]).U[1:k, 1:5]'
Y_rounded = U_rounded * U_rounded'

Y_rounded - Y_rounded'

svd(r["Y"]).U[:,1:k]

eigvals(r["Y"])

solution_summary(r["model"])

function compute_α(Y, γ, A, indices)
    (n, m) = size(A)
    α = zeros(size(A))
    for j in 1:m
        for i in 1:n
            if indices[i,j] ≥ 0.5
                α[i,j] = (
                    - γ * sum(
                        Y[i,l] * A[l,j] * indices[l,j]
                        for l in 1:n
                    )
                ) / (
                    1 + γ * sum(
                        Y[i,l] * indices[l,j]
                        for l in 1:n
                    )
                ) + A[i,j]
            end
        end    
    end
    return α
end
α = compute_α(r["Y"], γ, A, indices)
deriv_U = - γ * α * α' * r["U"]
argmin(- γ * α * α' * r["U"])

r["U"]

deriv_U_change = zeros(size(U_lower))
for i in 1:size(U_lower, 1)
    for j in 1:size(U_lower,2)
        if deriv_U[i,j] < 0.0
            deriv_U_change[i,j] = deriv_U[i,j] * (U_upper[i,j] - r["U"][i,j])
        else
            deriv_U_change[i,j] = deriv_U[i,j] * (U_lower[i,j] - r["U"][i,j])
        end
    end
end
deriv_U_change
# sanity check: deriv_U_change should be .≤ 0.0
@assert all(deriv_U_change .≤ 0.0)

eigvals(r["Y"])

isapprox.((r["U"] * r["U"]'), r["Y"]) # false
isapprox.((r["U"] * r["U"]'), r["Y"], rtol = 1e-6) # true

rank((r["U"] * r["U"]'))
rank(r["Y"]) # rank is 5 because of inexactness
inv(r["U"]' * r["U"])

α = - 1.0 * inv(r["Y"]) * r["X"]

diff, ind = findmax(abs.(α))

X = zeros(5,4)
X[ind] = diff

raw_status(r["model"])


primal_status(r["model"])

has_values(r["model"])

objective_value(r["model"])
dual_objective_value(r["model"])

primal_feasibility_report(r["model"])

include("matrix_completion.jl");

# example with solution infeasible for original
result = test_relax_frob_matrixcomp(
    1,4,5,10,0,
    fill(0.1, (5,1)), fill(0.5, (5,1)),
    ;
    solver_output=0,
)
eigvals(result["Y"])

sum(map(x -> abs(x - round(x)), eigvals(result["Y"])))

@test(master_problem_frob_matrixcomp_feasible(
    r["Y"],
    r["U"],
    r["t"],
    r["X"],
    r["Θ"],
))

# example with solution feasible for original
result = test_relax_frob_matrixcomp(
    1,3,4,8,0,
    fill(0.5,(4,1)), ones(4,1),
    ;
    solver_output=0,
);
@test(master_problem_frob_matrixcomp_feasible(
    result["Y"],
    result["U"],
    result["t"],
    result["X"],
    result["Θ"],
))

result = test_relax_frob_matrixcomp(
    2,5,7,25,0,
    -ones(7,2), ones(7,2),
    ;
    solver_output=0,
)
@test(!master_problem_frob_matrixcomp_feasible(
    result["Y"],
    result["U"],
    result["t"],
    result["X"],
    result["Θ"],
))

include("matrix_completion.jl")

result = test_relax_frob_matrixcomp(
    1,3,4,8,1,-ones(4,1), ones(4,1); γ = 10.0,
)
result = test_relax_frob_matrixcomp(
    1,3,4,8,1,-ones(4,1), ones(4,1); γ = 10.0,
)

tr(result["Y"])

result = test_relax_frob_matrixcomp(
    1,5,6,15,5,-ones(6,1), ones(6,1); γ = 10.0,
)
result = test_relax_frob_matrixcomp(
    1,5,6,15,5,-ones(6,1), ones(6,1); γ = 10.0,
)

tr(result["Y"])

result = test_relax_frob_matrixcomp(
    1,6,7,21,5,-ones(7,1), ones(7,1); γ = 10.0,
)
result = test_relax_frob_matrixcomp(
    1,6,7,21,5,-ones(7,1), ones(7,1); γ = 10.0,
)

result = test_relax_frob_matrixcomp(
    1,10,15,75,1,-ones(15,1), ones(15,1); γ = 10.0,
)
result = test_relax_frob_matrixcomp(
    1,10,15,75,1,-ones(15,1), ones(15,1); γ = 10.0,
)

result = test_relax_frob_matrixcomp(
    1,20,25,250,1,-ones(25,1), ones(25,1); γ = 10.0,
)
result = test_relax_frob_matrixcomp(
    1,20,25,250,1,-ones(25,1), ones(25,1); γ = 10.0,
)

result = test_relax_frob_matrixcomp(
    1,30,40,600,1,-ones(40,1), ones(40,1); γ = 10.0,
)
result = test_relax_frob_matrixcomp(
    1,30,40,600,1,-ones(40,1), ones(40,1); γ = 10.0,
)

result = test_relax_frob_matrixcomp(
    1,40,50,1000,1,-ones(50,1), ones(50,1); γ = 10.0,
)
result = test_relax_frob_matrixcomp(
    1,40,50,1000,1,-ones(50,1), ones(50,1); γ = 10.0,
)

tr(result["Y"])

isposdef(result["Y"] - result["U"] * result["U"]')

function test_alternating_minimization(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 0.0,
    ϵ::Float64 = 1e-10,
    max_iters::Int = 10000,
)
    (A, indices) = generate_matrixcomp_data(k, m, n, n_indices, seed)
    altmin_A_initial = zeros(size(A))
    for i in 1:n, j in 1:m
        if indices[i,j] == 1
            altmin_A_initial[i,j] = A[i,j]
        end
    end
    altmin_U_initial, _, _ = svd(altmin_A_initial)

    return alternating_minimization(
        A, n, k, indices, γ, λ;
        U_initial = altmin_U_initial,
        ϵ = ϵ, max_iters = max_iters,
    )
end

include("matrix_completion.jl")

results = test_alternating_minimization(1,4,5,10,0)

U

results = test_alternating_minimization(2, 15, 15, 67, 10)

results = test_alternating_minimization(1,3,4,8,1)
@test(rank(results["U"] * results["V"]) == 1)

include("matrix_completion.jl")

n = 2
k = 1

φ_lower = zeros(n-1,k)
φ_upper = fill(convert(Float64, pi), (n-1,k))
result = φ_ranges_to_U_ranges(φ_lower, φ_upper) 
@test(result[1] ≈ reshape([-1.0, 0.0], (n,k)))
@test(result[2] ≈ reshape([ 1.0, 1.0], (n,k)))

n = 2
k = 1

φ_lower = fill(convert(Float64, pi) / 2, (n-1,k))
φ_upper = fill(convert(Float64, pi), (n-1,k))
result = φ_ranges_to_U_ranges(φ_lower, φ_upper) 
@test(result[1] ≈ reshape([-1.0, 0.0], (n,k)))
@test(result[2] ≈ reshape([ 0.0, 1.0], (n,k)))

n = 2
k = 1

φ_lower = fill(convert(Float64, pi) / 4, (n-1,k))
φ_upper = fill(convert(Float64, pi) / 2, (n-1,k))
result = φ_ranges_to_U_ranges(φ_lower, φ_upper) 
@test(result[1] ≈ reshape([0.0, sqrt(2)/2], (n,k)))
@test(result[2] ≈ reshape([sqrt(2)/2, 1.0], (n,k)))

n = 3
k = 1

φ_lower = zeros(n-1,k)
φ_upper = fill(convert(Float64, pi), (n-1,k))
result = φ_ranges_to_U_ranges(φ_lower, φ_upper)
@test(result[1] ≈ reshape([-1.0, -1.0, 0.0], (n,k)))
@test(result[2] ≈ reshape([ 1.0,  1.0, 1.0], (n,k)))

n = 3
k = 1

φ_lower = fill(convert(Float64, pi) / 4, (n-1,k))
φ_upper = fill(convert(Float64, pi) / 2, (n-1,k))
result = φ_ranges_to_U_ranges(φ_lower, φ_upper)
@test(result[1] ≈ reshape([0.0, 0.0, 0.5], (n,k)))
@test(result[2] ≈ reshape([sqrt(2)/2, sqrt(2)/2, 1.0], (n,k)))


function test_branchandbound_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 0.0,
    noise::Bool = false,
    ϵ::Float64 = 0.01,
    branching_region::String = "angular",
    branching_type::String = "lexicographic",
    branch_point::String = "midpoint",
    node_selection::String = "breadthfirst",
    root_only::Bool = false,
    altmin_flag::Bool = true,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
    with_log::Bool = true,
)
    if !(branching_region in ["box", "angular", "polyhedral", "hybrid"])
        error("""
        Invalid input for branching region.
        Branching region must be either "box" or "angular" or "polyhedral" or "hybrid"; $branching_region supplied instead.
        """)
    end
    if !(branching_type in ["lexicographic", "bounds", "gradient"])
        error("""
        Invalid input for branching type.
        Branching type must be either "lexicographic" or "bounds" or "gradient"; $branching_type supplied instead.
        """)
    end
    if !(branch_point in ["midpoint", "current_point"])
        error("""
        Invalid input for branch point.
        Branch point must be either "midpoint" or "current_point"; $branch_point supplied instead.
        """)
    end
    if !(node_selection in ["breadthfirst", "bestfirst", "depthfirst"])
        error("""
        Invalid input for node selection.
        Node selection must be either "breadthfirst" or "bestfirst" or "depthfirst"; $node_selection supplied instead.
        """)
    end
    (A, indices) = generate_matrixcomp_data(
        k, m, n, n_indices, seed; 
        noise = noise, ϵ = ϵ,
    )

    log_time = Dates.now()
    solution, printlist, instance = branchandbound_frob_matrixcomp(
        k,
        A,
        indices,
        γ,
        λ,
        ;
        branching_region = branching_region,
        branching_type = branching_type,
        branch_point = branch_point,
        node_selection = node_selection,
        root_only = root_only,
        altmin_flag = altmin_flag,
        max_steps = max_steps,
        time_limit = time_limit,
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

include("matrix_completion.jl")

function run_experiments(
    k, m, n, n_indices, seeds,
    ;
    γ_ranges = [1.0],
    λ_ranges = [0.0],
    branching_region_ranges = ["angular", "box", "polyhedral", "hybrid"],
    max_steps = 10000,
    time_limit = 3600,
)
    all_run_details = DataFrame(
        k = Int[],
        m = Int[],
        n = Int[],
        n_indices = Int[],
        seed = Int[],
        γ = Float64[],
        λ = Float64[],
        branching_region = String[],
        max_steps = Int[],
        time_limit = Int[],
        log_time = [],
        start_time = [],
        end_time = [],
        time_taken = [],
        solve_time_altmin = [],
        solve_time_relaxation_feasibility = [],
        solve_time_relaxation = [],
        solve_time_U_ranges = [],
        solve_time_polyhedra = [],
        nodes_explored = [],
        nodes_total = [],
        nodes_relax_infeasible = [],
        nodes_relax_feasible = [],
        nodes_relax_feasible_pruned = [],
        nodes_master_feasible = [],
        nodes_master_feasible_improvement = [],
        nodes_relax_feasible_split = [],
    )
    for (seed, λ, branching_region, γ) in Iterators.product(
        seeds,
        λ_ranges,
        branching_region_ranges,
        γ_ranges, 
    )
        try
            solution, printlist, instance = test_branchandbound_frob_matrixcomp(
                k, m, n, n_indices, seed,
                γ = γ, λ = λ, 
                branching_region = branching_region,
                max_steps = max_steps, 
                time_limit = time_limit,
            )
            filename = "$(branching_region)_$(k)_$(m)_$(n)_$(n_indices)_$(seed)_$(γ)_$(λ).csv"
            run_log_filepath = "logs/run_logs/" * filename
            CSV.write(run_log_filepath, instance["run_log"])
            push!(
                all_run_details,
                [
                    k,
                    m,
                    n,
                    n_indices,
                    seed,
                    γ,
                    λ,
                    branching_region,
                    max_steps,
                    time_limit,
                    instance["run_details"]["log_time"],
                    instance["run_details"]["start_time"],
                    instance["run_details"]["end_time"],
                    instance["run_details"]["time_taken"],
                    instance["run_details"]["solve_time_altmin"],
                    instance["run_details"]["solve_time_relaxation_feasibility"],
                    instance["run_details"]["solve_time_relaxation"],
                    instance["run_details"]["solve_time_U_ranges"],
                    instance["run_details"]["solve_time_polyhedra"],
                    instance["run_details"]["nodes_explored"],
                    instance["run_details"]["nodes_total"],
                    instance["run_details"]["nodes_relax_infeasible"],
                    instance["run_details"]["nodes_relax_feasible"],
                    instance["run_details"]["nodes_relax_feasible_pruned"],
                    instance["run_details"]["nodes_master_feasible"],
                    instance["run_details"]["nodes_master_feasible_improvement"],
                    instance["run_details"]["nodes_relax_feasible_split"],
                ]
            )
        catch e
            continue
        end
    end
    return all_run_details
end

results = run_experiments(
    1, 4, 5, 10, [1];
    γ_ranges = [1.0],
    λ_ranges = [1.0],
    branching_region_ranges = ["angular"],
    max_steps = 100000,
    time_limit = 240,
)

run_experiments(
    1, 4, 5, 10, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    branching_region_ranges = ["angular", "polyhedral"],
    max_steps = 100000,
    time_limit = 240,
)
run_experiments(
    1, 5, 6, 15, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    branching_region_ranges = ["angular", "polyhedral"],
    max_steps = 100000,
    time_limit = 240,
)

run_experiments(
    1, 6, 7, 21, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    branching_region_ranges = ["angular", "polyhedral"],
    max_steps = 100000,
    time_limit = 480,
)

run_experiments(
    1, 10, 20, 100, 0:4;
    γ_ranges = [1.0],
    λ_ranges = [1.0],
    branching_region_ranges = ["angular", "polyhedral"],
    max_steps = 1000000,
    time_limit = 1200,
)

run_experiments(
    1, 4, 5, 10, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    branching_region_ranges = ["angular", "box"],
    max_steps = 100000,
    time_limit = 240,
)

run_experiments(
    1, 5, 6, 15, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    branching_region_ranges = ["angular", "box"],
    max_steps = 100000,
    time_limit = 600,
)

function test_angular_box_branchandbound_frob_matrixcomp(
    k, m, n, n_indices, seeds,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 0.0,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
)
    for seed in seeds
        angular_results = test_branchandbound_frob_matrixcomp(
            k, m, n, n_indices, seed,
            γ = γ, λ = λ,
            time_limit = time_limit, max_steps = max_steps,
            branching_region = "angular", root_only = false,
        )
        angular_run_log = angular_results[3]["run_log"]
        CSV.write("logs/angular_box_time_gap/angular_$(k)_$(m)_$(n)_$(n_indices)_$(seed)_$(γ)_$(λ).csv", angular_run_log)
        box_results = test_branchandbound_frob_matrixcomp(
            k, m, n, n_indices, seed,
            γ = γ, λ = λ,
            time_limit = time_limit, max_steps = max_steps,
            branching_region = "box", root_only = false,
        )
        box_run_log = box_results[3]["run_log"]
        CSV.write("logs/angular_box_time_gap/box_$(k)_$(m)_$(n)_$(n_indices)_$(seed)_$(γ)_$(λ).csv", box_run_log)

        plot(fmt = :png, yaxis = :log)
        plot!(angular_run_log[:,"runtime"], angular_run_log[:, "gap"], label = "angular")
        plot!(box_run_log[:,"runtime"], box_run_log[:, "gap"], label = "box")
        xlabel!("Time (s)")
        ylabel!("Optimality gap (%)")
        title!("k = $k, m = $m, n = $n, n_indices = $n_indices, seed = $seed, γ = $γ, λ = $λ")
        savefig("plots/angular_box_time_gap_$(k)_$(m)_$(n)_$(n_indices)_$(seed)_$(γ)_$(λ).png")
    end
end

test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 10.0, λ = 1.0,
    time_limit = 120, max_steps = 10000,
)
test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 2.0, λ = 1.0,
    time_limit = 120, max_steps = 10000,
)
test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 1.0, λ = 1.0,
    time_limit = 120, max_steps = 10000,
)

test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 0.5, λ = 1.0,
    time_limit = 120, max_steps = 10000,
)
test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 0.1, λ = 1.0,
    time_limit = 120, max_steps = 10000,
)

plot(
    xaxis = :log,
    yaxis = :log,
    xlabel = "gamma",
    ylabel = "Optimality gap",
    title = "Non-convergence with decreasing gamma",
    label = false
)
for seed in 6:10
    γ = 1.0
    df_γ_varying = DataFrame(
        γ = Float64[],
        converged = Bool[],
        nodes_explored = Int[],
        nodes_total = Float64[],
        final_gap = Float64[],
    )
    while true
        result = test_branchandbound_frob_matrixcomp(1,4,5,10,seed, γ = γ, branching_region = "hybrid", time_limit = 30, max_steps = 10000, root_only = false)
        if (result[3]["run_details"]["nodes_total"] - result[3]["run_details"]["nodes_explored"]) ≤ 2
            push!(
                df_γ_varying,
                [
                    γ,
                    true,
                    result[3]["run_details"]["nodes_explored"],
                    result[3]["run_details"]["nodes_total"],
                    result[3]["run_log"][end, "gap"],
                ]
            )
        elseif result[3]["run_log"][end, "gap"] ≈ result[3]["run_log"][1, "gap"]
            push!(
                df_γ_varying,
                [
                    γ,
                    false,
                    result[3]["run_details"]["nodes_explored"],
                    result[3]["run_details"]["nodes_total"],
                    result[3]["run_log"][end, "gap"],
                ]
            )
            break
        end
        γ = γ * 10^(-0.05)
    end
    df_γ_varying[:,"final_gap"] = max.(1e-6, df_γ_varying[:, "final_gap"])
    plot!(
        df_γ_varying[1:end,"γ"],
        df_γ_varying[1:end,"final_gap"],
        label = "Run $seed",
    )
end
plot!(
    legend = :topright
)



plot(
    yaxis = :log,
    xlabel = "lambda",
    ylabel = "Optimality gap",
    title = "Non-convergence with increasing lambda",
    label = false
)
for seed in 1:5
    λ = 0.0
    df_λ_varying = DataFrame(
        λ = Float64[],
        converged = Bool[],
        nodes_explored = Int[],
        nodes_total = Float64[],
        final_gap = Float64[],
    )
    while true
        result = test_branchandbound_frob_matrixcomp(1,4,5,10,seed, λ = λ, branching_region = "hybrid", time_limit = 30, max_steps = 10000, root_only = false)
        if (result[3]["run_details"]["nodes_total"] - result[3]["run_details"]["nodes_explored"]) ≤ 2
            push!(
                df_λ_varying,
                [
                    λ,
                    true,
                    result[3]["run_details"]["nodes_explored"],
                    result[3]["run_details"]["nodes_total"],
                    result[3]["run_log"][end, "gap"],
                ]
            )
        elseif result[3]["run_log"][end, "gap"] ≈ result[3]["run_log"][1, "gap"]
            push!(
                df_λ_varying,
                [
                    λ,
                    false,
                    result[3]["run_details"]["nodes_explored"],
                    result[3]["run_details"]["nodes_total"],
                    result[3]["run_log"][end, "gap"],
                ]
            )
            break
        end
        λ = λ + 0.01
    end
    df_λ_varying[:,"final_gap"] = max.(1e-6, df_λ_varying[:, "final_gap"])
    plot!(
        df_λ_varying[1:end,"λ"],
        df_λ_varying[1:end,"final_gap"],
        label = "Run $seed",
    )
end
plot!(
    legend = :topright
)


result = test_branchandbound_frob_matrixcomp(2,10,10,45,0, λ = 0.0,γ = 3.0, branching_region = "box", branching_type = "gradient", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(
    1,4,5,10,0, 
    λ = 0.0, γ = 1.0, 
    branching_region = "angular", 
    node_selection = "breadthfirst",
    noise = false,
    ϵ = 1.0,
    time_limit = 60, max_steps = 10000, root_only = false
)
result[3]["run_details"]

result = test_branchandbound_frob_matrixcomp(
    1,10,10,30,0, 
    λ = 0.0, γ = 5 / 0.3, 
    branching_region = "angular", 
    time_limit = 60, max_steps = 10000, root_only = false
)
result = test_branchandbound_frob_matrixcomp(
    1,10,10,30,0, 
    λ = 0.0, γ = 10 / 0.3, 
    branching_region = "angular", 
    time_limit = 60, max_steps = 10000, root_only = false
)
result = test_branchandbound_frob_matrixcomp(
    1,10,10,30,0, 
    λ = 0.0, γ = 20 / 0.3, 
    branching_region = "angular", 
    time_limit = 60, max_steps = 10000, root_only = false
)
result = test_branchandbound_frob_matrixcomp(1,10,10,30,0, λ = 0.0, γ = 20 / 0.3, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)

include("matrix_completion.jl")


result = test_branchandbound_frob_matrixcomp(1,12,12,36,3, λ = 0.0, γ = 40.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,8,8,24,2, λ = 0.0, γ = 40.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,10,10,40,3, λ = 0.0, γ = 40.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,10,10,40,3, λ = 0.0, γ = 40.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false, altmin_flag = false)




result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, 
branching_region = "angular",
branching_type = "bounds", 
branch_point = "midpoint",
time_limit = 60, max_steps = 10000, 
root_only = false)

result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, 
branching_region = "box", 
time_limit = 60, max_steps = 10000, 
root_only = false)

result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, 
branching_region = "box", 
branching_type = "bounds", 
branch_point = "midpoint",
time_limit = 60, max_steps = 10000, 
root_only = false)




# (1a): (depthfirst, current_point)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="depthfirst", branching_type = "gradient", branch_point = "current_point", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work

# (1b): (depthfirst, midpoint)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="depthfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work, but small difference between explored and total nodes

# (2a): (bestfirst, current_point)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="bestfirst", branching_type = "gradient", branch_point = "current_point", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work, but freq updates

# (2b): (bestfirst, midpoint)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="bestfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # works? freq updates with improvement of lower bound; no convergence

# (3a): (breadthfirst, current_point)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="breadthfirst", branching_type = "gradient", branch_point = "current_point", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work but some pruning

# (3b): (breadthfirst, midpoint)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="breadthfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work but some pruning




# (1a): (depthfirst, current_point)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", node_selection="depthfirst", branching_type = "gradient", branch_point = "current_point", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work

# (1b): (depthfirst, midpoint)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", node_selection="depthfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work, but small difference between explored and total nodes

# (2a): (bestfirst, current_point)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", node_selection="bestfirst", branching_type = "gradient", branch_point = "current_point", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work, but freq updates

# (2b): (bestfirst, midpoint)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", node_selection="bestfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # works? freq updates with improvement of lower bound; no convergence

# (3a): (breadthfirst, current_point)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", node_selection="breadthfirst", branching_type = "gradient", branch_point = "current_point", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work but some pruning

# (3b): (breadthfirst, midpoint)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", node_selection="breadthfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work but some pruning






# (1a): (depthfirst, current_point)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "box", node_selection="depthfirst", branching_type = "gradient", branch_point = "current_point", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work

# (1b): (depthfirst, midpoint)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "box", node_selection="depthfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work, but small difference between explored and total nodes

# (2a): (bestfirst, current_point)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "box", node_selection="bestfirst", branching_type = "gradient", branch_point = "current_point", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work, but freq updates

# (2b): (bestfirst, midpoint)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "box", node_selection="bestfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # works? freq updates with improvement of lower bound; no convergence

# (2b1): (bestfirst, midpoint) (γ = 30.0)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 30.0, branching_region = "box", node_selection="bestfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # works? freq updates with improvement of lower bound; no convergence

# (2b11): (bestfirst, midpoint) (γ = 30.0) (lexicographic)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 30.0, branching_region = "box", node_selection="bestfirst", branching_type = "lexicographic", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # works

# (3a): (breadthfirst, current_point)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "box", node_selection="breadthfirst", branching_type = "gradient", branch_point = "current_point", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work but some pruning

# (3b): (breadthfirst, midpoint)
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "box", node_selection="breadthfirst", branching_type = "gradient", branch_point = "midpoint", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work but some pruning


include("matrix_completion.jl")

result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="depthfirst", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work - nodes finish exploring but objective not updated
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="bestfirst", time_limit = 60, max_steps = 10000, root_only = false) # works
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="breadthfirst", time_limit = 60, max_steps = 10000, root_only = false) # works

result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", node_selection="depthfirst", time_limit = 60, max_steps = 10000, root_only = false) # doesn't work - nodes finish exploring but objective not updated
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", node_selection="bestfirst", time_limit = 60, max_steps = 10000, root_only = false) # works
result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "angular", node_selection="breadthfirst", time_limit = 60, max_steps = 10000, root_only = false) # works


result = test_branchandbound_frob_matrixcomp(1,5,5,12,2, λ = 0.0, γ = 50.0, branching_region = "hybrid", node_selection="bestfirst", time_limit = 60, max_steps = 10000, root_only = false) # works
result = test_branchandbound_frob_matrixcomp(1,5,5,12,0, λ = 0.0, γ = 30.0, branching_region = "hybrid", node_selection="breadthfirst", time_limit = 60, max_steps = 10000, root_only = false) # works


result = test_branchandbound_frob_matrixcomp(1,5,5,12,0, λ = 0.0, γ = 30.0, branching_region = "angular", node_selection="bestfirst", time_limit = 60, max_steps = 10000, root_only = false) # works!
a_be = result[3]["run_log"].objective[end]
result[3]["run_details"]

result = test_branchandbound_frob_matrixcomp(1,5,5,12,0, λ = 0.0, γ = 30.0, branching_region = "angular", node_selection="breadthfirst", time_limit = 60, max_steps = 10000, root_only = false) # works!
a_br = result[3]["run_log"].objective[end]
result[3]["run_details"]

result[1]["X"]
result[1]["objective"]

U_lower = [0.38268343236508984; 0.4899692023389578; -0.4267766952966367; -0.13845632468708977; 0.5085517315707161;;]
U_upper = [0.4713967368259978; 0.6532814824381883; -0.23864617445066644; 4.3456818641094234e-17; 0.7097037067561123;;]

result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 15.0, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 10.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 40.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 30.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 12.0, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 10.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, λ = 0.0, γ = 40.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, λ = 0.0, γ = 30.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, λ = 0.0, γ = 20.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, λ = 0.0, γ = 10.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)


result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 40.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 30.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 20.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, γ = 10.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)


result = test_branchandbound_frob_matrixcomp(1,5,6,15,0, λ = 0.0, γ = 20 / 0.5, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,5,6,15,0, λ = 0.0, γ = 20 / 0.5, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,6,7,21,0, λ = 0.0, γ = 20 / 0.5, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,8,9,27,0, λ = 0.0, γ = 20 / 0.3, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, branching_region = "angular", time_limit = 60, max_steps = 10000, root_only = false)

include("matrix_completion.jl")

result = test_branchandbound_frob_matrixcomp(1,5,6,15,16, branching_region = "hybrid", time_limit = 10, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,6,7,21,14, branching_region = "hybrid", time_limit = 10, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,7,8,21,8, branching_region = "hybrid", time_limit = 10, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,7,8,28,8, γ = 20 / 0.5, branching_region = "hybrid", time_limit = 100, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,8,9,36,0, γ = 20 / 0.5, branching_region = "hybrid", time_limit = 100, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,9,10,45,0, γ = 40.0, branching_region = "hybrid", time_limit = 100, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,9,10,45,0, γ = 30.0, branching_region = "hybrid", time_limit = 100, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,9,10,45,0, γ = 20.0, branching_region = "hybrid", time_limit = 100, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,9,10,45,0, γ = 10.0, branching_region = "hybrid", time_limit = 100, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,9,10,45,0, γ = 8.0, branching_region = "hybrid", time_limit = 100, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,9,10,45,0, γ = 5.0, branching_region = "hybrid", time_limit = 100, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,10,10,30,10, branching_region = "hybrid", time_limit = 120, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(2,15,15,67,10, branching_region = "hybrid", time_limit = 600, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(2,15,15,67,10, branching_region = "angular", time_limit = 600, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,12,12,36,5, branching_region = "hybrid", time_limit = 120, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,12,12,36,5, branching_region = "hybrid", time_limit = 120, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,4, branching_region = "angular", time_limit = 120, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

angular_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
angular_run_log = angular_result[3]["run_log"]
plot(angular_run_log[:,"runtime"], angular_run_log[:, "gap"], yaxis=:log)

include("matrix_completion.jl")

hybrid_SDP_result[3]["run_details"]



hybrid_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "hybrid", time_limit = 1200, max_steps = 20000, root_only = false)
polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "polyhedral", time_limit = 1200, max_steps = 20000, root_only = false)
angular_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "angular", time_limit = 1200, max_steps = 20000, root_only = false)
box_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "box", time_limit = 1200, max_steps = 20000, root_only = false)



polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,5,6,15,1, branching_region = "polyhedral", time_limit = 120, max_steps = 10000, root_only = false)
polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,6,7,21,1, branching_region = "polyhedral", time_limit = 240, max_steps = 10000, root_only = false)
polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,7,8,28,2, branching_region = "polyhedral", time_limit = 480, max_steps = 10000, root_only = false)




polyhedral_SOCP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "polyhedral", time_limit = 120, max_steps = 10000, root_only = false)
polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "polyhedral", time_limit = 120, max_steps = 10000, root_only = false)


# not converged
angular_SOCP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
# converged
angular_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
# not converged
box_SOCP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)
# errored: SLOW_PROGRESS
box_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

stacktrace()

box_result = test_branchandbound_frob_matrixcomp(1,4,5,10,3, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

box_run_log = box_result[3]["run_log"]
plot!(box_run_log[:,"runtime"], box_run_log[:, "gap"], yaxis=:log)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,2, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,2, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,3, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,3, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,4, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,4, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,5, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,5, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,5,6,15,0, branching_region = "angular", time_limit = 240, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,5,6,15,1, branching_region = "angular", time_limit = 240, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,5,6,15,2, branching_region = "angular", time_limit = 240, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,5,6,15,3, branching_region = "angular", time_limit = 240, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,5,6,15,4, branching_region = "angular", time_limit = 240, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,5,6,15,5, branching_region = "angular", time_limit = 240, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,5,6,15,6, branching_region = "angular", time_limit = 240, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,3,4,8,5, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,3,4,8,5, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,3,5,9,0, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,3,5,9,0, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,3,5,9,1, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,3,5,9,1, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,5,6,24,0, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,5,6,24,0, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,5,6,24,1, branching_region = "angular", time_limit = 240, max_steps = 50000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,5,6,24,1, branching_region = "box", time_limit = 240, max_steps = 50000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,6,8,24,1, branching_region = "angular", time_limit = 240, max_steps = 50000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,6,8,24,1, branching_region = "box", time_limit = 240, max_steps = 50000, root_only = false)

angular_times = Dict()
for n in [4, 5, 6, 8, 10, 12, 15, 18, 20, 24, 30, 36, 42, 50]
    try
        solution, printlist, instance = test_branchandbound_frob_matrixcomp(
            1, n, n+1, Int(n * (n+1) // 2), 0,
            γ = 0.5,
            branching_region = "angular", 
            time_limit = 3600, 
            max_steps = 1000000, 
            root_only = false,
        )
    e
    time_taken = instance["run_details"]["time_taken"]
    println("$n: $time_taken")
    angular_times[n] = time_taken
end

result = test_branchandbound_frob_matrixcomp(1,10,15,50,0, branching_region = "angular", time_limit = 600, max_steps = 50000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,10,15,50,0, branching_region = "box", time_limit = 600, max_steps = 50000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,3,4,8,1, time_limit = 120, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,3,4,8,1, time_limit = 120, max_steps = 10000)

result = test_branchandbound_frob_matrixcomp(1,3,4,8,1; time_limit = 120, max_steps = 10000)

test_branchandbound_frob_matrixcomp(1,3,4,8,5; time_limit = 120, max_steps = 10000)
test_branchandbound_frob_matrixcomp(1,3,4,8,5; time_limit = 120, max_steps = 10000)

for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,3,4,8,seed; time_limit = 120, max_steps = 10000)
    test_branchandbound_frob_matrixcomp(1,3,4,8,seed; time_limit = 120, max_steps = 10000)
end

result = test_branchandbound_frob_matrixcomp(1,3,5,9,0, max_steps = 100000, time_limit = 120)

result = test_branchandbound_frob_matrixcomp(1,3,5,9,0, max_steps = 100000, time_limit = 120)

for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,3,5,9,seed, max_steps = 100000, time_limit = 120)
    test_branchandbound_frob_matrixcomp(1,3,5,9,seed, max_steps = 100000, time_limit = 120)
end

result = test_branchandbound_frob_matrixcomp(1,5,6,24,0, max_steps = 200000, time_limit = 600)

result = test_branchandbound_frob_matrixcomp(1,5,6,24,0, max_steps = 200000, time_limit = 600)

result = test_branchandbound_frob_matrixcomp(1,10,15,50,0, max_steps = 200000, time_limit = 3600)

result = test_branchandbound_frob_matrixcomp(1,10,15,50,0, max_steps = 200000, time_limit = 3600)

for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,10,15,50,seed, max_steps = 100000, time_limit = 360, root_only = true)
    test_branchandbound_frob_matrixcomp(1,10,15,50,seed,max_steps = 100000, time_limit = 360, root_only = true)
end
for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,50,100,500,seed, max_steps = 200000, time_limit = 1200, root_only = true)
    test_branchandbound_frob_matrixcomp(1,50,100,500,seed, max_steps = 200000, time_limit = 1200, root_only = true)
end

result = test_branchandbound_frob_matrixcomp(1,5,6,24,1, max_steps = 200000, time_limit = 300)
result = test_branchandbound_frob_matrixcomp(1,5,6,24,1, max_steps = 200000, time_limit = 300)

result["U"]

result["Y"] ≈ result["U"] * result["U"]'
result["Y"] - result["U"] * result["U"]'

svd(result["Y"])

eigvals(result["Y"] - result["U"] * result["U"]')

isposdef(Symmetric(result["Y"] - result["U"] * result["U"]'))

tr(result["U"] * result["U"]')
tr(result["Y"])

result = test_branchandbound_frob_matrixcomp(1,40,50,1000,1; γ = 10.0,time_limit = 600)

k = 1
n = 50
m = 40
model = Model(Gurobi.Optimizer)
@variable(model, X[1:n, 1:m])
@variable(model, Θ[1:m, 1:m], Symmetric)

@constraint(model, 
    [i in 1:n, j in 1:m],
    [
        result["Y"][i,i] + Θ[j,j];
        result["Y"][i,i] - Θ[j,j];
        2 * X[i,j]
    ] in SecondOrderCone()
)
@constraint(model,
    [i in 1:m, j in i:m],
    [
        Θ[i,i] + Θ[j,j];
        Θ[i,i] - Θ[j,j];
        2 * Θ[i,j]
    ] in SecondOrderCone() 
)

optimize!(model)

termination_status(model)

result = test_branchandbound_frob_matrixcomp(1,50,100,500,0, max_steps = 200000, time_limit = 600)
result = test_branchandbound_frob_matrixcomp(1,50,100,500,0, max_steps = 200000, time_limit = 600)

for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,50,100,500,seed, max_steps = 200000, time_limit = 600)
    test_branchandbound_frob_matrixcomp(1,50,100,500,seed, max_steps = 200000, time_limit = 600)
end

result = test_branchandbound_frob_matrixcomp(2,5,6,24,0, max_steps = 200000, time_limit = 600)

result["U"]

result["Y"] ≈ result["U"] * result["U"]'

svd(result["Y"])

eigvals(result["Y"] - result["U"] * result["U"]')

isposdef(Symmetric(result["Y"] - result["U"] * result["U"]') + 1e-6 * I)

tr(result["U"] * result["U"]')
tr(result["Y"])


result = test_branchandbound_frob_matrixcomp(2,5,6,24,1, max_steps = 200000, time_limit = 300)

result = test_branchandbound_frob_matrixcomp(5,20,30,400,1, max_steps = 200000, time_limit = 300)

result = test_branchandbound_frob_matrixcomp(10,50,60,1400,1, max_steps = 20000000000, time_limit = 300)
