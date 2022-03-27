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
    relaxation::String = "SDP",
    branching_region::String = "angular",
    solver_output::Int = 1,
)
    if !(relaxation in ["SDP", "SOCP"])
        error("""
        Invalid input for relaxation method.
        Relaxation must be either "SDP" or "SOCP".
        """)
    end
    if !(branching_region in ["box", "angular", "polyhedral", "hybrid"])
        error("""
        Invalid input for branching region.
        Branching region must be either "box" or "angular" or "polyhedral" or "hybrid"; $branching_region supplied instead.
        """)
    end
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
    A = randn(Float64, (n, k)) * randn(Float64, (k, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    return relax_frob_matrixcomp(
        n, k, relaxation, branching_region,
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

r = test_relax_frob_matrixcomp(
    1, 4, 5, 10, 1, 
    U_lower, U_upper,
)

raw_status(r["model"])


primal_status(r["model"])

has_values(r["model"])

objective_value(r["model"])
dual_objective_value(r["model"])

primal_feasibility_report(r["model"])

include("matrix_completion.jl");

# example with solution infeasible for original
result = test_relax_frob_matrixcomp(
    1,3,4,8,0,
    -ones(4,1), ones(4,1),
    ;
    relaxation = "SDP",
    solver_output=0,
);
@test(!master_problem_frob_matrixcomp_feasible(
    result["Y"],
    result["U"],
    result["t"],
    result["X"],
    result["Θ"],
))

# example with solution feasible for original
result = test_relax_frob_matrixcomp(
    1,3,4,8,0,
    fill(0.5,(4,1)), ones(4,1),
    ;
    relaxation = "SDP",
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
    relaxation = "SDP",
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
    relaxation = "SOCP",
)
result = test_relax_frob_matrixcomp(
    1,3,4,8,1,-ones(4,1), ones(4,1); γ = 10.0,
    relaxation = "SDP",
)

tr(result["Y"])

result = test_relax_frob_matrixcomp(
    1,5,6,15,5,-ones(6,1), ones(6,1); γ = 10.0,
    relaxation = "SOCP",
)
result = test_relax_frob_matrixcomp(
    1,5,6,15,5,-ones(6,1), ones(6,1); γ = 10.0,
    relaxation = "SDP",
)

tr(result["Y"])

result = test_relax_frob_matrixcomp(
    1,6,7,21,5,-ones(7,1), ones(7,1); γ = 10.0,
    relaxation = "SOCP",
)
result = test_relax_frob_matrixcomp(
    1,6,7,21,5,-ones(7,1), ones(7,1); γ = 10.0,
    relaxation = "SDP",
)

result = test_relax_frob_matrixcomp(
    1,10,15,75,1,-ones(15,1), ones(15,1); γ = 10.0,
    relaxation = "SOCP",
)
result = test_relax_frob_matrixcomp(
    1,10,15,75,1,-ones(15,1), ones(15,1); γ = 10.0,
    relaxation = "SDP",
)

result = test_relax_frob_matrixcomp(
    1,20,25,250,1,-ones(25,1), ones(25,1); γ = 10.0,
    relaxation = "SOCP",
)
result = test_relax_frob_matrixcomp(
    1,20,25,250,1,-ones(25,1), ones(25,1); γ = 10.0,
    relaxation = "SDP",
)

result = test_relax_frob_matrixcomp(
    1,30,40,600,1,-ones(40,1), ones(40,1); γ = 10.0,
    relaxation = "SOCP",
)
result = test_relax_frob_matrixcomp(
    1,30,40,600,1,-ones(40,1), ones(40,1); γ = 10.0,
    relaxation = "SDP",
)

result = test_relax_frob_matrixcomp(
    1,40,50,1000,1,-ones(50,1), ones(50,1); γ = 10.0,
    relaxation = "SOCP",
)
result = test_relax_frob_matrixcomp(
    1,40,50,1000,1,-ones(50,1), ones(50,1); γ = 10.0,
    relaxation = "SDP",
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
    A = randn(Float64, (n, k)) * randn(Float64, (k, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    return alternating_minimization(
        A, k, indices, γ, λ,
    )
end

U, V = test_alternating_minimization(1,3,4,8,1)
@test(rank(U * V) == 1)

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
    relaxation::String = "SDP",
    branching_region::String = "angular",
    branching_type::String = "lexicographic",
    root_only::Bool = false,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
    with_log::Bool = true,
)
    if !(relaxation in ["SDP", "SOCP"])
        error("""
        Invalid input for relaxation method.
        Relaxation must be either "SDP" or "SOCP".
        """)
    end
    if !(branching_region in ["box", "angular", "polyhedral", "hybrid"])
        error("""
        Invalid input for branching region.
        Branching region must be either "box" or "angular" or "polyhedral" or "hybrid"; $branching_region supplied instead.
        """)
    end
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
    A = randn(Float64, (n, k)) * randn(Float64, (k, m))
    indices = generate_masked_bitmatrix(n, m, n_indices)

    log_time = Dates.now()
    solution, printlist, instance = branchandbound_frob_matrixcomp(
        k,
        A,
        indices,
        γ,
        λ,
        ;
        relaxation = relaxation,
        branching_region = branching_region,
        branching_type = branching_type,
        root_only = root_only,
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
    relaxation_ranges = ["SDP", "SOCP"],
    branching_region_ranges = ["angular", "box", "polyhedral", "hybrid"],
    max_steps = 10000,
    time_limit = 3600,
)
    for (seed, λ, relaxation, branching_region, γ) in Iterators.product(
        seeds,
        λ_ranges,
        relaxation_ranges,
        branching_region_ranges,
        γ_ranges, 
    )
        try
            solution, printlist, instance = test_branchandbound_frob_matrixcomp(
                k, m, n, n_indices, seed,
                γ = γ, λ = λ, 
                relaxation = relaxation, 
                branching_region = branching_region,
                max_steps = max_steps, 
                time_limit = time_limit,
            )
            filename = "$(branching_region)_$(relaxation)_$(k)_$(m)_$(n)_$(n_indices)_$(seed)_$(γ)_$(λ).csv"
            run_log_filepath = "logs/run_logs/" * filename
            CSV.write(run_log_filepath, instance["run_log"])
        catch e
            continue
        end
    end
end

run_experiments(
    1, 4, 5, 10, [1];
    γ_ranges = [1.0],
    λ_ranges = [1.0],
    relaxation_ranges = ["SDP"],
    branching_region_ranges = ["angular"],
    max_steps = 100000,
    time_limit = 240,
)

run_experiments(
    1, 4, 5, 10, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    relaxation_ranges = ["SDP"],
    branching_region_ranges = ["angular", "polyhedral"],
    max_steps = 100000,
    time_limit = 240,
)
run_experiments(
    1, 5, 6, 15, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    relaxation_ranges = ["SDP"],
    branching_region_ranges = ["angular", "polyhedral"],
    max_steps = 100000,
    time_limit = 240,
)

run_experiments(
    1, 6, 7, 21, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    relaxation_ranges = ["SDP"],
    branching_region_ranges = ["angular", "polyhedral"],
    max_steps = 100000,
    time_limit = 480,
)

run_experiments(
    1, 10, 20, 100, 0:4;
    γ_ranges = [1.0],
    λ_ranges = [1.0],
    relaxation_ranges = ["SDP"],
    branching_region_ranges = ["angular", "polyhedral"],
    max_steps = 1000000,
    time_limit = 1200,
)

run_experiments(
    1, 4, 5, 10, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    relaxation_ranges = ["SDP"],
    branching_region_ranges = ["angular", "box"],
    max_steps = 100000,
    time_limit = 240,
)

run_experiments(
    1, 5, 6, 15, 0:4;
    γ_ranges = [1.0, 2.0, 0.5, 5.0, 0.2],
    λ_ranges = [1.0],
    relaxation_ranges = ["SDP"],
    branching_region_ranges = ["angular", "box"],
    max_steps = 100000,
    time_limit = 600,
)

function test_angular_box_branchandbound_frob_matrixcomp(
    k, m, n, n_indices, seeds,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 0.0,
    relaxation::String = "SDP",
    max_steps::Int = 10000,
    time_limit::Int = 3600,
)
    for seed in seeds
        angular_results = test_branchandbound_frob_matrixcomp(
            k, m, n, n_indices, seed,
            γ = γ, λ = λ,
            time_limit = time_limit, max_steps = max_steps,
            relaxation = relaxation,
            branching_region = "angular", root_only = false,
        )
        angular_run_log = angular_results[3]["run_log"]
        CSV.write("logs/angular_box_time_gap/angular_$(relaxation)_$(k)_$(m)_$(n)_$(n_indices)_$(seed)_$(γ)_$(λ).csv", angular_run_log)
        box_results = test_branchandbound_frob_matrixcomp(
            k, m, n, n_indices, seed,
            γ = γ, λ = λ,
            time_limit = time_limit, max_steps = max_steps,
            relaxation = relaxation,
            branching_region = "box", root_only = false,
        )
        box_run_log = box_results[3]["run_log"]
        CSV.write("logs/angular_box_time_gap/box_$(relaxation)_$(k)_$(m)_$(n)_$(n_indices)_$(seed)_$(γ)_$(λ).csv", box_run_log)

        plot(fmt = :png, yaxis = :log)
        plot!(angular_run_log[:,"runtime"], angular_run_log[:, "gap"], label = "angular")
        plot!(box_run_log[:,"runtime"], box_run_log[:, "gap"], label = "box")
        xlabel!("Time (s)")
        ylabel!("Optimality gap (%)")
        title!("k = $k, m = $m, n = $n, n_indices = $n_indices, seed = $seed, γ = $γ, λ = $λ, $relaxation")
        savefig("plots/angular_box_time_gap_$(relaxation)_$(k)_$(m)_$(n)_$(n_indices)_$(seed)_$(γ)_$(λ).png")
    end
end

test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 10.0, λ = 1.0,
    relaxation = "SDP",
    time_limit = 120, max_steps = 10000,
)
test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 2.0, λ = 1.0,
    relaxation = "SDP",
    time_limit = 120, max_steps = 10000,
)
test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 1.0, λ = 1.0,
    relaxation = "SDP",
    time_limit = 120, max_steps = 10000,
)

test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 0.5, λ = 1.0,
    relaxation = "SDP",
    time_limit = 120, max_steps = 10000,
)
test_angular_box_branchandbound_frob_matrixcomp(
    1, 4, 5, 10, [0,1,2,3,4],
    γ = 0.1, λ = 1.0,
    relaxation = "SDP",
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


result = test_branchandbound_frob_matrixcomp(2,10,10,50,0, λ = 0.0, branching_region = "box", branching_type = "gradient", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,10,10,30,0, λ = 0.0, γ = 20 / 0.3, branching_region = "hybrid", time_limit = 60, max_steps = 10000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,10,10,30,0, λ = 0.0, γ = 20 / 0.3, branching_region = "box", branching_type = "gradient", time_limit = 60, max_steps = 10000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, λ = 0.0, branching_region = "hybrid", time_limit = 60, max_steps = 10000, root_only = false)


result = test_branchandbound_frob_matrixcomp(1,5,6,15,16, branching_region = "hybrid", time_limit = 10, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,6,7,21,14, branching_region = "hybrid", time_limit = 10, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,7,8,21,8, branching_region = "hybrid", time_limit = 10, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,10,10,30,10, branching_region = "hybrid", time_limit = 120, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(2,15,15,67,10, branching_region = "hybrid", time_limit = 600, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,12,12,36,5, branching_region = "hybrid", time_limit = 120, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,12,12,36,5, branching_region = "hybrid", time_limit = 120, max_steps = 100000, root_only = false)

result = test_branchandbound_frob_matrixcomp(1,4,5,10,4, branching_region = "angular", time_limit = 120, max_steps = 100000, root_only = false)
result = test_branchandbound_frob_matrixcomp(1,4,5,10,0, branching_region = "box", time_limit = 120, max_steps = 10000, root_only = false)

angular_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "angular", time_limit = 120, max_steps = 10000, root_only = false)
angular_run_log = angular_result[3]["run_log"]
plot(angular_run_log[:,"runtime"], angular_run_log[:, "gap"], yaxis=:log)

include("matrix_completion.jl")

hybrid_SDP_result[3]["run_details"]



hybrid_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "hybrid", relaxation = "SDP", time_limit = 1200, max_steps = 20000, root_only = false)
polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "polyhedral", relaxation = "SDP", time_limit = 1200, max_steps = 20000, root_only = false)
angular_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "angular", relaxation = "SDP", time_limit = 1200, max_steps = 20000, root_only = false)
box_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "box", relaxation = "SDP", time_limit = 1200, max_steps = 20000, root_only = false)



polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,5,6,15,1, branching_region = "polyhedral", relaxation = "SDP", time_limit = 120, max_steps = 10000, root_only = false)
polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,6,7,21,1, branching_region = "polyhedral", relaxation = "SDP", time_limit = 240, max_steps = 10000, root_only = false)
polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,7,8,28,2, branching_region = "polyhedral", relaxation = "SDP", time_limit = 480, max_steps = 10000, root_only = false)




polyhedral_SOCP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "polyhedral", relaxation = "SOCP", time_limit = 120, max_steps = 10000, root_only = false)
polyhedral_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "polyhedral", relaxation = "SDP", time_limit = 120, max_steps = 10000, root_only = false)


# not converged
angular_SOCP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "angular", relaxation = "SOCP", time_limit = 120, max_steps = 10000, root_only = false)
# converged
angular_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "angular", relaxation = "SDP", time_limit = 120, max_steps = 10000, root_only = false)
# not converged
box_SOCP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "box", relaxation = "SOCP", time_limit = 120, max_steps = 10000, root_only = false)
# errored: SLOW_PROGRESS
box_SDP_result = test_branchandbound_frob_matrixcomp(1,4,5,10,1, branching_region = "box", relaxation = "SDP", time_limit = 120, max_steps = 10000, root_only = false)

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

result = test_branchandbound_frob_matrixcomp(1,3,4,8,1; time_limit = 120, max_steps = 10000, relaxation = "SOCP")

test_branchandbound_frob_matrixcomp(1,3,4,8,5; time_limit = 120, max_steps = 10000, relaxation = "SDP")
test_branchandbound_frob_matrixcomp(1,3,4,8,5; time_limit = 120, max_steps = 10000, relaxation = "SOCP")

for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,3,4,8,seed; time_limit = 120, max_steps = 10000, relaxation = "SDP")
    test_branchandbound_frob_matrixcomp(1,3,4,8,seed; time_limit = 120, max_steps = 10000, relaxation = "SOCP")
end

result = test_branchandbound_frob_matrixcomp(1,3,5,9,0, max_steps = 100000, time_limit = 120)

result = test_branchandbound_frob_matrixcomp(1,3,5,9,0, max_steps = 100000, time_limit = 120, relaxation = "SOCP")

for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,3,5,9,seed, max_steps = 100000, time_limit = 120, relaxation = "SDP")
    test_branchandbound_frob_matrixcomp(1,3,5,9,seed, max_steps = 100000, time_limit = 120, relaxation = "SOCP")
end

result = test_branchandbound_frob_matrixcomp(1,5,6,24,0, max_steps = 200000, time_limit = 600)

result = test_branchandbound_frob_matrixcomp(1,5,6,24,0, max_steps = 200000, time_limit = 600, relaxation = "SOCP")

result = test_branchandbound_frob_matrixcomp(1,10,15,50,0, max_steps = 200000, time_limit = 3600)

result = test_branchandbound_frob_matrixcomp(1,10,15,50,0, max_steps = 200000, time_limit = 3600, relaxation = "SOCP")

for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,10,15,50,seed, max_steps = 100000, time_limit = 360, relaxation = "SDP", root_only = true)
    test_branchandbound_frob_matrixcomp(1,10,15,50,seed,max_steps = 100000, time_limit = 360, relaxation = "SOCP", root_only = true)
end
for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,50,100,500,seed, max_steps = 200000, time_limit = 1200, relaxation = "SOCP", root_only = true)
    test_branchandbound_frob_matrixcomp(1,50,100,500,seed, max_steps = 200000, time_limit = 1200, relaxation = "SDP", root_only = true)
end

result = test_branchandbound_frob_matrixcomp(1,5,6,24,1, max_steps = 200000, time_limit = 300)
result = test_branchandbound_frob_matrixcomp(1,5,6,24,1, max_steps = 200000, time_limit = 300, relaxation = "SOCP")

result["U"]

result["Y"] ≈ result["U"] * result["U"]'
result["Y"] - result["U"] * result["U"]'

svd(result["Y"])

eigvals(result["Y"] - result["U"] * result["U"]')

isposdef(Symmetric(result["Y"] - result["U"] * result["U"]'))

tr(result["U"] * result["U"]')
tr(result["Y"])

result = test_branchandbound_frob_matrixcomp(1,40,50,1000,1; γ = 10.0,time_limit = 600, relaxation = "SOCP")

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

result = test_branchandbound_frob_matrixcomp(1,50,100,500,0, max_steps = 200000, time_limit = 600, relaxation = "SOCP")
result = test_branchandbound_frob_matrixcomp(1,50,100,500,0, max_steps = 200000, time_limit = 600, relaxation = "SDP")

for seed in 1:5
    test_branchandbound_frob_matrixcomp(1,50,100,500,seed, max_steps = 200000, time_limit = 600, relaxation = "SOCP")
    test_branchandbound_frob_matrixcomp(1,50,100,500,seed, max_steps = 200000, time_limit = 600, relaxation = "SDP")
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
