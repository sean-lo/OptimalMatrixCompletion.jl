using Test
using Random

include("matrix_completion.jl")
include("utils.jl")

function generate_masked_bitmatrix(
    dim1::Int,
    dim2::Int,
    sparsity::Int,
)
    index_pairs = randperm(dim1 * dim2)[1:sparsity]
    index_vec = zeros(dim1 * dim2)
    index_vec[index_pairs] .= 1.0
    return reshape(index_vec, (dim1, dim2))
end

function test_SDP_relax_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    U_lower::Array{Float64,2},
    U_upper::Array{Float64,2},
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 1.0,
    solver_output::Int = 1,
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

    return SDP_relax_frob_matrixcomp(
        U_lower,
        U_upper,
        A,
        indices,
        γ,
        λ,
        ;
        solver_output = solver_output,
    )
end

include("matrix_completion.jl");

# example with solution infeasible for original
result = test_SDP_relax_frob_matrixcomp(
    1,3,4,8,0,
    -ones(4,1), ones(4,1),
    ; 
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
result = test_SDP_relax_frob_matrixcomp(
    1,3,4,8,0,
    fill(0.5,(4,1)), ones(4,1),
    ; 
    solver_output=0,
)
@test(master_problem_frob_matrixcomp_feasible(
    result["Y"],
    result["U"],
    result["t"],
    result["X"],
    result["Θ"],
))

result = test_SDP_relax_frob_matrixcomp(
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

function test_SOCP_relax_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    U_lower::Array{Float64,2},
    U_upper::Array{Float64,2},
    ;
    γ::Float64 = 10.0,
    λ::Float64 = 1.0,
    solver_output::Int = 1,
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

    return SOCP_relax_frob_matrixcomp(
        U_lower,
        U_upper,
        A,
        indices,
        γ,
        λ,
        ;
        solver_output = solver_output,
    )
end

include("matrix_completion.jl")

result = test_SOCP_relax_frob_matrixcomp(
    1,3,4,8,1,-ones(4,1), ones(4,1); γ = 10.0,
)
result = test_SDP_relax_frob_matrixcomp(
    1,3,4,8,1,-ones(4,1), ones(4,1); γ = 10.0,
)

tr(result["Y"])

result = test_SOCP_relax_frob_matrixcomp(
    1,5,6,15,5,-ones(6,1), ones(6,1); γ = 10.0,
)
result = test_SDP_relax_frob_matrixcomp(
    1,5,6,15,5,-ones(6,1), ones(6,1); γ = 10.0,
)

tr(result["Y"])

result = test_SOCP_relax_frob_matrixcomp(
    1,6,7,21,5,-ones(7,1), ones(7,1); γ = 10.0,
)
result = test_SDP_relax_frob_matrixcomp(
    1,6,7,21,5,-ones(7,1), ones(7,1); γ = 10.0,
)

result = test_SOCP_relax_frob_matrixcomp(
    1,10,15,75,1,-ones(15,1), ones(15,1); γ = 10.0,
)
result = test_SDP_relax_frob_matrixcomp(
    1,10,15,75,1,-ones(15,1), ones(15,1); γ = 10.0,
)

result = test_SOCP_relax_frob_matrixcomp(
    1,20,25,250,1,-ones(25,1), ones(25,1); γ = 10.0,
)
result = test_SDP_relax_frob_matrixcomp(
    1,20,25,250,1,-ones(25,1), ones(25,1); γ = 10.0,
)

result = test_SOCP_relax_frob_matrixcomp(
    1,30,40,600,1,-ones(40,1), ones(40,1); γ = 10.0,
)
result = test_SDP_relax_frob_matrixcomp(
    1,30,40,600,1,-ones(40,1), ones(40,1); γ = 10.0,
)

result = test_SOCP_relax_frob_matrixcomp(
    1,40,50,1000,1,-ones(50,1), ones(50,1); γ = 10.0,
)
result = test_SDP_relax_frob_matrixcomp(
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

    return alternating_minimization(
        A, k, indices, γ, λ,
    )
end

U, V = test_alternating_minimization(1,3,4,8,1)
@test(rank(U * V) == 1)

function test_branchandbound_frob_matrixcomp(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    γ::Float64 = 1.0,
    λ::Float64 = 1.0,
    relaxation::String = "SDP",
    root_only::Bool = false,
    max_steps::Int = 10000,
    time_limit::Int = 3600,
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

    return branchandbound_frob_matrixcomp(
        k,
        A,
        indices,
        γ,
        λ,
        ;
        relaxation = relaxation,
        root_only = root_only,
        max_steps = max_steps,
        time_limit = time_limit,
    )
end

include("matrix_completion.jl")


result = test_branchandbound_frob_matrixcomp(1,3,4,8,1, time_limit = 120, max_steps = 10000, root_only = true)

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
