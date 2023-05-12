using LinearAlgebra
using Random
using Compat

using JuMP
using MathOptInterface
using Mosek
using MosekTools

function generate_masked_bitmatrix(
    dim1::Int,
    dim2::Int,
    sparsity::Int,
    seed::Int,
)
    Random.seed!(seed)
    while true
        index_pairs = randperm(dim1 * dim2)[1:sparsity]
        index_vec = falses(dim1 * dim2)
        index_vec[index_pairs] .= true
        indices = reshape(index_vec, (dim1, dim2))
        if (
            all(any(indices, dims=1))
            && all(any(indices, dims=2))
        )
            return indices
        end
    end
end

function generate_sparse_masked_bitmatrix(
    dim1::Int,
    dim2::Int,
    sparsity::Int,
    seed::Int,
)
    Random.seed!(seed)
    indices = falses(dim1, dim2)
    # Stage 1: sample max(dim1, dim2) entries such that each row and column has at least 1 entry
    n_filled = max(dim1, dim2)
    perm = randperm(n_filled)
    if dim1 == dim2
        for i in 1:dim1
            indices[i,perm[i]] = true
        end
    elseif dim1 < dim2
        for j in 1:dim2
            if perm[j] > dim1
                indices[rand(1:dim1),j] = true
            else
                indices[perm[j],j] = true
            end
        end
    elseif dim1 > dim2
        for i in 1:dim1
            if perm[i] > dim2
                indices[i,rand(1:dim2)] = true
            else
                indices[i,perm[i]] = true
            end
        end
    end
    # Stage 2: sample uniformly at random 
    # (sparsity - max(dim1, dim2)) entries
    # from all other possibilities
    options = setdiff(1:(dim1*dim2), findall(reshape(indices, (dim1*dim2))))
    indices[shuffle(options)[1:(sparsity - n_filled)]] .= true
    return indices
end

function generate_matrixcomp_data(
    k::Int,
    m::Int,
    n::Int,
    n_indices::Int,
    seed::Int,
    ;
    noise::Bool = false,
    ϵ::Float64 = 0.01,
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
    if (n > 1000) || (m > 1000)
        error("""
        Currently does not support n > 1000, m > 1000. 
        Supplied n = $n, m = $m.
        """)
    end
    # 4 sources of randomness
    seeds = abs.(rand(MersenneTwister(seed), Int, 4)) 
    A_left = randn(MersenneTwister(seeds[1]), Float64, (1000, k))[1:n, :]
    A_right = randn(MersenneTwister(seeds[2]), Float64, (k, 1000))[:, 1:m]
    A = A_left * A_right
    if noise
        A_noise = randn(MersenneTwister(seeds[3]), Float64, (1000, 1000))[1:n, 1:m]
        A = A + ϵ * A_noise
    end
    if (n + m) * k ≤ n_indices < Int(ceil((n + m) * k * log10(min(n, m))))
        indices = generate_sparse_masked_bitmatrix(n, m, n_indices, seeds[4])
    else
        indices = generate_masked_bitmatrix(n, m, n_indices, seeds[4])
    end
    return A, indices
end