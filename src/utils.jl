using Random

function generate_masked_bitmatrix(
    n::Int,
    m::Int,
    sparsity::Int,
    seed::Int,
    ;
    max_iters::Int = 100,
)
    Random.seed!(seed)
    iter = 0
    while true
        index_pairs = randperm(n * m)[1:sparsity]
        index_vec = falses(n * m)
        index_vec[index_pairs] .= true
        indices = reshape(index_vec, (n, m))
        if (
            all(any(indices, dims=1))
            && all(any(indices, dims=2))
        ) || iter >= max_iters
            return indices
        end
        iter += 1
    end
end

function generate_sparse_masked_bitmatrix(
    n::Int,
    m::Int,
    sparsity::Int,
    seed::Int,
)
    Random.seed!(seed)
    indices = falses(n, m)
    # Stage 1: sample max(n, m) entries such that each row and column has at least 1 entry
    n_filled = max(n, m)
    perm = randperm(n_filled)
    if n == m
        for i in 1:n
            indices[i,perm[i]] = true
        end
    elseif n < m
        for j in 1:m
            if perm[j] > n
                indices[rand(1:n),j] = true
            else
                indices[perm[j],j] = true
            end
        end
    elseif n > m
        for i in 1:n
            if perm[i] > m
                indices[i,rand(1:m)] = true
            else
                indices[i,perm[i]] = true
            end
        end
    end
    # Stage 2: sample uniformly at random
    # (sparsity - max(n, m)) entries
    # from all other possibilities
    options = setdiff(1:(n*m), findall(reshape(indices, (n*m))))
    indices[shuffle(options)[1:(sparsity - n_filled)]] .= true
    return indices
end

function generate_matrix_completion_data(
    k::Int,
    n::Int,
    m::Int,
    n_indices::Int,
    seed::Int,
    ;
    n_max::Int = 10000,
    m_max::Int = 10000,
    ϵ::Float64 = 0.01,
)
    if !(n ≤ m)
        error("""
        Input matrix A must have size (n, m) with n <= m.
        n = $n, m = $m supplied instead.
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
    # 4 sources of randomness
    seeds = abs.(rand(MersenneTwister(seed), Int, 4))
    A_left = randn(MersenneTwister(seeds[1]), Float64, (n_max, k))[1:n, :]
    A_right = randn(MersenneTwister(seeds[2]), Float64, (k, m_max))[:, 1:m]
    A = A_left * A_right

    A_noise = randn(MersenneTwister(seeds[3]), Float64, (n_max, m_max))[1:n, 1:m]
    A = A + ϵ * A_noise

    if (n + m) * k ≤ n_indices < Int(ceil((n + m) * k * log10(n * m)))
        indices = generate_sparse_masked_bitmatrix(n, m, n_indices, seeds[4])
    else
        indices = generate_masked_bitmatrix(n, m, n_indices, seeds[4])
    end
    return A, indices
end
