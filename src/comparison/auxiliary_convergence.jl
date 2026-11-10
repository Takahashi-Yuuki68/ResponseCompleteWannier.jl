struct AuxiliaryConvergenceResult
    naux::Vector{Int}
    capture::Vector{Float64}
    absolute_error::Vector{Float64}
    relative_error::Vector{Float64}
end

"""
    evaluate_auxiliary_convergence(A, B, D, b, z, Qfull; ranks=0:size(Qfull, 2))

Compare exact Schur solutions with reduced auxiliary models built from the
leading columns of the orthonormal basis `Qfull`. `capture` is the fraction
of the available auxiliary basis retained at each rank. The exact solution
is computed with `schur_solve`; each approximation uses `reduce_schur` and
`reduced_schur_solve`.
"""
function evaluate_auxiliary_convergence(A, B, D, b, z, Qfull;
                                        ranks=0:size(Qfull, 2))
    _validate_schur_dimensions(A, B, D, b)
    size(Qfull, 1) == size(D, 1) ||
        throw(DimensionMismatch("Qfull and D dimensions do not match"))
    gram = adjoint(Qfull) * Qfull
    identity = Matrix{eltype(gram)}(I, size(Qfull, 2), size(Qfull, 2))
    isapprox(gram, identity; atol=1e-10, rtol=1e-10) ||
        throw(ArgumentError("columns of Qfull must be orthonormal"))

    rank_values = ranks isa Integer ? [ranks] : collect(ranks)
    all(rank -> rank isa Integer, rank_values) ||
        throw(ArgumentError("ranks must contain integers"))
    all(rank -> 0 <= rank <= size(Qfull, 2), rank_values) ||
        throw(ArgumentError("ranks must be between zero and size(Qfull, 2)"))

    exact = schur_solve(A, B, D, z, b)
    exact_norm = norm(exact)
    absolute_error = Float64[]
    relative_error = Float64[]
    for rank in rank_values
        Q = Qfull[:, 1:rank]
        model = reduce_schur(B, D, Q)
        approximate = reduced_schur_solve(A, model, z, b)
        error = norm(approximate - exact)
        push!(absolute_error, error)
        push!(relative_error, error / max(exact_norm, eps(Float64)))
    end

    AuxiliaryConvergenceResult(
        Int[rank_values...],
        Float64[rank / max(size(Qfull, 2), 1) for rank in rank_values],
        absolute_error,
        relative_error,
    )
end
