struct ResponseComplementSchurResult{RC,SM}
    response_complement::RC
    selected_rank::Int
    selected_capture::Float64
    schur_model::SM
end

"""
    build_response_complement_schur_model(responses, W, B, D;
                                          capture_target=nothing,
                                          naux=nothing)

Build an algebraic response-completeness model by projecting `responses`
outside the orthonormal Wannier subspace `W`, compressing the residual, and
using the selected vectors to reduce the Schur auxiliary space. Exactly one
of `capture_target` and `naux` must be supplied. This is an algebra API only:
it does not create physical Wannier/MRWF data or a DFTK adapter. The
`responses` and `W` matrices must use the same ambient row-space basis
representation as the eliminated space represented by `D`; in particular,
their row count must equal `size(D, 1)`, and `size(B, 2)` must equal that
same eliminated-space dimension.
"""
function build_response_complement_schur_model(
        responses::AbstractMatrix, W::AbstractMatrix,
        B::AbstractMatrix, D::AbstractMatrix;
        capture_target::Union{Nothing,Real}=nothing,
        naux::Union{Nothing,Integer}=nothing)
    isnothing(capture_target) == isnothing(naux) &&
        throw(ArgumentError("specify exactly one of capture_target and naux"))
    size(responses, 1) == size(W, 1) ||
        throw(DimensionMismatch("responses and W must have the same row count"))
    size(D, 1) == size(D, 2) ||
        throw(DimensionMismatch("D must be square"))
    size(B, 2) == size(D, 1) ||
        throw(DimensionMismatch("B columns and D dimensions do not match"))
    size(responses, 1) == size(D, 1) ||
        throw(DimensionMismatch(
            "response residual space and Schur eliminated space must have the same dimension",
        ))
    residuals = build_response_residual(responses, W)
    complement = compress_response_residual(residuals)
    selected_rank = if isnothing(capture_target)
        naux
    else
        select_auxiliary_rank(complement; capture_target)
    end
    selected_complement = compress_response_residual(residuals; naux=selected_rank)
    schur_model = reduce_schur(B, D, selected_complement.basis)
    selected_capture = selected_rank == 0 ? 0.0 :
        selected_complement.cumulative_capture[selected_rank]
    ResponseComplementSchurResult(
        selected_complement,
        selected_rank,
        selected_capture,
        schur_model,
    )
end
