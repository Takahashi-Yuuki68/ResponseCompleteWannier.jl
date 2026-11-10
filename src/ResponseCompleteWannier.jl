module ResponseCompleteWannier

using LinearAlgebra

include("schur/schur_operator.jl")
include("complement/response_residual.jl")
include("comparison/auxiliary_convergence.jl")
include("comparison/dielectric_reference.jl")
include("comparison/response_comparison.jl")

export SchurAuxiliaryModel,
       ResponseComplement,
       AuxiliaryConvergenceResult,
       DielectricSpectrum,
       DielectricComparison,
       ResponseComplementSchurResult,
       auxiliary_solve,
       build_response_residual,
       build_response_complement_schur_model,
       compress_response_residual,
       compare_dielectric_spectra,
       evaluate_auxiliary_convergence,
       full_block_solve,
       linearized_matrix,
       load_pw_reference,
       reduce_schur,
       reduced_schur_solve,
       select_auxiliary_rank,
       schur_self_energy,
       schur_solve,
       write_comparison_table

end # module ResponseCompleteWannier
