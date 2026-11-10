using LinearAlgebra
using Test
using ResponseCompleteWannier

@testset "Schur reduction and auxiliary linearization" begin
    A = ComplexF64[2.5 0.2im; -0.2im 3.0]
    D = Diagonal([1.3, 2.1, 4.2]) |> Matrix
    B = ComplexF64[0.1 0.2im 0.005; -0.1im 0.15 0.008]
    b = ComplexF64[1.0, 0.2im]
    Q = Matrix{Float64}(I, 3, 3)[:, 1:2]
    reduced_model = reduce_schur(B, D, Q)

    @testset "full block equals exact Schur" begin
        for z in (0.2 + 0.3im, 2.8 + 0.4im)
            full_solution = full_block_solve(A, B, D, z, b)
            schur_solution = schur_solve(A, B, D, z, b)
            @test full_solution[1:size(A, 1)] ≈ schur_solution atol=1e-12
            @test (A - z * I - schur_self_energy(B, D, z)) *
                  schur_solution ≈ b atol=1e-12
        end
    end

    @testset "reduced Schur and auxiliary system agree" begin
        for z in (0.2 + 0.3im, 2.8 + 0.4im)
            reduced_solution = reduced_schur_solve(A, reduced_model, z, b)
            auxiliary_solution = auxiliary_solve(A, reduced_model, z, b)
            @test auxiliary_solution[1:size(A, 1)] ≈ reduced_solution atol=1e-12
            @test norm(reduced_solution - schur_solve(A, B, D, z, b)) < 1e-3
        end

        full_model = reduce_schur(B, D, Matrix{Float64}(I, 3, 3))
        @test linearized_matrix(A, full_model) ≈ [A B; B' D]
        for z in (0.2 + 0.3im, 2.8 + 0.4im)
            exact = schur_solve(A, B, D, z, b)
            linearized = auxiliary_solve(A, full_model, z, b)
            @test linearized[1:size(A, 1)] ≈ exact atol=1e-12
        end
    end
end

@testset "Response residual projection and SVD" begin
    W = Matrix{Float64}(I, 8, 8)[:, 1:2]
    responses = zeros(8, 4)
    responses[1:2, :] .= [1.0 0.0 2.0 1.0; 0.0 1.0 0.0 -1.0]
    responses[3, 2] = 4.0
    responses[4, 3] = 2.0
    responses[5, 4] = 1.0

    residuals = build_response_residual(responses, W)
    @test W' * residuals ≈ zeros(2, 4) atol=1e-14
    @test W * (W' * responses) + residuals ≈ responses atol=1e-14

    complement = compress_response_residual(residuals)
    @test complement.singular_values ≈ [4.0, 2.0, 1.0, 0.0]
    @test complement.cumulative_capture[1:3] ≈ [16 / 21, 20 / 21, 1.0]

    heldout = zeros(8, 1)
    heldout[3:5, 1] .= 1.0
    heldout_residual = build_response_residual(heldout, W)
    errors = [
        let basis = compress_response_residual(residuals; naux=n).basis
            norm(heldout_residual - basis * (basis' * heldout_residual))
        end
        for n in 0:3
    ]
    @test all(diff(errors) .< 0)
    @test errors[end] < 1e-14
end

@testset "Auxiliary rank selection and convergence" begin
    residuals = Diagonal([4.0, 2.0, 1.0, 0.0]) |> Matrix
    complement = compress_response_residual(residuals)
    @test select_auxiliary_rank(complement; capture_target=0.0) == 0
    @test select_auxiliary_rank(complement; capture_target=0.95) == 2
    @test select_auxiliary_rank(complement; capture_target=1.0) == 3
    @test select_auxiliary_rank(compress_response_residual(zeros(4, 2));
                                capture_target=0.95) == 0
    @test_throws ArgumentError select_auxiliary_rank(complement; capture_target=-0.1)
    @test_throws ArgumentError select_auxiliary_rank(complement; capture_target=1.1)
    @test_throws ArgumentError select_auxiliary_rank(complement; capture_target=NaN)

    A = ComplexF64[2.5 0.2im; -0.2im 3.0]
    D = Diagonal([1.3, 2.1, 4.2]) |> Matrix
    B = ComplexF64[0.1 0.2im 0.005; -0.1im 0.15 0.008]
    b = ComplexF64[1.0, 0.2im]
    Qfull = Matrix{Float64}(I, 3, 3)
    convergence = evaluate_auxiliary_convergence(
        A, B, D, b, 0.2 + 0.3im, Qfull; ranks=[0, 1, 3],
    )
    @test convergence.naux == [0, 1, 3]
    @test convergence.capture == [0.0, 1 / 3, 1.0]
    @test convergence.absolute_error[3] < 1e-12
    @test convergence.relative_error[3] < 1e-12
    @test_throws ArgumentError evaluate_auxiliary_convergence(
        A, B, D, b, 0.2 + 0.3im, Qfull; ranks=[4],
    )
    @test_throws DimensionMismatch evaluate_auxiliary_convergence(
        A, B, D, b, 0.2 + 0.3im, Matrix{Float64}(I, 2, 2),
    )
end

@testset "PW dielectric reference and comparison" begin
    reference_path = joinpath(
        @__DIR__, "..", "data", "reference", "Si", "pw_sternheimer_dense_peak.dat",
    )
    reference = load_pw_reference(reference_path)
    @test length(reference.omega_ev) == 7
    @test reference.omega_ev ≈ collect(3.0:0.1:3.6)
    peak = argmax(imag.(reference.epsilon))
    @test reference.omega_ev[peak] ≈ 3.3
    @test imag(reference.epsilon[peak]) ≈ 3.4031661882533535

    identical = compare_dielectric_spectra(reference, reference)
    @test identical.max_abs_error_real == 0.0
    @test identical.max_abs_error_imag == 0.0
    @test identical.rms_error_real == 0.0
    @test identical.rms_error_imag == 0.0
    @test identical.peak_energy_error_ev == 0.0
    @test identical.peak_height_relative_error == 0.0

    candidate = DielectricSpectrum(
        reference.omega_ev,
        reference.epsilon .+ complex.(0.1, -0.2),
        reference.loss .+ 0.01,
    )
    comparison = compare_dielectric_spectra(reference, candidate)
    @test comparison.max_abs_error_real ≈ 0.1
    @test comparison.max_abs_error_imag ≈ 0.2
    @test comparison.rms_error_real ≈ 0.1
    @test comparison.rms_error_imag ≈ 0.2
    @test_throws ArgumentError compare_dielectric_spectra(
        reference,
        DielectricSpectrum(reference.omega_ev .+ 0.01, reference.epsilon, reference.loss),
    )

    mktempdir() do directory
        output = joinpath(directory, "comparison.dat")
        @test write_comparison_table(output, reference, candidate) == output
        @test isfile(output)
        @test_throws ArgumentError write_comparison_table(
            joinpath(directory, "missing", "comparison.dat"), reference, candidate,
        )
    end
end

@testset "PW reference parser validation" begin
    mktempdir() do directory
        function write_case(name, contents)
            path = joinpath(directory, name)
            write(path, contents)
            path
        end
        @test_throws ArgumentError load_pw_reference(
            write_case("columns.dat", "1 2 3\n"),
        )
        @test_throws ArgumentError load_pw_reference(
            write_case("nan.dat", "1 2 3 4 5 NaN 7 8\n"),
        )
        @test_throws ArgumentError load_pw_reference(
            write_case("inf.dat", "1 2 3 4 5 6 Inf 8\n"),
        )
        @test_throws ArgumentError load_pw_reference(
            write_case("nonmonotone.dat",
                       "2 0 0 0 0 1 1 0\n1 0 0 0 0 1 1 0\n"),
        )
        @test_throws ArgumentError load_pw_reference(
            write_case("empty.dat", "# comment\n\n"),
        )
    end
end

@testset "Response-completeness Schur integration" begin
    W = zeros(Float64, 8, 0)
    responses = Diagonal([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]) |> Matrix
    B = ComplexF64[
        0.10 0.02im 0.03 0.04im 0.05 0.06im 0.07 0.08im;
        -0.02im 0.11 0.04im 0.05 0.06im 0.07 0.08im 0.09
    ]
    D = Diagonal([1.3, 2.1, 4.2, 5.3, 6.4, 7.5, 8.6, 9.7]) |> Matrix
    result = build_response_complement_schur_model(
        responses, W, B, D; capture_target=0.95,
    )
    residuals = build_response_residual(responses, W)
    @test size(result.response_complement.basis, 1) == 8
    @test result.response_complement.basis' * result.response_complement.basis ≈
          Matrix{Float64}(I, result.selected_rank, result.selected_rank)
    @test 1 <= result.selected_rank <= 8
    @test result.selected_capture >= 0.95
    @test size(result.schur_model.T) == (result.selected_rank, result.selected_rank)
    full_result = build_response_complement_schur_model(
        responses, W, B, D; naux=8,
    )
    @test full_result.selected_rank == 8
    @test full_result.selected_capture ≈ 1.0
    A = ComplexF64[2.5 0.2im; -0.2im 3.0]
    b = ComplexF64[1.0, 0.2im]
    z = 0.2 + 0.3im
    exact = schur_solve(A, B, D, z, b)
    reconstructed = reduced_schur_solve(A, full_result.schur_model, z, b)
    @test reconstructed ≈ exact atol=1e-12
    @test_throws ArgumentError build_response_complement_schur_model(
        responses, W, B, D; capture_target=0.95, naux=2,
    )
    @test_throws ArgumentError build_response_complement_schur_model(
        responses, W, B, D,
    )
    dimension_error = try
        build_response_complement_schur_model(
            responses, W, B[:, 1:3],
            Diagonal([1.3, 2.1, 4.2]) |> Matrix;
            naux=3,
        )
        nothing
    catch error
        error
    end
    @test dimension_error isa DimensionMismatch
    @test dimension_error.msg ==
      "response residual space and Schur eliminated space must have the same dimension"
end
