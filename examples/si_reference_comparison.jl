#!/usr/bin/env julia

using ResponseCompleteWannier

const REFERENCE_PATH = joinpath(
    @__DIR__, "..", "data", "reference", "Si", "pw_sternheimer_dense_peak.dat",
)
const SOS_PATH = joinpath(
    @__DIR__, "..", "data", "reference", "Si", "pw_sos_validation.dat",
)

function read_max_sos_relative_error(path)
    values = Float64[]
    for line in eachline(path)
        stripped = strip(line)
        if isempty(stripped) || startswith(stripped, "#")
            continue
        end
        fields = split(stripped)
        length(fields) == 8 ||
            throw(ArgumentError("expected 8 columns in SOS validation data"))
        push!(values, parse(Float64, fields[6]))
    end
    isempty(values) && throw(ArgumentError("SOS validation data is empty"))
    maximum(values)
end

function print_reference_summary(reference)
    peak = argmax(imag.(reference.epsilon))
    println("Si PW Sternheimer reference")
    println("  Ecut: 15 Ha")
    println("  k-grid: 4 x 4 x 4 full mesh")
    println("  SCF bands: 12")
    println("  electronic temperature: 1.0e-4 Ha")
    println("  SCF tolerance: 1.0e-8")
    println("  broadening eta: 0.10 eV")
    println("  qred: (0.25, 0, 0), (0.50, 0, 0)")
    println("  q -> 0: two-point linear extrapolation in |q|^2")
    println("  frequency points: ", length(reference.omega_ev))
    println("  first sampled local maximum: ", reference.omega_ev[peak], " eV")
    println("  peak Re epsilon_M: ", real(reference.epsilon[peak]))
    println("  peak Im epsilon_M: ", imag(reference.epsilon[peak]))
    println("  peak loss: ", reference.loss[peak])
    println("  max Sternheimer-Schur vs 12-band SOS relative difference: ",
          read_max_sos_relative_error(SOS_PATH))
    println("  Wannier/MRWF physical adapter: not implemented")
end

function main(args)
    reference = load_pw_reference(REFERENCE_PATH)
    print_reference_summary(reference)
    isempty(args) && return

    candidate = load_pw_reference(args[1])
    comparison = compare_dielectric_spectra(reference, candidate)
    println("Candidate comparison")
    println("  max absolute Re error: ", comparison.max_abs_error_real)
    println("  max absolute Im error: ", comparison.max_abs_error_imag)
    println("  RMS Re error: ", comparison.rms_error_real)
    println("  RMS Im error: ", comparison.rms_error_imag)
    println("  sampled peak energy error: ", comparison.peak_energy_error_ev, " eV")
    println("  sampled peak height relative error: ",
            comparison.peak_height_relative_error)
    length(args) >= 2 && write_comparison_table(args[2], reference, candidate)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
