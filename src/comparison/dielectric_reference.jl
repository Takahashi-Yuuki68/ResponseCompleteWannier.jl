struct DielectricSpectrum
    omega_ev::Vector{Float64}
    epsilon::Vector{ComplexF64}
    loss::Vector{Float64}

    function DielectricSpectrum(omega_ev, epsilon, loss)
        length(omega_ev) == length(epsilon) == length(loss) ||
            throw(DimensionMismatch("spectrum fields must have equal lengths"))
        isempty(omega_ev) && throw(ArgumentError("dielectric spectrum is empty"))
        all(isfinite, omega_ev) || throw(ArgumentError("omega_ev contains NaN or Inf"))
        all(isfinite, real.(epsilon)) && all(isfinite, imag.(epsilon)) ||
            throw(ArgumentError("epsilon contains NaN or Inf"))
        all(isfinite, loss) || throw(ArgumentError("loss contains NaN or Inf"))
        all(diff(omega_ev) .> 0) ||
            throw(ArgumentError("omega_ev must be strictly increasing"))
        new(Float64.(omega_ev), ComplexF64.(epsilon), Float64.(loss))
    end
end

function _parse_reference_float(token, line_number)
    try
        value = parse(Float64, token)
        isfinite(value) ||
            throw(ArgumentError("non-finite value on data line $line_number"))
        return value
    catch error
        error isa ArgumentError ||
            throw(ArgumentError("invalid numeric value on data line $line_number"))
        rethrow()
    end
end

"""
    load_pw_reference(path)

Read the validated eight-column PW dielectric reference format. Comment and
blank lines are ignored; columns 1, 6, 7, and 8 provide photon energy,
the real and imaginary macroscopic dielectric function, and loss.
"""
function load_pw_reference(path)
    omega_ev = Float64[]
    epsilon = ComplexF64[]
    loss = Float64[]
    open(path, "r") do io
        for (line_number, line) in enumerate(eachline(io))
            stripped = strip(line)
            if isempty(stripped) || startswith(stripped, "#")
                continue
            end
            fields = split(stripped)
            length(fields) == 8 ||
                throw(ArgumentError("expected 8 columns on data line $line_number"))
            values = [_parse_reference_float(field, line_number) for field in fields]
            push!(omega_ev, values[1])
            push!(epsilon, complex(values[6], values[7]))
            push!(loss, values[8])
        end
    end
    DielectricSpectrum(omega_ev, epsilon, loss)
end

struct DielectricComparison
    max_abs_error_real::Float64
    max_abs_error_imag::Float64
    rms_error_real::Float64
    rms_error_imag::Float64
    reference_peak_energy_ev::Float64
    candidate_peak_energy_ev::Float64
    peak_energy_error_ev::Float64
    reference_peak_height::Float64
    candidate_peak_height::Float64
    peak_height_relative_error::Float64
end

"""
    compare_dielectric_spectra(reference, candidate)

Compare real and imaginary dielectric-function samples on an identical
frequency grid. The sampled peak is the first maximum of `Im(epsilon)`.
"""
function compare_dielectric_spectra(reference::DielectricSpectrum,
                                    candidate::DielectricSpectrum)
    reference.omega_ev == candidate.omega_ev ||
        throw(ArgumentError("reference and candidate frequency grids differ"))
    delta_real = real.(candidate.epsilon) .- real.(reference.epsilon)
    delta_imag = imag.(candidate.epsilon) .- imag.(reference.epsilon)
    reference_peak_index = argmax(imag.(reference.epsilon))
    candidate_peak_index = argmax(imag.(candidate.epsilon))
    reference_peak_height = imag(reference.epsilon[reference_peak_index])
    candidate_peak_height = imag(candidate.epsilon[candidate_peak_index])
    peak_height_relative_error = if iszero(reference_peak_height)
        iszero(candidate_peak_height) ? 0.0 : Inf
    else
        abs(candidate_peak_height - reference_peak_height) /
        abs(reference_peak_height)
    end
    DielectricComparison(
        maximum(abs.(delta_real)),
        maximum(abs.(delta_imag)),
        sqrt(sum(abs2, delta_real) / length(delta_real)),
        sqrt(sum(abs2, delta_imag) / length(delta_imag)),
        reference.omega_ev[reference_peak_index],
        candidate.omega_ev[candidate_peak_index],
        candidate.omega_ev[candidate_peak_index] -
        reference.omega_ev[reference_peak_index],
        reference_peak_height,
        candidate_peak_height,
        peak_height_relative_error,
    )
end

"""
    write_comparison_table(path, reference, candidate)

Write a whitespace-separated dielectric comparison table without adding a
plotting dependency. The parent directory must already exist.
"""
function write_comparison_table(path, reference::DielectricSpectrum,
                                candidate::DielectricSpectrum)
    reference.omega_ev == candidate.omega_ev ||
        throw(ArgumentError("reference and candidate frequency grids differ"))
    parent = dirname(path)
    isdir(parent) ||
        throw(ArgumentError("parent directory does not exist: $parent"))
    open(path, "w") do io
        println(io, "# omega_eV Re_epsilon_reference Im_epsilon_reference ",
                "Re_epsilon_candidate Im_epsilon_candidate delta_Re delta_Im ",
                "loss_reference loss_candidate")
        for i in eachindex(reference.omega_ev)
            println(io, join((
                reference.omega_ev[i],
                real(reference.epsilon[i]),
                imag(reference.epsilon[i]),
                real(candidate.epsilon[i]),
                imag(candidate.epsilon[i]),
                real(candidate.epsilon[i] - reference.epsilon[i]),
                imag(candidate.epsilon[i] - reference.epsilon[i]),
                reference.loss[i],
                candidate.loss[i],
            ), " "))
        end
    end
    return path
end
