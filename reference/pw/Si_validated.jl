#!/usr/bin/env julia
# Si.jl
# Material-dependent example/configuration for KemlinSpectrum.jl.
include("KemlinSpectrum.jl")
using DFTK

Base.@kwdef struct SiConfig
    a::Float64=10.26                 # bohr
    pseudopotential::String="hgh/lda/Si-q4.hgh"
    Ecut::Float64=15.0               # Hartree
    kgrid::NTuple{3,Int}=(4,4,4)
    temperature::Float64=0.01        # Hartree
    n_bands::Int=12
    occupation_threshold::Float64=1e-8
    scf_tol::Float64=1e-8
    use_symmetries::Bool=true
end

function silicon_model(cfg::SiConfig=SiConfig())
    lattice=cfg.a/2 .* [[0 1 1];[1 0 1];[1 1 0]]
    Si=ElementPsp(:Si,load_psp(cfg.pseudopotential))
    positions=[[1/8,1/8,1/8],[-1/8,-1/8,-1/8]]
    elements=[Si,Si]
    model_DFT(
        lattice,
        elements,
        positions;
        functionals=LDA(),
        temperature=cfg.temperature,
        smearing=Smearing.FermiDirac(),
        symmetries=cfg.use_symmetries,
    )
end

function run_si_scf(cfg::SiConfig=SiConfig())
    model = silicon_model(cfg)

    basis = PlaneWaveBasis(
        model;
        Ecut=cfg.Ecut,
        kgrid=cfg.kgrid,
    )

    nbandsalg = FixedBands(
        n_bands_converge=cfg.n_bands,
        n_bands_compute=cfg.n_bands,
        occupation_threshold=cfg.occupation_threshold,
    )

    self_consistent_field(
        basis;
        tol=cfg.scf_tol,
        nbandsalg,
    )
end

function print_si_next_step()
    println("Si material setup is ready. Next supply a physically defined perturbation dHpsi.")
    println("Then call compute_delta_psi_omega_schur(...) from KemlinSpectrum.jl.")
    println("Dielectric assembly is intentionally not guessed here; next implement the SternheimerGW-like density-response route.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    print_si_next_step()
end
