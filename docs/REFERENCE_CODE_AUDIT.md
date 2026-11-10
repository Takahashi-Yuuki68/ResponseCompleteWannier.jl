# Reference-code audit: `reference/pw`

This audit records the conventions confirmed by the checked-in
`KemlinSpectrum_validated.jl` and `Si_validated.jl` files. It does not claim
that this package is a DFTK adapter.

## Si setup

`Si_validated.jl` defines `SiConfig` with lattice parameter `a = 10.26`
bohr, `hgh/lda/Si-q4.hgh`, `Ecut = 15.0` Ha, `kgrid = (4, 4, 4)`,
`temperature = 0.01` Ha in the source configuration, `n_bands = 12`,
occupation threshold `1e-8`, SCF tolerance `1e-8`, and symmetries enabled.
The checked-in validated data used for this assignment documents the
low-temperature reference as `1.0e-4` Ha; this distinction must remain
explicit rather than silently inferred.

`run_si_scf` constructs the DFTK model, a `PlaneWaveBasis` with the configured
cutoff and k mesh, and a `FixedBands` algorithm with 12 converged and computed
bands. Occupied and extra states are classified by
`occupation_threshold`; the validated low-temperature reference has four
occupied bands at every k point. The source does not provide an independent
serialized occupation table, so that statement is a validated-data
condition, not a new inference from this package.

## Dynamic Sternheimer conventions

`sternheimer_solver_dynamic` explicitly uses
`z = branch * omega + im * eta`, with `branch = +1` for the resonant
equation and `branch = -1` for the antiresonant equation. The response
assembly calls both branches and combines them using the conjugated
antiresonant contraction shown in `density_response_column_schur`.
The checked-in validated data uses broadening `eta = 0.10` eV for the reported
Si spectrum; default values in the prototype source are separate code
defaults and must not be substituted into the reference metadata.
The source expresses the solver shift in its native energy units, but this
checkout does not contain a confirmed eV-to-Hartree conversion at the
reference-data boundary. The conversion used to produce the checked-in
table is therefore recorded as unconfirmed here rather than inferred.

## k/q and Brillouin-zone bookkeeping

The caller supplies `kplusq[ik]`, the target index for `k_i + q`. The
implementation checks
`kptq.coordinate - kpt.coordinate - qred` against an integer
`fold_shift`, then subtracts that shift from the effective `G` and `G'`
indices. This is the confirmed BZ-folding correction. The qred values in the
validated Si reference are `(0.25, 0, 0)` and `(0.50, 0, 0)`.

## Density response and dielectric assembly

For each occupied source state, `density_response_column_schur` solves the
resonant and antiresonant Sternheimer equations at `k + q`, contracts the
response with the shifted source state, and sums over k points and occupied
bands. The weight is the DFTK k-point weight times
`occupation / filled_occupation` and the explicit `spin_factor` (default
2.0). The accumulated response is divided by
`basis.model.unit_cell_volume`.

`chi0_matrix_schur` builds `chi0_GG'` one perturbing `G'` column at a time.
`dielectric_matrix_from_chi0` forms
`epsilon_GG' = delta_GG' - v_G(q) chi0_GG'`, with
`v_G(q) = 4*pi / |q+G|^2` in atomic units. The optional symmetrized form
uses the square roots of the Coulomb factors. `macroscopic_epsilon` inverts
the dielectric matrix and returns
`epsilon_M = 1 / [epsilon^{-1}]_00`, retaining local-field effects.

The optical calculation avoids exactly `q = 0`, evaluates finite-q cases,
and fits real and imaginary `epsilon_M(q)` against `|q|^2`. The validated
Si reference uses the two-point linear extrapolation in `|q|^2`.

## SOS validation

`pw_sos_validation.dat` records the relative Sternheimer-Schur versus
12-band truncated-SOS differences at 3.00, 3.25, and 3.50 eV:
0.004848502554456517, 0.004063355994414608, and
0.004812975216665147. The maximum is below 0.5%.

## Runtime boundary

`reference/pw` is not included by `src/ResponseCompleteWannier.jl` and is
not a package runtime dependency. Its DFTK prototype remains reference
material. The audit does not confirm `U(k)`, MMN, Wannier functions, or a
physical corrected Wannier/MRWF dielectric spectrum; those interfaces remain
unimplemented.
