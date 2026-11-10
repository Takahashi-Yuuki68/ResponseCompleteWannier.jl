# ResponseCompleteWannier.jl

This package currently provides a tested algebraic foundation for response
completeness experiments. It is not yet a physical Wannier/MRWF dielectric
calculator.

## Implemented

- Exact algebraic Schur solves and frequency-independent auxiliary
  linearization.
- Projection of response columns outside an orthonormal Wannier subspace.
- Dense SVD response-complement bases and capture-based auxiliary-rank
  selection.
- Auxiliary-basis convergence evaluation against an exact Schur solution.
- Loading and comparing the validated Si PW Sternheimer reference spectrum.
- Whitespace comparison-table output without a plotting dependency.
- `examples/si_reference_comparison.jl`, which reports the reference
  conditions and compares a user-supplied candidate file without inventing
  candidate physics.

The comparison API is algebraic and data-format based. It does not construct
physical Wannier/MRWF orbitals or a DFTK response.

The response-complement basis `Q` is passed to the Schur reduction as a basis
of the eliminated space. Therefore, the row space of `responses` and `W`,
the columns of `B`, and the `D` space must use the same ambient basis
representation and dimension. A physical adapter must perform any required
projection or basis transformation before calling the algebraic integration
API.

## Si PW reference

The checked-in reference uses Si with Ecut 15 Ha, a 4 x 4 x 4 full k mesh,
12 SCF bands, electronic temperature 1.0e-4 Ha, SCF tolerance 1.0e-8,
broadening eta 0.10 eV, and qred values `(0.25, 0, 0)` and
`(0.50, 0, 0)`. The optical limit is a two-point linear extrapolation in
`|q|^2`. The frequency grid is 3.0--3.6 eV in 0.1 eV steps. Its first
sampled local maximum is at 3.3 eV, where
`Re epsilon_M = 10.377858154800162`,
`Im epsilon_M = 3.4031661882533535`, and
`loss = 0.02853054965994824`.
The Sternheimer-Schur versus 12-band truncated-SOS relative differences are
within 0.5% for the listed validation points.

## Driver

From the repository root, after dependencies are installed, the reference
summary can be printed with:

```text
julia --project=. examples/si_reference_comparison.jl
```

With a candidate in the same eight-column format, the driver prints error
statistics:

```text
julia --project=. examples/si_reference_comparison.jl candidate.dat
```

The second argument writes the comparison table:

```text
julia --project=. examples/si_reference_comparison.jl candidate.dat comparison.dat
```

## Tests

The package test suite is run with:

```text
julia --project=. -e "using Pkg; Pkg.test()"
```

The existing Schur and response-residual tests remain in place; new tests
cover rank selection, auxiliary convergence, reference parsing, comparison,
table output, and the algebraic integration API.

## Explicitly not implemented

- DFTK ground-state adapter.
- A PW Sternheimer operator in this package.
- `U(k)`, MMN, and k/q mapping adapters.
- Physical Wannier/MRWF basis generation.
- A physical dielectric spectrum after response-completeness correction.
- The final physical PW Sternheimer versus Wannier/MRWF comparison.

The files under `reference/pw` are audit/reference material and are not
included in the package runtime.
