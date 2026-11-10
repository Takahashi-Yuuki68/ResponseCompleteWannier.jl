# Mathematical model and status of claims

This note separates established identities from the proposed numerical
workflow. It is not a claim that the methods have been validated against a
DFTK calculation. The dynamic frequency and gauge conventions of the
Kemlin prototype remain unknown pending source inspection; the synthetic
prototype here uses a scalar shift `z` and a fixed Hermitian block system.

## 1. Plane-wave dynamic Sternheimer response

**[KNOWN]** A linear-response orbital correction can be represented
abstractly as a frequency-dependent linear equation

\[
\mathcal{L}(q,z)\,\delta\psi(q,z)=s(q,z),
\qquad z=\omega+i\eta\ \text{in one common convention}.
\]

Here `𝓛` includes the ground-state Hamiltonian, occupied-state projection,
q-dependent coupling, and any occupation/gauge treatment. The displayed
choice of `z` is illustrative only: the sign of `ω`, placement of `iη`, and
the relation between positive- and negative-frequency channels must be
copied from and checked against the reference implementation before use.
This repository currently has no PW Sternheimer operator.

## 2. Block decomposition

**[EXTENSION]** The standalone algebraic prototype models a fixed shifted
block operator

\[
H-zI =
\begin{bmatrix} A & B \\ B^\dagger & D \end{bmatrix}
-z\begin{bmatrix}I&0\\0&I\end{bmatrix},
\qquad
(H-zI)\begin{bmatrix}x\\y\end{bmatrix}
=\begin{bmatrix}b\\0\end{bmatrix}.
\]

For square Hermitian `A` and `D`, `B` has shape `n × m`, `x,b` have length
`n`, and `y` has length `m`. This is a synthetic algebraic model; identifying
these blocks with any particular DFTK/Kemlin subspaces is future work.

## 3. Exact Schur complement

**[KNOWN]** If `D-zI` is invertible, eliminate
\[
y=-(D-zI)^{-1}B^\dagger x
\]
to obtain
\[
\left[A-zI-B(D-zI)^{-1}B^\dagger\right]x=b.
\]
The implementation evaluates the action using a linear solve, not an
explicit inverse. A unit test compares this result to the corresponding full
block solve.

## 4. Wannier projector and response residual

**[KNOWN]** For an orthonormal basis matrix `W`, the projector is
\[
P_W=WW^\dagger,\qquad P_W^\dagger=P_W,\quad P_W^2=P_W.
\]
For response columns `Ψ`, the orthogonal residual is
\[
R=(I-P_W)\Psi.
\]
The implementation checks orthonormality and tests the projection identities.
This is not yet a Wannier adapter: no `U(k)`, MMN, k/q mapping, or DFTK
wavefunction representation is present.

## 5. Response-complement SVD

**[PROPOSED]** Given residual columns, compute
\[
R=U\Sigma V^\dagger
\]
and retain the leading `m` left singular vectors as a
`ResponseComplement`. The captured squared Frobenius norm is
\[
\frac{\sum_{i=1}^{m}\sigma_i^2}{\sum_i\sigma_i^2}.
\]
The implementation provides ordinary dense SVD only. Low rank, predictive
value on held-out frequencies/q points, and improved dielectric convergence
are hypotheses requiring cross-validation on physical calculations.

## 6. Effective Wannier plus response-complement space

**[PROPOSED]** If each retained response-complement vector is orthogonal to
`W`, one may represent the enlarged space as
\[
\mathcal{H}_{\mathrm{eff}}=\operatorname{span}(W)\oplus
\operatorname{span}(U_{1:m}).
\]
The synthetic projection test demonstrates residual capture only; it does
not construct or validate an effective Hamiltonian or dielectric response.
`ResponseComplement` names this Hilbert-space correction and is distinct from
the Schur auxiliary model below.

## 7. Rational approximation to Schur self-energy

**[APPROXIMATION]** Choose an orthonormal reduced basis `Q` for the eliminated
space and define
\[
T=Q^\dagger DQ,\qquad \widetilde B=BQ,\qquad
\Sigma_{\mathrm{ROM}}(z)=\widetilde B(T-zI)^{-1}\widetilde B^\dagger.
\]
This approximates `Σ(z)=B(D-zI)⁻¹B†` unless `Q` spans an invariant subspace
containing all relevant coupled directions. This prototype accepts `Q` from
the caller and checks orthonormality; it does not yet construct rational
Krylov bases or quantify errors over a frequency domain.

## 8. Auxiliary linearization

**[KNOWN]** Eliminating `y` from
\[
\left[
\begin{bmatrix}A&\widetilde B\\
\widetilde B^\dagger&T\end{bmatrix}
-zI\right]
\begin{bmatrix}x\\y\end{bmatrix}
=\begin{bmatrix}b\\0\end{bmatrix}
\]
gives the reduced rational Schur system
\[
\left[A-zI-\widetilde B(T-zI)^{-1}\widetilde B^\dagger\right]x=b.
\]
The code uses the distinct type `SchurAuxiliaryModel` for these eliminated
variables. Tests verify equivalence of the reduced and enlarged systems for
multiple complex shifts.

## 9. Conditions for shift invariance

**[EXTENSION]** The enlarged reduced matrix
\[
H_{\mathrm{ROM}}=
\begin{bmatrix}A&\widetilde B\\
\widetilde B^\dagger&T\end{bmatrix}
\]
must be independent of `z`; frequency may then enter only as the scalar
shift `H_ROM-zI`. The eliminated Schur operator itself remains
frequency-dependent. This algebraic structure is necessary, but not
sufficient, to establish that a particular multi-shift Krylov implementation
is valid or efficient. No multi-shift solver is implemented here.

## 10. Auxiliary-rank convergence and comparison

**[KNOWN]** `select_auxiliary_rank` selects the smallest sampled SVD rank
whose cumulative squared-singular-value capture reaches a requested target.
The zero target and zero-residual cases return rank zero. The convergence
routine compares each reduced solution with the exact algebraic Schur
solution and reports absolute and relative vector errors. It does not assume
that errors are monotone in rank.

**[PROPOSED]** `build_response_complement_schur_model` connects response
residual SVD vectors to a Schur auxiliary model. This connection is an
algebraic API only. It does not identify `B`, `D`, `W`, or the response
columns with physical DFTK or Wannier/MRWF objects.

The response-complement basis `Q` has the same ambient row-space
representation as the input response columns. When it is passed to
`reduce_schur(B, D, Q)`, that ambient basis must also be the eliminated
space represented by `D`; consequently,
`size(responses, 1) == size(D, 1) == size(B, 2)` is required. A physical
adapter must explicitly project or transform between spaces before invoking
this algebraic API.

## 11. Dielectric reference data

**[KNOWN]** The reference loader accepts the checked-in eight-column sampled
format and compares complex dielectric samples on an identical frequency
grid. Its peak is explicitly a sampled maximum of `Im(epsilon)`, not a claim
about a continuously converged local maximum.

**[APPROXIMATION]** A finite sampled comparison and RMS/max errors cannot
establish agreement between physical PW and Wannier/MRWF response functions
outside the supplied grid.
