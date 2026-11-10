#!/usr/bin/env julia
# KemlinSpectrum.jl
# Material-independent static Kemlin Schur core + unified finite-frequency Schur extension.
# Target: DFTK v0.5.x-era API used by gkemlin/response-calculations-metals.
# This file deliberately stops at delta_psi(omega). Density-response/dielectric assembly belongs in a validated layer above.

using DFTK
using LinearAlgebra
using LinearMaps
using IterativeSolvers

# ---------------- Original/static Kemlin-compatible core ----------------
function sternheimer_solver(Hk, psik, epsnk, rhs, n; callback=info->nothing,
                            psik_extra=zeros(size(psik,1),0), epsk_extra=zeros(0),
                            Hpsik_extra=zeros(size(psik,1),0), abstol=1e-9,
                            reltol=0, verbose=false)
    basis=Hk.basis; kpoint=Hk.kpoint
    P(phi)=psik*(psik'*phi); Pextra(phi)=psik_extra*(psik_extra'*phi)
    Pcomputed(phi)=P(phi)+Pextra(phi); Q(phi)=phi-P(phi); R(phi)=phi-Pcomputed(phi)
    H(phi)=Hk*phi-epsnk*phi
    Dex=Diagonal(real.(epsk_extra .- epsnk))
    b=-Q(rhs)
    bb=R(b-Hpsik_extra*(Dex \ (psik_extra'*b)))
    function RAR(phi)
        r=R(phi)
        R(H(r))-R(Hpsik_extra)*(Dex \ (Hpsik_extra'*r))
    end
    precon=PreconditionerTPA(basis,kpoint); DFTK.precondprep!(precon,psik[:,n])
    function Rldiv!(x,y); x .= R(precon \ R(y)); end
    J=LinearMap{eltype(psik)}(RAR,size(Hk,1))
    dR,ch=cg(J,bb;Pl=DFTK.FunctionPreconditioner(Rldiv!),abstol,reltol,verbose,log=true)
    callback((;basis,kpoint,ch,n))
    alpha=Dex \ (psik_extra'*(b-H(dR)))
    psik_extra*alpha+dR
end

function compute_alpha_mn(fm,fn,ratio)
    ratio==0 && return ratio
    ratio*fn/(fn^2+fm^2)
end

@views DFTK.@timing function apply_chi0_schur(ham,psi,occ,eF,eigenvalues,dHpsi;
                                               occupation_threshold,kwargs_sternheimer...)
    basis=ham.basis; model=basis.model; temperature=model.temperature
    filled=DFTK.filled_occupation(model); T=eltype(basis); Nk=length(basis.kpoints)
    maskocc=map(ok->isless.(occupation_threshold,ok),occ)
    maskex=map(ok->(!isless).(occupation_threshold,ok),occ)
    po=[psi[ik][:,m] for (ik,m) in enumerate(maskocc)]; px=[psi[ik][:,m] for (ik,m) in enumerate(maskex)]
    eo=[eigenvalues[ik][m] for (ik,m) in enumerate(maskocc)]; ex=[eigenvalues[ik][m] for (ik,m) in enumerate(maskex)]
    oo=[occ[ik][m] for (ik,m) in enumerate(maskocc)]
    deF=zero(T); docc=[zero(oo[ik]) for ik=1:Nk]
    if temperature>0
        D=zero(T)
        for ik=1:Nk,(n,en) in enumerate(eo[ik])
            enred=(en-eF)/temperature; den=real(dot(po[ik][:,n],dHpsi[ik][:,n]))
            fpn=filled*Smearing.occupation_derivative(model.smearing,enred)/temperature
            docc[ik][n]=den*fpn; D+=fpn*basis.kweights[ik]
        end
        D=DFTK.mpi_sum(D,basis.comm_kpts)
        dtot=DFTK.mpi_sum(sum(basis.kweights .* sum.(docc)),basis.comm_kpts); deF=dtot/D
        for ik=1:Nk,(n,en) in enumerate(eo[ik])
            fpn=filled*Smearing.occupation_derivative(model.smearing,(en-eF)/temperature)/temperature
            docc[ik][n]-=fpn*deF
        end
    end
    dpsi=zero.(psi)
    for ik=1:Nk
        psik=po[ik]; dpsik=dpsi[ik]; epsk=eo[ik]; pex=px[ik]; Hk=ham.blocks[ik]
        Hpex=iszero(length(pex)) ? zeros(size(psik,1),0) : Hk*pex
        for n=1:length(epsk)
            fn=filled*Smearing.occupation(model.smearing,(epsk[n]-eF)/temperature)
            for m=1:length(epsk)
                fm=filled*Smearing.occupation(model.smearing,(epsk[m]-eF)/temperature)
                ratio=filled*Smearing.occupation_divided_difference(model.smearing,epsk[m],epsk[n],eF,temperature)
                amn=compute_alpha_mn(fm,fn,ratio)
                dpsik[:,n].+=psik[:,m].*amn.*(dot(psik[:,m],dHpsi[ik][:,n])*(n!=m))
            end
            dpsik[:,n].+=sternheimer_solver(Hk,psik,epsk[n],dHpsi[ik][:,n],n;
                                             psik_extra=pex,epsk_extra=ex[ik],Hpsik_extra=Hpex,
                                             kwargs_sternheimer...)
        end
    end
    doccupation=zero.(occ)
    for (ik,m) in enumerate(maskocc); doccupation[ik][m].+=docc[ik]; end
    (;dpsi,doccupation,deF)
end

# ---------------- Unified dynamic Schur engine ----------------
# branch=+1 solves H-eps_n-(omega+i*eta); branch=-1 solves H-eps_n-(-omega+i*eta).
# The i*eta convention is kept explicit. Validate +/- assembly against an SOS reference before building observables.
function sternheimer_solver_dynamic(Hk,psi_occ,eps_n,rhs,n;omega::Real,eta::Real=1e-3,branch::Int=+1,
        callback=info->nothing,psi_extra=zeros(eltype(psi_occ),size(psi_occ,1),0),
        eps_extra=zeros(0),Hpsi_extra=zeros(eltype(psi_occ),size(psi_occ,1),0),
        abstol=1e-9,reltol=0.0,restart=30,maxiter=400,verbose=false)
    branch in (-1,+1) || error("branch must be +1 or -1")
    basis=Hk.basis; kpoint=Hk.kpoint
    CT=promote_type(eltype(psi_occ),ComplexF64); z=CT(branch*omega+im*eta)
    P(phi)=psi_occ*(psi_occ'*phi); Pex(phi)=psi_extra*(psi_extra'*phi)
    Pc(phi)=P(phi)+Pex(phi); Q(phi)=phi-P(phi); R(phi)=phi-Pc(phi)
    A(phi)=Hk*phi-(eps_n+z)*phi
    d=CT.(eps_extra .- eps_n) .- z
    solveD(v)=isempty(d) ? similar(v,0) : v./d
    b=-Q(CT.(rhs)); Aex=CT.(Hpsi_extra).-((eps_n+z).*CT.(psi_extra))
    bb=R(b-Aex*solveD(psi_extra'*b))
    function Schur(phi)
        r=R(phi); ar=A(r)
        R(ar)-R(Aex)*solveD(Aex'*r)
    end
    precon=PreconditionerTPA(basis,kpoint); DFTK.precondprep!(precon,psi_occ[:,n])
    function Rldiv!(x,y); x .= R(precon \ R(y)); end
    J=LinearMap{CT}(Schur,size(Hk,1))
    dR,ch=gmres(J,bb;Pl=DFTK.FunctionPreconditioner(Rldiv!),abstol,reltol,restart,maxiter,verbose,log=true)
    callback((;basis,kpoint,ch,n,omega,eta,branch,z))
    alpha=solveD(psi_extra'*(b-A(dR))); dpsi=psi_extra*alpha+dR
    (;dpsi,dR,alpha,history=ch,z)
end

function compute_delta_psi_omega_schur(ham,psi,occ,eigenvalues,dHpsi;omega::Real,eta::Real=1e-3,
        branch::Int=+1,occupation_threshold,kwargs_sternheimer...)
    Nk=length(ham.basis.kpoints); mo=map(o->isless.(occupation_threshold,o),occ); mx=map(o->.!isless.(occupation_threshold,o),occ)
    po=[psi[ik][:,m] for (ik,m) in enumerate(mo)]; px=[psi[ik][:,m] for (ik,m) in enumerate(mx)]
    eo=[eigenvalues[ik][m] for (ik,m) in enumerate(mo)]; ex=[eigenvalues[ik][m] for (ik,m) in enumerate(mx)]
    out=[zeros(ComplexF64,size(psi[ik],1),size(psi[ik],2)) for ik=1:Nk]; histories=Any[]
    for ik=1:Nk
        Hk=ham.blocks[ik]; Hpex=size(px[ik],2)==0 ? zeros(ComplexF64,size(po[ik],1),0) : Hk*px[ik]
        for n in eachindex(eo[ik])
            r=sternheimer_solver_dynamic(Hk,po[ik],eo[ik][n],dHpsi[ik][:,n],n;omega,eta,branch,
                psi_extra=px[ik],eps_extra=ex[ik],Hpsi_extra=Hpex,kwargs_sternheimer...)
            out[ik][:,n].=r.dpsi; push!(histories,(;ik,n,history=r.history))
        end
    end
    (;dpsi=out,omega,eta,branch,histories,mask_occ=mo,mask_extra=mx)
end

compute_delta_psi_grid_schur(ham,psi,occ,eigenvalues,dHpsi,omegas;kwargs...) =
    [compute_delta_psi_omega_schur(ham,psi,occ,eigenvalues,dHpsi;omega=w,kwargs...) for w in omegas]

# ---------------- Material-independent adaptive-extra-band helpers ----------------
function adaptive_extra_bands(H,psi,epsvals;target=3.0,max_add=24)
    N=size(psi,2); N==0 && return (psi_extra=similar(psi,size(psi,1),0),eps_extra=Float64[],n_added=0,ratio=0.0)
    prec=PreconditionerTPA(H.basis,H.kpoint); DFTK.precondprep!(prec,psi[:,end])
    work=copy(psi); vals=copy(epsvals); ratio=Inf; added=0
    while ratio>target && added<max_add
        phi=randn(eltype(psi),size(psi,1)); phi-=work*(work'*phi); phi/=norm(phi)
        res=DFTK.lobpcg_hyper(H,[work phi];prec,tol=max(abs(vals[end]-vals[1])/100,1e-8))
        work=res.X; vals=res.λ; added=size(work,2)-N; ex=vals[end]
        den=max(ex-vals[min(N,length(vals))],Base.eps(Float64)); ratio=sqrt(max((ex-vals[1])/den,0.0))
    end
    (;psi_extra=work[:,N+1:end],eps_extra=vals[N+1:end],n_added=added,ratio)
end

function adaptive_extra_bands_allk(scfres;target=3.0,max_add=24)
    [adaptive_extra_bands(scfres.ham.blocks[ik],scfres.ψ[ik],scfres.eigenvalues[ik];target,max_add) for ik in eachindex(scfres.ψ)]
end

# No dielectric_spectrum() here on purpose. SternheimerGW constructs epsilon from the density response
# to plane-wave perturbations. We should implement/validate that layer separately rather than label a generic
# <probe|delta_psi> contraction as an optical dielectric function.

# ---------------- SternheimerGW-like density response -> dielectric matrix ----------------
# Eq. (4)-(8) layer. This is an explicit finite-q, plane-wave perturbation implementation.
# IMPORTANT: q must connect existing k points. The caller supplies kplusq[ik] = index of k_i+q.
# qcart and Gcart are Cartesian reciprocal vectors in bohr^-1; volume is in bohr^3.
# Coulomb convention: v(q+G)=4*pi/|q+G|^2 (atomic units, 3D periodic system).

"""Map a reciprocal vector to a plane-wave coefficient index for a DFTK k point."""
function _g_index_map(basis,kpoint)
    Gs=DFTK.G_vectors(basis,kpoint)
    Dict(Tuple(Int.(G))=>i for (i,G) in enumerate(Gs))
end

"""Apply exp(i G.r) to a periodic Bloch part represented in a PW basis.
Gred is an integer reciprocal-lattice vector. Source and target k-point PW bases may differ."""
function apply_exp_iG(basis,k_source,k_target,psi_source,Gred)
    Gs=DFTK.G_vectors(basis,k_source); Gt=DFTK.G_vectors(basis,k_target); tmap=Dict(Tuple(Int.(G))=>i for (i,G) in enumerate(Gt))
    out=zeros(ComplexF64,length(Gt)); dG=Int.(Gred)
    for (is,G) in enumerate(Gs)
        it=get(tmap,Tuple(Int.(G .+ dG)),0); it!=0 && (out[it]+=psi_source[is])
    end
    out
end

"""Eq. (8): b=(1-P_occ) exp(i G'.r)|u_nk>, represented at k+q.
For finite q the phase exp(i q.r) is carried by the Bloch-vector change k -> k+q;
Gprime_red is therefore only the reciprocal-lattice (integer G') part."""
function sternheimergw_rhs(basis,psi_k,psi_kq_occ,kpt_k,kpt_kq,n,Gprime_red)
    raw=apply_exp_iG(basis,kpt_k,kpt_kq,psi_k[:,n],Gprime_red)
    raw-psi_kq_occ*(psi_kq_occ'*raw)
end

"""Compute one density-response column delta_n_G,G'(q,omega) from dynamic Schur responses.
This follows the SternheimerGW Eq. (5)-(8) logic: solve at k+q for each occupied n,k,
then contract u_nk^*(r) delta_u_n,k+q(r,omega) and Fourier transform by G.
The +/- frequency branches are both included using the same retarded convention as the solver.
`kplusq` maps every source k index to its k+q target index.
"""
function density_response_column_schur(
        ham, psi, occ, eigenvalues,
        Gred_list, Gprime_red, kplusq, qred;
        omega::Real, eta::Real=1e-3,
        occupation_threshold=1e-8, spin_factor=2.0,
        kwargs_sternheimer...)
    basis=ham.basis; Nk=length(basis.kpoints); nG=length(Gred_list)
    dn=zeros(ComplexF64,nG)
    for ik in 1:Nk
        jk = kplusq[ik]
        kpt = basis.kpoints[ik]
        kptq = basis.kpoints[jk]
        Hq = ham.blocks[jk]

        fold_raw = (
            collect(kptq.coordinate)
            .- collect(kpt.coordinate)
            .- qred
        )
        fold_shift = round.(Int, fold_raw)

        norm(fold_raw .- fold_shift) < 1e-10 ||
            error("Invalid k+q fold for ik=$ik, jk=$jk")

        mo=isless.(occupation_threshold,occ[jk]); mx=.!mo
        psi_q_occ=psi[jk][:,mo]; psi_q_ex=psi[jk][:,mx]; eps_q_ex=eigenvalues[jk][mx]
        Hpex=size(psi_q_ex,2)==0 ? zeros(ComplexF64,size(psi[jk],1),0) : Hq*psi_q_ex
        for n in eachindex(eigenvalues[ik])
            occ[ik][n] <= occupation_threshold && continue
            Gprime_effective = Int.(Gprime_red) .- fold_shift

            rhs = sternheimergw_rhs(
                basis,
                psi[ik],
                psi_q_occ,
                kpt,
                kptq,
                n,
                Gprime_effective,
            )
            # SternheimerGW Eq. (6): reference energy is eps_nk while operator lives at k+q.
            rp=sternheimer_solver_dynamic(Hq,psi_q_occ,eigenvalues[ik][n],rhs,1;omega,eta,branch=+1,
                 psi_extra=psi_q_ex,eps_extra=eps_q_ex,Hpsi_extra=Hpex,kwargs_sternheimer...)
            rm=sternheimer_solver_dynamic(Hq,psi_q_occ,eigenvalues[ik][n],rhs,1;omega,eta,branch=-1,
                 psi_extra=psi_q_ex,eps_extra=eps_q_ex,Hpsi_extra=Hpex,kwargs_sternheimer...)
            w=basis.kweights[ik]*(occ[ik][n]/DFTK.filled_occupation(basis.model))*spin_factor
            # Reciprocal-space Fourier coefficient of u_nk^* delta_u_{n,k+q}.
            # Evaluate <u_nk|exp(-i G.r)|delta_u> in PW coefficients.
            for (iG, Gred) in enumerate(Gred_list)
                G_effective = Int.(Gred) .- fold_shift

                bra_shift = apply_exp_iG(
                    basis,
                    kpt,
                    kptq,
                    psi[ik][:, n],
                    G_effective,
                )

                dn[iG] += w * (
                    dot(bra_shift, rp.dpsi)
                    + conj(dot(bra_shift, rm.dpsi))
                )
            end
        end
    end
    volume = basis.model.unit_cell_volume
    DFTK.mpi_sum(dn, basis.comm_kpts) / volume
end

"""Build chi0_GG'(q,omega) one plane-wave perturbation G' at a time."""
function chi0_matrix_schur(
        ham, psi, occ, eigenvalues,
        Gred_list, kplusq, qred;
        omega::Real, eta::Real=1e-3,
        occupation_threshold=1e-8, spin_factor=2.0,
        kwargs_sternheimer...)
    nG=length(Gred_list); chi=zeros(ComplexF64,nG,nG)
    for jG in 1:nG
        chi[:, jG] .= density_response_column_schur(
            ham,
            psi,
            occ,
            eigenvalues,
            Gred_list,
            Gred_list[jG],
            kplusq,
            qred;
            omega,
            eta,
            occupation_threshold,
            spin_factor,
            kwargs_sternheimer...,
        )
    end
    chi
end

"""Bare 3D Coulomb factors v_G(q)=4*pi/|q+G|^2 in atomic units."""
function coulomb_3d(qcart,Gcart_list;head_value=nothing)
    v=zeros(Float64,length(Gcart_list))
    for (i,G) in enumerate(Gcart_list)
        qG=qcart .+ G; q2=dot(qG,qG)
        if q2==0
            isnothing(head_value) && error("q+G=0 Coulomb head diverges. Use finite q or supply a controlled head_value.")
            v[i]=head_value
        else
            v[i]=4pi/q2
        end
    end
    v
end

"""Eq. (4)-style dielectric matrix epsilon_GG'=delta_GG'-v_G(q)*chi0_GG'.
Set symmetrized=true for epsilon~=I-sqrt(v)*chi0*sqrt(v)."""
function dielectric_matrix_from_chi0(chi0,qcart,Gcart_list;symmetrized=false,head_value=nothing)
    v=coulomb_3d(qcart,Gcart_list;head_value)
    n=size(chi0,1); Igg=Matrix{ComplexF64}(I,n,n)
    symmetrized ? Igg-Diagonal(sqrt.(v))*chi0*Diagonal(sqrt.(v)) : Igg-Diagonal(v)*chi0
end

"""End-to-end Eq. (4)-(8) wrapper: dynamic Schur delta_psi -> delta_n/chi0 -> epsilon_GG'.
Gred_list supplies integer reciprocal vectors; Gcart_list the corresponding Cartesian bohr^-1 vectors.
The mapping kplusq is required and makes finite-q bookkeeping explicit instead of silently approximating q."""
function dielectric_matrix_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,qcart,kplusq;
        omega::Real,eta::Real=1e-3,occupation_threshold=1e-8,spin_factor=2.0,
        symmetrized=false,head_value=nothing,kwargs_sternheimer...)
    qred = ham.basis.model.inv_recip_lattice * qcart

    chi0 = chi0_matrix_schur(
        ham,
        psi,
        occ,
        eigenvalues,
        Gred_list,
        kplusq,
        qred;
        omega,
        eta,
        occupation_threshold,
        spin_factor,
        kwargs_sternheimer...,
    )
    epsilon=dielectric_matrix_from_chi0(chi0,qcart,Gcart_list;symmetrized,head_value)
    (;chi0,epsilon,epsilon_inv=inv(epsilon),omega,eta,qcart,Gred_list)
end

# ---------------- Finite-q optical-limit head/wing treatment ----------------
# For a 3D periodic solid, avoid q=0. Evaluate epsilon_GG'(q,omega) at one or
# several small nonzero q vectors compatible with the k mesh, invert the full
# matrix (thus retaining local-field effects), and take eps_M=1/epsinv_00.

"""Locate the G=(0,0,0) entry in a reduced reciprocal-vector list."""
function g0_index(Gred_list)
    i=findfirst(G->all(Int.(G).==0),Gred_list)
    isnothing(i) && error("Gred_list must contain G=(0,0,0) for macroscopic epsilon.")
    i
end

"""Macroscopic dielectric function including local-field effects:
epsilon_M(q,omega)=1/[epsilon(q,omega)^(-1)]_00."""
function macroscopic_epsilon(epsilon,Gred_list)
    i0=g0_index(Gred_list)
    einv=inv(epsilon)
    (;epsilon_M=inv(einv[i0,i0]),epsilon_inv=einv,iG0=i0)
end

"""One finite-q optical calculation. qcart must be nonzero.
The returned epsilon_M uses the full inverse dielectric matrix, so head/wing/body
coupling (local-field effects) is retained automatically."""
function optical_finite_q_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,qcart,kplusq;
        omega::Real,eta::Real=1e-3,occupation_threshold=1e-8,spin_factor=2.0,
        symmetrized=false,kwargs_sternheimer...)
    norm(qcart)>0 || error("Finite-q optical treatment requires qcart != 0. Do not insert q=0 into 4pi/q^2.")
    r=dielectric_matrix_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,qcart,kplusq;
        omega,eta,occupation_threshold,spin_factor,symmetrized,head_value=nothing,kwargs_sternheimer...)
    mac=macroscopic_epsilon(r.epsilon,Gred_list)
    (;r...,epsilon_M=mac.epsilon_M,iG0=mac.iG0)
end

"""Evaluate epsilon_M(omega) on a frequency grid at fixed finite q."""
function optical_spectrum_finite_q_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,qcart,kplusq,omegas;
        eta::Real=1e-3,occupation_threshold=1e-8,spin_factor=2.0,symmetrized=false,
        kwargs_sternheimer...)
    results=[optical_finite_q_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,qcart,kplusq;
        omega=w,eta,occupation_threshold,spin_factor,symmetrized,kwargs_sternheimer...) for w in omegas]
    epsM=ComplexF64[r.epsilon_M for r in results]
    (;omegas=collect(omegas),epsilon_M=epsM,epsilon1=real.(epsM),epsilon2=imag.(epsM),
      loss=-imag.(1 ./ epsM),results,qcart)
end

"""Finite-q convergence / q->0 extrapolation of epsilon_M at one omega.
`cases` is a vector of named tuples `(qcart=..., kplusq=...)`, ordered arbitrarily.
Fits epsilon_M(q)=epsilon_M(0)+a*|q|^2 by default (appropriate as a numerical
extrapolation for a centrosymmetric/even-in-q optical limit); set fit_power=1
for a simple linear-|q| diagnostic. Always inspect the raw q dependence."""
function optical_q_to_zero_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,cases;
        omega::Real,eta::Real=1e-3,occupation_threshold=1e-8,spin_factor=2.0,
        symmetrized=false,fit_power::Int=2,kwargs_sternheimer...)
    length(cases)>=2 || error("Provide at least two finite-q cases for q->0 extrapolation.")
    qs=Float64[]; vals=ComplexF64[]; raw=Any[]
    for c in cases
        r=optical_finite_q_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,c.qcart,c.kplusq;
            omega,eta,occupation_threshold,spin_factor,symmetrized,kwargs_sternheimer...)
        push!(qs,norm(c.qcart)); push!(vals,r.epsilon_M); push!(raw,r)
    end
    p=sortperm(qs); qs=qs[p]; vals=vals[p]; raw=raw[p]
    x=qs.^fit_power; X=hcat(ones(length(x)),x)
    cr=X\real.(vals); ci=X\imag.(vals); eps0=complex(cr[1],ci[1])
    fitted=complex.(X*cr,X*ci)
    (;qnorm=qs,epsilon_M_q=vals,epsilon_M_q0=eps0,fitted,fit_power,raw)
end

"""Frequency-resolved q->0 spectrum from several finite-q calculations."""
function optical_spectrum_q_to_zero_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,cases,omegas;
        eta::Real=1e-3,occupation_threshold=1e-8,spin_factor=2.0,symmetrized=false,
        fit_power::Int=2,kwargs_sternheimer...)
    fits=[optical_q_to_zero_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,cases;
        omega=w,eta,occupation_threshold,spin_factor,symmetrized,fit_power,kwargs_sternheimer...) for w in omegas]
    eps0=ComplexF64[f.epsilon_M_q0 for f in fits]
    (;omegas=collect(omegas),epsilon_M=eps0,epsilon1=real.(eps0),epsilon2=imag.(eps0),
      loss=-imag.(1 ./ eps0),fits,fit_power)
end

# ---------------- Plotting helpers for optical diagnostics ----------------
# Requires Plots.jl only when these routines are called.

"""Collect the no-local-field epsilon_00 from a finite-q result."""
function microscopic_head_epsilon(result)
    i0=result.iG0
    result.epsilon[i0,i0]
end

"""Generate the three requested optical diagnostic figures.
(a) Re epsilon_M(omega), Im epsilon_M(omega) for the q->0 extrapolated spectrum.
(b) Im epsilon_M(q,omega) for every finite-q case plus the q->0 extrapolation.
(c) Im epsilon_00 versus Im{1/[epsilon^-1]_00}, showing the local-field correction.

Returns a named tuple `(panel_a, panel_b, panel_c, finite_q_spectra, q0_spectrum)`.
If `prefix` is not `nothing`, PNG files `<prefix>_a.png`, `_b.png`, `_c.png` are saved.
"""
function plot_optical_diagnostics_schur(ham,psi,occ,eigenvalues,Gred_list,Gcart_list,cases,omegas;
        eta::Real=1e-3,occupation_threshold=1e-8,spin_factor=2.0,symmetrized=false,
        fit_power::Int=2,prefix=nothing,kwargs_sternheimer...)
    @eval import Plots
    length(cases)>=2 || error("At least two finite-q cases are required for the q->0 panel.")

    # Compute each finite-q spectrum once. Keep full epsilon matrices for panel (c).
    finite_specs=Any[]
    for c in cases
        push!(finite_specs,optical_spectrum_finite_q_schur(
            ham,psi,occ,eigenvalues,Gred_list,Gcart_list,c.qcart,c.kplusq,omegas;
            eta,occupation_threshold,spin_factor,symmetrized,kwargs_sternheimer...))
    end

    # Extrapolate epsilon_M(q,omega) -> q=0 from the already computed finite-q spectra.
    qnorm=[norm(c.qcart) for c in cases]
    p=sortperm(qnorm); qnorm=qnorm[p]; finite_specs=finite_specs[p]
    x=qnorm.^fit_power; X=hcat(ones(length(x)),x)
    eps0=zeros(ComplexF64,length(omegas))
    for iw in eachindex(omegas)
        y=ComplexF64[s.epsilon_M[iw] for s in finite_specs]
        cr=X\real.(y); ci=X\imag.(y); eps0[iw]=complex(cr[1],ci[1])
    end
    q0=(;omegas=collect(omegas),epsilon_M=eps0,epsilon1=real.(eps0),epsilon2=imag.(eps0),
         loss=-imag.(1 ./ eps0),qnorm,fit_power)

    # (a) Macroscopic dielectric function in the optical limit.
    pa=Plots.plot(omegas,q0.epsilon1;label="Re epsilon_M",xlabel="omega",ylabel="epsilon_M",
                  title="(a) Macroscopic dielectric function")
    Plots.plot!(pa,omegas,q0.epsilon2;label="Im epsilon_M")

    # (b) finite-q convergence plus q->0 extrapolation.
    pb=Plots.plot(;xlabel="omega",ylabel="Im epsilon_M",title="(b) Finite-q optical convergence")
    for (iq,s) in enumerate(finite_specs)
        Plots.plot!(pb,omegas,s.epsilon2;label="|q|=$(round(qnorm[iq],sigdigits=4))")
    end
    Plots.plot!(pb,omegas,q0.epsilon2;label="q -> 0",linewidth=3)

    # (c) Local-field-effect comparison at the smallest available finite q.
    # epsilon_00 is the no-matrix-inversion head; epsilon_M=1/[epsilon^-1]_00 includes wings/body.
    smin=finite_specs[1]
    eps00=ComplexF64[microscopic_head_epsilon(r) for r in smin.results]
    pc=Plots.plot(omegas,imag.(eps00);label="Im epsilon_00",xlabel="omega",ylabel="Im epsilon",
                  title="(c) Local-field effect, smallest finite q")
    Plots.plot!(pc,omegas,smin.epsilon2;label="Im {1/[epsilon^-1]_00}",linewidth=2)

    if !isnothing(prefix)
        Plots.savefig(pa,string(prefix,"_a.png")); Plots.savefig(pb,string(prefix,"_b.png")); Plots.savefig(pc,string(prefix,"_c.png"))
    end
    (;panel_a=pa,panel_b=pb,panel_c=pc,finite_q_spectra=finite_specs,q0_spectrum=q0,
      epsilon00_smallest_q=eps00,qnorm)
end
