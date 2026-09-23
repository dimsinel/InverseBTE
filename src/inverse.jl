#=
    inverse.jl
    ----------
    Differentiable inverse shielding optimisation.

    Problem statement:
      Given a fixed external proton environment S_ext (binned spectrum, normalised)
      and a regulatory TID limit D_max [Gy], find:

        θ* = argmin  m(θ)  =  ρ_Al·t_Al + ρ_Ta·t_Ta + ρ_Poly·t_Poly  [g/cm²]
             θ ∈ [θ_lo, θ_hi]
             s.t.  TID_surrogate(S_ext, θ) ≤ D_max

    Strategy:
      1. Penalty-augmented objective (fast, gradient-based via Adam/L-BFGS).
      2. Optional: Exact constrained solve via OptimizationMOI + Ipopt.

    The surrogate TID is fully differentiable through Zygote.jl, so exact
    analytic gradients are available for the constraint and objective.
=#

module Inverse

using Lux
using Zygote
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using ComponentArrays
using JLD2
using Printf

export InverseConfig, solve_inverse, InverseResult

# ---------------------------------------------------------------------------
# Density constants (g/cm³)
# ---------------------------------------------------------------------------
const RHO_AL   = 2.699
const RHO_TA   = 16.65
const RHO_POLY = 0.94

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

"""
    InverseConfig

Configuration for the inverse shielding optimisation.

# Fields
- `D_max_Gy`      : TID dose limit [Gy] (regulatory upper bound)
- `penalty_lambda` : quadratic penalty coefficient for constraint violation
- `t_bounds`       : (lo, hi) thickness bounds per material [mm]
                     Default: Al=(1,20), Ta=(0.5,5), Poly=(5,50)
- `optimizer`      : `:adam_lbfgs` (default) or `:lbfgs`
- `adam_iters`     : iterations for Adam warm-start
- `lbfgs_iters`    : iterations for L-BFGS refinement
- `lr_adam`        : Adam learning rate for inverse step
- `seed`           : RNG seed for warm-start noise
"""
Base.@kwdef struct InverseConfig
    D_max_Gy       :: Float64  = 1.0          # 1 Gy limit (typical for electronics)
    penalty_lambda :: Float64  = 1e4          # large penalty for constraint violation
    t_bounds_Al    :: Tuple{Float64,Float64}  = (1.0, 20.0)   # mm
    t_bounds_Ta    :: Tuple{Float64,Float64}  = (0.5,  5.0)   # mm
    t_bounds_Poly  :: Tuple{Float64,Float64}  = (5.0, 50.0)   # mm
    optimizer      :: Symbol   = :adam_lbfgs
    adam_iters     :: Int      = 500
    lbfgs_iters    :: Int      = 1000
    lr_adam        :: Float64  = 5e-3
    seed           :: Int      = 42
end

# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------

"""
    InverseResult

Optimisation result from `solve_inverse`.

# Fields
- `t_Al_mm`, `t_Ta_mm`, `t_Poly_mm` : optimal thicknesses [mm]
- `TID_Gy`        : surrogate-predicted TID at optimum [Gy]
- `neutron_flux`  : surrogate-predicted fast-neutron flux at optimum
- `mass_areal`    : total areal mass at optimum [g/cm²]
- `converged`     : whether D_max constraint is satisfied
- `retcode`       : optimiser return code
"""
struct InverseResult
    t_Al_mm      :: Float64
    t_Ta_mm      :: Float64
    t_Poly_mm    :: Float64
    TID_Gy       :: Float64
    neutron_flux :: Float64
    mass_areal   :: Float64
    converged    :: Bool
    retcode
end

# ---------------------------------------------------------------------------
# Helper: surrogate inference in physical units
# ---------------------------------------------------------------------------

"""
    predict_physical(model, ps, st, x_input, y_mean, y_std)
    → (TID_Gy, neutron_flux)

Run the surrogate on a single input vector and return physical-space outputs.
`y_mean`, `y_std` are the normalisation stats (NormStats fields).
"""
function predict_physical(model, ps, st, x_input::AbstractVector, y_mean, y_std)
    x_mat = reshape(x_input, :, 1)     # column vector → (N × 1) matrix
    y_norm, _ = Lux.apply(model, x_mat, ps, st)
    y_log  = y_norm[:, 1] .* Float32.(y_std) .+ Float32.(y_mean)
    y_phys = 10 .^ y_log
    return Float64(y_phys[1]), Float64(y_phys[2])  # TID_Gy, neutron_flux
end

# ---------------------------------------------------------------------------
# Feature vector builder for inverse problem
# ---------------------------------------------------------------------------

"""
    build_input(S_ext_norm, t_Al, t_Ta, t_Poly, T_AL_MAX, T_TA_MAX, T_POLY_MAX)
    → Vector{Float32}

Construct the surrogate input feature vector from:
- `S_ext_norm` : normalised external spectrum (N_E element vector)
- shielding thicknesses (mm), normalised by their max bounds
"""
function build_input(
    S_ext_norm   :: AbstractVector,
    t_Al  :: Real, t_Ta :: Real, t_Poly :: Real,
    T_AL_MAX  = 20.0, T_TA_MAX = 5.0, T_POLY_MAX = 50.0,
)
    return Float32.(vcat(
        S_ext_norm,
        t_Al   / T_AL_MAX,
        t_Ta   / T_TA_MAX,
        t_Poly / T_POLY_MAX,
    ))
end

# ---------------------------------------------------------------------------
# Areal mass objective
# ---------------------------------------------------------------------------

"""
    areal_mass(t_Al_mm, t_Ta_mm, t_Poly_mm) → Float64

Total areal mass of the shielding stack [g/cm²].
"""
areal_mass(t_Al, t_Ta, t_Poly) =
    (RHO_AL * t_Al + RHO_TA * t_Ta + RHO_POLY * t_Poly) / 10.0  # g/cm²

# ---------------------------------------------------------------------------
# Penalty-augmented objective
# ---------------------------------------------------------------------------

"""
    penalty_objective(t_vec, model, ps, st, S_ext_norm,
                      y_mean, y_std, cfg) → scalar

Penalty-augmented Lagrangian objective for the inverse problem:

  L(θ) = m(θ) + λ · max(0, TID(θ) - D_max)²

where m(θ) is the areal mass and TID is the surrogate prediction.
`t_vec` = [t_Al, t_Ta, t_Poly] in mm (unnormalised).
"""
function penalty_objective(t_vec, model, ps, st, S_ext_norm, y_mean, y_std, cfg)
    t_Al, t_Ta, t_Poly = t_vec[1], t_vec[2], t_vec[3]

    # Feature vector
    x = build_input(S_ext_norm, t_Al, t_Ta, t_Poly)

    # Surrogate prediction
    x_mat  = reshape(x, :, 1)
    y_norm, _ = Lux.apply(model, x_mat, ps, st)
    y_log  = y_norm[:, 1] .* Float32.(y_std) .+ Float32.(y_mean)
    TID_Gy_pred = 10 .^ y_log[1]  # differentiable

    # Objective: areal mass
    mass = areal_mass(t_Al, t_Ta, t_Poly)

    # Penalty for constraint violation
    violation = max(0, TID_Gy_pred - Float32(cfg.D_max_Gy))
    penalty   = Float64(cfg.penalty_lambda) * violation^2

    return mass + penalty
end

# ---------------------------------------------------------------------------
# Box-projection for bound constraints
# ---------------------------------------------------------------------------

function project_bounds(t_vec, cfg::InverseConfig)
    return [
        clamp(t_vec[1], cfg.t_bounds_Al...),
        clamp(t_vec[2], cfg.t_bounds_Ta...),
        clamp(t_vec[3], cfg.t_bounds_Poly...),
    ]
end

# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

"""
    solve_inverse(model, ps, st, S_ext_norm, norm_stats;
                  cfg=InverseConfig()) → InverseResult

Solve the inverse shielding design problem.

# Arguments
- `model`        : trained Lux surrogate model
- `ps`           : trained parameters (ComponentArray)
- `st`           : trained state
- `S_ext_norm`   : external proton spectrum, normalised (N_E-vector, same basis
                   as training input). For mono-energetic test: one-hot vector.
- `norm_stats`   : NormStats from training (provides y_mean, y_std)
- `cfg`          : InverseConfig

# Returns
`InverseResult` with optimal thicknesses, predicted dose, and mass.
"""
function solve_inverse(
    model       :: Lux.AbstractLuxLayer,
    ps,
    st,
    S_ext_norm  :: AbstractVector,
    norm_stats;
    cfg         :: InverseConfig = InverseConfig(),
)
    y_mean = norm_stats.y_mean
    y_std  = norm_stats.y_std

    # Warm-start: midpoint of bounds
    t0 = Float64[
        0.5 * (cfg.t_bounds_Al[1]   + cfg.t_bounds_Al[2]),
        0.5 * (cfg.t_bounds_Ta[1]   + cfg.t_bounds_Ta[2]),
        0.5 * (cfg.t_bounds_Poly[1] + cfg.t_bounds_Poly[2]),
    ]

    # Closure capturing model + data
    obj_fn = function(t, _extras)
        penalty_objective(t, model, ps, st, S_ext_norm, y_mean, y_std, cfg)
    end

    # Lower/upper bounds as vectors
    lb = [cfg.t_bounds_Al[1], cfg.t_bounds_Ta[1], cfg.t_bounds_Poly[1]]
    ub = [cfg.t_bounds_Al[2], cfg.t_bounds_Ta[2], cfg.t_bounds_Poly[2]]

    # Phase 1: Adam warm-start
    @info "Inverse solve — Phase 1: Adam warm-start ($(cfg.adam_iters) iters)"
    opt_prob_adam = OptimizationProblem(
        OptimizationFunction(obj_fn, AutoZygote()),
        t0,
    )
    sol_adam = solve(
        opt_prob_adam,
        Adam(cfg.lr_adam);
        maxiters = cfg.adam_iters,
        show_trace = false,
    )
    # Ensure warm-start u0 is strictly inside bounds for Fminbox
    t_adam = clamp.(sol_adam.u, lb .+ 1e-5, ub .- 1e-5)

    # Phase 2: L-BFGS refinement (box-constrained via Fminbox)
    @info "Inverse solve — Phase 2: L-BFGS refinement ($(cfg.lbfgs_iters) iters)"
    opt_prob_lbfgs = OptimizationProblem(
        OptimizationFunction(obj_fn, AutoZygote()),
        t_adam,
        nothing;
        lb = lb,
        ub = ub,
    )
    sol = solve(
        opt_prob_lbfgs,
        Fminbox(LBFGS());
        maxiters   = cfg.lbfgs_iters,
        show_trace = false,
    )
    t_opt = clamp.(sol.u, lb, ub)

    # Final evaluation
    TID_Gy, n_flux = predict_physical(model, ps, st, 
        build_input(S_ext_norm, t_opt...), y_mean, y_std)
    mass = areal_mass(t_opt...)
    converged = TID_Gy ≤ cfg.D_max_Gy

    @printf("\n=== Inverse Result ===\n")
    @printf("  t_Al   = %.2f mm\n",   t_opt[1])
    @printf("  t_Ta   = %.2f mm\n",   t_opt[2])
    @printf("  t_Poly = %.2f mm\n",   t_opt[3])
    @printf("  TID    = %.3e Gy  (limit: %.3e Gy)\n", TID_Gy, cfg.D_max_Gy)
    @printf("  Φ_fast = %.3e (steps/primary/cm³)\n", n_flux)
    @printf("  Mass   = %.3f g/cm²\n", mass)
    @printf("  Constraint satisfied: %s\n", converged ? "YES ✓" : "NO ✗ (increase penalty λ)")

    return InverseResult(
        t_opt[1], t_opt[2], t_opt[3],
        TID_Gy, n_flux, mass,
        converged, sol.retcode,
    )
end

end # module Inverse
