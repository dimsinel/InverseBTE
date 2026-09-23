#=
    stage4_inverse_opt.jl
    ---------------------
    Stage 4: Differentiable inverse shielding optimisation.

    Loads the trained neural surrogate and solves:

      min  areal_mass(t_Al, t_Ta, t_Poly)
       t
      s.t. TID_surrogate(S_ext, t) <= D_max   [Gy]
           t_min <= t <= t_max                 [mm]

    using automatic differentiation (Zygote) through the surrogate.

    External environment `S_ext` is provided as a normalised spectrum vector.
    Default: worst-case flat GCR-equivalent spectrum at maximum proton fluence.

    Saves result to `data/processed/opt_result.jld2` and prints a summary.

    Usage:
      cd InverseBTE
      julia --project scripts/stage4_inverse_opt.jl

    Optional arguments:
      --D-max <Gy>     TID dose limit (default 1.0 Gy)
      --E-peak <MeV>   Single-energy test scenario (uses one-hot spectrum)
      --flat-spectrum  Use uniform flat external spectrum (GCR proxy)
=#

using DrWatson
@quickactivate "InverseBTE"

using JLD2
using Lux
using LuxCUDA
using MLDataDevices
using ComponentArrays
import CairoMakie: Axis, Figure, lines!, scatter!, hlines!, vlines!, axislegend, save

include(joinpath(srcdir(), "dataset.jl"))
include(joinpath(srcdir(), "surrogate.jl"))
include(joinpath(srcdir(), "training.jl"))
include(joinpath(srcdir(), "inverse.jl"))
using .Dataset
using .Surrogate
using .Training
using .Inverse

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

D_MAX_GY = 1.0       # default TID limit [Gy]
USE_FLAT = false
E_PEAK   = nothing

let args = copy(ARGS)
    while !isempty(args)
        a = popfirst!(args)
        a == "--D-max"        && (global D_MAX_GY = parse(Float64, popfirst!(args)))
        a == "--E-peak"       && (global E_PEAK   = parse(Float64, popfirst!(args)); )
        a == "--flat-spectrum" && (global USE_FLAT = true)
    end
end

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const N_E_BINS   = 30
const E_BINS_MEV = 10 .^ range(log10(10.0), log10(1000.0), N_E_BINS)

# Optimisation config
const INV_CFG = InverseConfig(
    D_max_Gy       = D_MAX_GY,
    penalty_lambda = 1e5,     # strong penalty for precise constraint enforcement
    t_bounds_Al    = (1.0, 20.0),
    t_bounds_Ta    = (0.5,  5.0),
    t_bounds_Poly  = (5.0, 50.0),
    optimizer      = :adam_lbfgs,
    adam_iters     = 500,
    lbfgs_iters    = 1000,
    lr_adam        = 5e-3,
)

const SURROGATE_CFG = SurrogateConfig(
    n_input      = N_E_BINS + 3,
    hidden_dim   = 128,
    n_res_blocks = 4,
    n_output     = 2,
)

# ---------------------------------------------------------------------------
# Load trained surrogate
# ---------------------------------------------------------------------------

@info "Stage 4: Inverse shielding optimisation"
@info "CUDA functional: $(CUDA.functional())"

# Use GPU for surrogate inference during sensitivity sweeps (fast)
# The optimizer (Adam/L-BFGS) still runs on CPU; surrogate calls inside
# penalty_objective are on CPU to stay AD-compatible with Optimization.jl.
# For large sensitivity sweeps, we can batch on GPU.
const INFER_DEV = CUDA.functional() ? gpu_device() : cpu_device()
@info "  Inference device: $(typeof(INFER_DEV))"

ckpt_file      = datadir("processed", "surrogate_best.jld2")
normstats_file = datadir("processed", "normstats.jld2")

!isfile(ckpt_file)      && error("No checkpoint found. Run stage3 first: $ckpt_file")
!isfile(normstats_file) && error("Missing normstats. Run stage2 first: $normstats_file")

best_ps, _ = load_checkpoint(ckpt_file)
normdata   = jldopen(normstats_file, "r")
norm_stats = NormStats(normdata["y_mean"], normdata["y_std"])
close(normdata)

# Re-build model architecture (must match training)
model, _, st = init_surrogate(SURROGATE_CFG)
# Convert ps to same type as loaded checkpoint if needed
best_ps = ComponentArray(best_ps)

@info "  Surrogate loaded from: $ckpt_file"

# ---------------------------------------------------------------------------
# Define external environment S_ext
# ---------------------------------------------------------------------------

if !isnothing(E_PEAK)
    # Single mono-energetic beam (one-hot spectrum)
    @info "  External spectrum: mono-energetic at E=$(E_PEAK) MeV"
    S_ext = Dataset.build_spectrum_vector(E_PEAK, E_BINS_MEV)
elseif USE_FLAT
    # Flat GCR-equivalent spectrum (all bins equal weight, normalised)
    @info "  External spectrum: flat (GCR proxy, uniform weights)"
    S_ext = Float32.(ones(N_E_BINS) / N_E_BINS)
else
    # Default: worst-case solar proton event — spectrum peaked at ~100 MeV
    @info "  External spectrum: SPE-proxy (100 MeV peak, soft power-law)"
    E_ref = 100.0   # MeV
    γ     = -1.5    # spectral index
    w = (E_BINS_MEV ./ E_ref) .^ γ
    w ./= sum(w)    # normalise to sum=1
    S_ext = Float32.(w)
end

@info "  D_max = $(D_MAX_GY) Gy"

# ---------------------------------------------------------------------------
# Solve inverse problem
# ---------------------------------------------------------------------------

result = solve_inverse(
    model, best_ps, st, S_ext, norm_stats;
    cfg = INV_CFG,
)

# ---------------------------------------------------------------------------
# Save result
# ---------------------------------------------------------------------------

result_file = datadir("processed", "opt_result.jld2")
@tagsave(result_file,
    Dict(
        "t_Al_mm"      => result.t_Al_mm,
        "t_Ta_mm"      => result.t_Ta_mm,
        "t_Poly_mm"    => result.t_Poly_mm,
        "TID_Gy"       => result.TID_Gy,
        "neutron_flux" => result.neutron_flux,
        "mass_areal"   => result.mass_areal,
        "converged"    => result.converged,
        "D_max_Gy"     => D_MAX_GY,
        "S_ext"        => S_ext,
    );
    safe = true,
)
@info "  Result saved → $result_file"

# ---------------------------------------------------------------------------
# Sensitivity scan: sweep t_Al while t_Ta, t_Poly fixed at optimum
# ---------------------------------------------------------------------------

@info "Plotting TID vs t_Al sensitivity (other params at optimum)"

t_Al_scan = range(INV_CFG.t_bounds_Al..., 40)
TID_scan  = Float64[]

for t_Al in t_Al_scan
    x_scan = Inverse.build_input(S_ext, t_Al, result.t_Ta_mm, result.t_Poly_mm)
    tid, _ = Inverse.predict_physical(model, best_ps, st, x_scan, norm_stats.y_mean, norm_stats.y_std)
    push!(TID_scan, tid)
end

fig = Figure(size=(800, 450))
ax = Axis(fig[1,1],
    xlabel = "Al shield thickness [mm]",
    ylabel = "Predicted TID [Gy]",
    title  = "TID sensitivity to Al thickness\n(Ta=$(round(result.t_Ta_mm;digits=2)) mm, Poly=$(round(result.t_Poly_mm;digits=1)) mm)",
    yscale = log10,
)
lines!(ax, collect(t_Al_scan), TID_scan, color=:royalblue, linewidth=2.5)
hlines!(ax, [D_MAX_GY], color=:red, linestyle=:dash, label="D_max = $(D_MAX_GY) Gy")
vlines!(ax, [result.t_Al_mm], color=:green, linestyle=:dot, label="Optimal t_Al = $(round(result.t_Al_mm;digits=2)) mm")
axislegend(ax; position=:rt)
save(plotsdir("TID_sensitivity.png"), fig)
@info "Sensitivity plot saved → plots/TID_sensitivity.png"

@info "\nStage 4 complete."
