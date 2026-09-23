#=
    InverseBTE.jl
    -------------
    Main module for the InverseBTE project.

    Fast Neural Surrogate for Hadron/Proton Radiation Transport
    and Inverse Shielding Optimization in a Satellite Subsystem.

    Stage pipeline:
      1. Geant4.jl data generation  (scripts/stage1_geant4_sweep.jl)
      2. Dataset preprocessing      (scripts/stage2_preprocess.jl)
      3. Surrogate training         (scripts/stage3_train_surrogate.jl)
      4. Inverse optimisation       (scripts/stage4_inverse_opt.jl)
=#

module InverseBTE

# --- Sub-modules ---
include("geometry.jl")
include("scoring.jl")
include("dataset.jl")
include("surrogate.jl")
include("training.jl")
include("inverse.jl")

using .Geometry
using .Scoring
using .Dataset
using .Surrogate
using .Training
using .Inverse

# --- GPU backend (loaded on import so CUDA extensions are triggered) ---
using LuxCUDA
using MLDataDevices

# --- Re-export public API ---
export ShieldParams, SatelliteDetector, shield_bounds
export build_scorers, FastNeutronAccumulator, extract_results, reset_accumulator!
export load_and_preprocess, make_dataloaders, NormStats, inverse_transform_y, shield_mass_areal
export SurrogateConfig, build_surrogate, init_surrogate
export TrainingConfig, train_surrogate!, load_checkpoint
export InverseConfig, InverseResult, solve_inverse

"""
    InverseBTE

Four-stage pipeline for fast neural surrogate radiation transport
and inverse shielding design.

See individual sub-module docs:
  - `Geometry`  : parametric satellite detector for Geant4.jl
  - `Scoring`   : TID and fast-neutron scoring utilities
  - `Dataset`   : data loading, normalization, batching
  - `Surrogate` : Lux.jl ResNet architecture
  - `Training`  : Adam + L-BFGS training loop
  - `Inverse`   : differentiable inverse optimisation

Quick start::

  using DrWatson
  @quickactivate "InverseBTE"
  using InverseBTE

  # Stage 1: run from scripts/stage1_geant4_sweep.jl
  # Stage 2: run from scripts/stage2_preprocess.jl
  # Stage 3:
  cfg = SurrogateConfig(n_input = 33)
  model, ps, st = init_surrogate(cfg)
  # ... load dataloaders ...
  best_ps, best_st, history = train_surrogate!(model, ps, st, train_dl, val_dl)

  # Stage 4:
  result = solve_inverse(model, best_ps, best_st, S_ext, norm_stats)
"""
InverseBTE

end # module InverseBTE
