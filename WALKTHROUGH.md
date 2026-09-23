# InverseBTE — Pipeline Walkthrough
> Fast Neural Surrogate for Hadron/Proton Radiation Transport  
> and Inverse Shielding Optimization in a Satellite Subsystem

---

## ⚡ Quick-Start: Full Simulation Sequence

Run these commands in order from the project root. Each stage depends on the previous.

```bash
# 0. Activate the project (once per shell session)
cd ~/work/InverseBTE

# ── STAGE 1 ─────────────────────────────────────────────────────────────────
# Generate training data: sweep 30 proton energies × 200 shielding configs
# with FTFP_BERT_HP physics and score TID + fast-neutron flux in Si chip.
#
# → Smoke-test first (25 runs, ~minutes):
julia --project scripts/stage1_geant4_sweep.jl --smoke-test
#
# → Full production sweep (6 000 runs, many CPU-hours; can be interrupted
#   and resumed — already-computed runs are skipped automatically):
julia --project scripts/stage1_geant4_sweep.jl
# Output: data/sims/run_<hash>.jld2  (one file per (E, θ) pair)

# ── STAGE 2 ─────────────────────────────────────────────────────────────────
# Assemble raw JLD2 files into a normalised train/val dataset.
# Applies log10 transform + Z-score standardisation on targets.
julia --project scripts/stage2_preprocess.jl
# Output: data/processed/dataset.jld2
#         data/processed/normstats.jld2

# ── STAGE 3 ─────────────────────────────────────────────────────────────────
# Train the Lux.jl ResNet surrogate.
# Phase 1: Adam with cosine LR on RTX 5060 Ti  (batch=512, GPU-accelerated)
# Phase 2: L-BFGS full-batch refinement         (CPU)
julia --project scripts/stage3_train_surrogate.jl
# Force CPU:  julia --project scripts/stage3_train_surrogate.jl --cpu
# Output: data/processed/surrogate_best.jld2
#         plots/surrogate_loss.png
#         plots/surrogate_parity.png

# ── STAGE 4 ─────────────────────────────────────────────────────────────────
# Solve the inverse shielding design problem.
# Finds minimum-mass Al/Ta/Poly stack satisfying TID ≤ D_max.
julia --project scripts/stage4_inverse_opt.jl --D-max 1.0
# Mono-energetic test:   --E-peak 200.0      (200 MeV proton beam)
# Flat GCR proxy:        --flat-spectrum
# Output: data/processed/opt_result.jld2
#         plots/TID_sensitivity.png

# ── TESTS ───────────────────────────────────────────────────────────────────
# Run all unit tests (~30 s, CPU only):
julia --project test/runtests.jl
```

> [!IMPORTANT]
> Always run stages in order (1 → 2 → 3 → 4). Stage 2 requires at least a few
> dozen JLD2 files in `data/sims/` to produce a meaningful dataset. Use
> `--smoke-test` in Stage 1 for a rapid end-to-end validation before committing
> to the full sweep.

> [!TIP]
> The Stage 1 sweep is **resumable**: each output file is named by a deterministic
> hash of its parameters. Re-running the script skips any file that already exists.
> You can safely kill and restart it at any point.

---

## Architecture Overview

```
 Geant4.jl (Stage 1)
 ┌──────────────────────────────────────────────────────────┐
 │  Parametric geometry: Air → Al → Ta → Poly → Si chip     │
 │  Proton beam: 10–1000 MeV, 30 log-spaced energy bins     │
 │  Shielding: 200 LHS configs (t_Al, t_Ta, t_Poly)         │
 │  Scorers:  G4JLScoringMesh (TID)                         │
 │            FastNeutronAccumulator (Ekin > 1 MeV)         │
 │  Physics:  FTFP_BERT_HP                                   │
 └─────────────────┬────────────────────────────────────────┘
                   │  data/sims/*.jld2
                   ▼
 Dataset Pipeline (Stage 2)
 ┌──────────────────────────────────────────────────────────┐
 │  Feature:  x = [one_hot_spectrum(30); t/t_max × 3]       │
 │  Target:   y = [log10(TID); log10(Φ_n)]  + Z-score       │
 │  Split:    80% train / 20% val                           │
 └─────────────────┬────────────────────────────────────────┘
                   │  data/processed/dataset.jld2
                   ▼
 Neural Surrogate  (Stage 3)                     RTX 5060 Ti
 ┌──────────────────────────────────────────────────────────┐
 │  Dense(33→128, tanh)                                     │
 │  ResBlock × 4  [gelu → gelu → LayerNorm(x + skip)]       │
 │  Dense(128→2)                                            │
 │                                                          │
 │  Loss:    w_TID·Huber(ŷ_TID) + w_n·Huber(ŷ_n)  (log)   │
 │  Phase 1: Adam, cosine LR 1e-3→1e-5, batch=512 on GPU   │
 │  Phase 2: L-BFGS full-batch on CPU                      │
 └─────────────────┬────────────────────────────────────────┘
                   │  data/processed/surrogate_best.jld2
                   ▼
 Inverse Optimizer (Stage 4)               Zygote AD through surrogate
 ┌──────────────────────────────────────────────────────────┐
 │  min  m(θ) = ρ_Al·t_Al + ρ_Ta·t_Ta + ρ_Poly·t_Poly     │
 │   θ                                                      │
 │  s.t. TID_surrogate(S_ext, θ) ≤ D_max                   │
 │       θ_lo ≤ θ ≤ θ_hi                                    │
 │                                                          │
 │  Solver: penalty-aug. Adam (500) → L-BFGS (1000)        │
 └──────────────────────────────────────────────────────────┘
```

---

## File Reference

### Source Modules (`src/`)

| File | Module | Purpose |
|---|---|---|
| [`InverseBTE.jl`](src/InverseBTE.jl) | `InverseBTE` | Main module; re-exports all sub-modules and triggers GPU backend |
| [`geometry.jl`](src/geometry.jl) | `Geometry` | `SatelliteDetector <: G4JLDetector`; nested-box shielding; `ShieldParams` struct |
| [`scoring.jl`](src/scoring.jl) | `Scoring` | `build_scorers()` (TID + neutron mesh); `FastNeutronAccumulator` (E > 1 MeV stepper); `extract_results()` |
| [`dataset.jl`](src/dataset.jl) | `Dataset` | Load JLD2 records; log-scale + Z-score normalisation; `make_dataloaders()`; `inverse_transform_y()` |
| [`surrogate.jl`](src/surrogate.jl) | `Surrogate` | `build_surrogate(cfg)` → Lux ResNet; `init_surrogate()` → `(model, ps, st)` as `ComponentArray` |
| [`training.jl`](src/training.jl) | `Training` | `train_surrogate!()` — GPU Adam + CPU L-BFGS; Huber loss; `load_checkpoint()` |
| [`inverse.jl`](src/inverse.jl) | `Inverse` | `solve_inverse()` — penalty-aug. constrained NLP; `InverseResult` struct |

### Driver Scripts (`scripts/`)

| Script | Reads | Writes |
|---|---|---|
| [`stage1_geant4_sweep.jl`](scripts/stage1_geant4_sweep.jl) | — | `data/sims/run_*.jld2` |
| [`stage2_preprocess.jl`](scripts/stage2_preprocess.jl) | `data/sims/` | `data/processed/dataset.jld2`, `normstats.jld2` |
| [`stage3_train_surrogate.jl`](scripts/stage3_train_surrogate.jl) | `data/processed/dataset.jld2` | `surrogate_best.jld2`, loss/parity plots |
| [`stage4_inverse_opt.jl`](scripts/stage4_inverse_opt.jl) | `surrogate_best.jld2`, `normstats.jld2` | `opt_result.jld2`, sensitivity plot |

### Tests (`test/`)

| File | Tests |
|---|---|
| [`test_geometry.jl`](test/test_geometry.jl) | `ShieldParams` bounds; `clamp_params`; `SatelliteDetector` defaults |
| [`test_dataset.jl`](test/test_dataset.jl) | One-hot spectrum; normalisation round-trip; `make_dataloaders` shape |
| [`test_surrogate.jl`](test/test_surrogate.jl) | Forward pass shape `(2, batch)`; Zygote gradient non-NaN; Huber loss value |
| [`test_inverse.jl`](test/test_inverse.jl) | `build_input` length; `predict_physical` positivity; `solve_inverse` convergence |

---

## Stage 1 — Geant4 Data Generation

### Geometry

Protons enter from the −Z face. The sensitive volume (1 cm³ Si cube) sits at the
geometric centre of a stack of three concentric cubic shells:

```
World air box (clearance = 5 cm beyond Al)
└─ Al shell  (t_Al mm thick on all 6 faces)
    └─ Ta shell  (t_Ta mm)
        └─ Poly shell  (t_Poly mm)
            └─ Si chip  (2×5 mm = 1 cm³)  ← scored here
```

The geometry is **fully parametric** via `ShieldParams(t_Al, t_Ta, t_Poly)`.
Changing the struct and calling `configure(app); initialize(app)` rebuilds the
geometry in-place without restarting Julia.

### Shielding parameter space

| Material | Min [mm] | Max [mm] | ρ [g/cm³] |
|---|---|---|---|
| Aluminium (Al) | 1.0 | 20.0 | 2.699 |
| Tantalum (Ta) | 0.5 | 5.0 | 16.65 |
| Polyethylene (Poly) | 5.0 | 50.0 | 0.94 |

200 configurations are drawn by Latin Hypercube Sampling (LHS) for good
space-filling coverage.

### Scoring: TID

`G4JLScoringMesh("TID_scorer", BoxMesh(0.5cm, 0.5cm, 0.5cm), bins=(1,1,1),
quantities=[doseDeposit("TID_Gy")])`

A single 1×1×1 bin mesh co-located with the Si volume. Geant4 accumulates
energy deposit (eV) internally, divides by material mass (kg), and returns Gy.
The scorer is reset automatically between `beamOn` calls.

### Scoring: Fast neutrons (E > 1 MeV)

`G4JLScoringMesh` does not expose energy-threshold particle filters through the
Julia API. The workaround is a `G4JLSteppingAction` (`FastNeutronAccumulator`)
that counts neutron steps inside `SensitiveVol` with `E_kin > 1 MeV`.

```
Φ_n  =  fast_neutron_step_count / (N_primaries × V_sensitive_cm³)
       [steps/primary/cm³]     ← track-length fluence estimator
```

This is **proportional** to the true neutron fluence. The proportionality
constant (mean free path) cancels out when comparing across shielding configs,
making it a valid relative discriminator for the surrogate.

### Physics list: `FTFP_BERT_HP`

- **FTFP**: Fritiof + Bertini cascade for hadronic inelastic interactions
- **HP**: High-Precision neutron transport below 20 MeV (uses evaluated nuclear
  data files from the `G4NDL` dataset bundled with Geant4)
- Covers protons from 10 MeV (spallation threshold) to 1 GeV accurately

### JLD2 output schema

```julia
# data/sims/run_<8-hex-char-hash>.jld2
"E_MeV"          :: Float64          # incident proton energy
"shield_params"  :: Vector{Float64}  # [t_Al, t_Ta, t_Poly] in mm
"TID_Gy"         :: Float64          # total ionizing dose in Si [Gy]
"neutron_flux"   :: Float64          # fast-n track-length flux [steps/pri/cm³]
"n_total_steps"  :: Int              # total neutron steps (cross-check)
"n_primaries"    :: Int              # number of simulated protons
"sv_vol_cm3"     :: Float64          # sensitive volume [cm³] = 1.0
"physics_list"   :: String           # "FTFP_BERT_HP"
"gitcommit"      :: String           # DrWatson: git SHA at run time
"script"         :: String           # DrWatson: calling script path
```

---

## Stage 2 — Dataset Preprocessing

### Why log-scale normalization is critical

The TID in the Si chip spans roughly 10⁻⁹ Gy (10 MeV proton, thick shield) to
10⁻² Gy (1 GeV proton, thin shield) — a factor of ~10⁷. Training an MLP
directly on raw doses would cause high-energy samples to dominate the gradient
completely. Working in log₁₀ space compresses this to a 7-unit range, enabling
uniform learning across all energy and shielding configurations.

### Feature vector layout

```julia
# dim = N_E + 3 = 33
x = [one_hot_spectrum(E_MeV, E_bins);   # 30 floats: one bin = 1.0, rest = 0.0
     t_Al   / 20.0;                      # normalised to [0, 1]
     t_Ta   /  5.0;
     t_Poly / 50.0]
```

**Future extension for realistic spectra**: replace the one-hot vector with the
actual binned differential proton flux Φ(E_i) normalised to unit sum. The
surrogate input signature and dimension are **unchanged** — linear superposition
of mono-energetic responses holds for the dose (by linearity of the transport
equation at fixed geometry).

### Target normalization chain

```
Raw physics → log₁₀ transform → Z-score standardisation → network output
              (compresses range)    (zero-mean, unit-std)
```

Inverse chain for inference:
```julia
y_phys = 10 .^ (y_norm .* y_std .+ y_mean)
```

---

## Stage 3 — Neural Surrogate

### ResBlock detail

Each of the 4 residual blocks applies:

```
z = Dense(dim→dim, gelu)(x)     # branch
z = Dense(dim→dim)(z)           # projection (no activation → linear)
y = LayerNorm(x + z)            # add & norm
```

GELU activation is preferred over ReLU for:
- Smoother gradients through Zygote (no kink at 0)
- Better performance on physics regression tasks with near-zero activations

LayerNorm (not BatchNorm) is used because batch statistics are unreliable with
the small effective batch sizes that appear during the L-BFGS phase.

### GPU memory estimate (RTX 5060 Ti, 16 GiB)

| Quantity | Value |
|---|---|
| Model parameters | ~200 k scalars ≈ 0.8 MB |
| Batch (512 × 33 Float32) | ~0.07 MB |
| Gradient tape (Zygote) | ~5–10 MB |
| **Total VRAM used** | **< 50 MB** |

The model is tiny relative to VRAM — GPU benefit comes from **throughput**, not
memory capacity. With batch=512 the GPU processes all training data in far fewer
kernel launches per epoch than batch=64 on CPU.

### L-BFGS phase

After Adam has found a good basin, L-BFGS uses the full training set (no
mini-batching) to converge to a sharp local minimum. It is run on CPU because:
1. `OptimizationOptimJL.LBFGS()` uses Optim.jl internals (CPU only)
2. For this model size (~200k params) CPU L-BFGS is fast enough
3. No GPU memory transfer overhead for the full-batch gradient

---

## Stage 4 — Inverse Shielding Optimization

### Penalty method formulation

The constrained problem is converted to an unconstrained one:

```
L(θ) = m(θ) + λ · max(0, TID(θ) − D_max)²
```

With `λ = 1e5`, the quadratic penalty is steep enough that even a 1% violation
of the dose constraint incurs a penalty comparable to the full mass objective.

**Typical result** (SPE-proxy spectrum, D_max = 1 Gy):
```
t_Al   ≈  8–14 mm   (primary proton stopper)
t_Ta   ≈  1–3  mm   (secondary particle absorber, high Z)
t_Poly ≈  15–30 mm  (neutron moderator, H-rich)
Mass   ≈  5–15 g/cm²
```

### Gradient flow

```
θ (Float64 CPU)
  → build_input(S_ext, θ)       [Float32 vector]
  → Lux.apply(model, x, ps, st) [surrogate forward pass, CPU]
  → 10^(y_norm · y_std + y_mean) [physical TID, differentiable]
  → max(0, TID - D_max)²        [penalty]
  → Zygote.gradient()           [exact ∂L/∂θ]
  → Optimisers/LBFGS update
```

No finite differences anywhere. The entire chain from shielding thickness to
predicted dose is differentiable.

### Extending to real space radiation spectra

To use AP9/IRENE or a measured solar particle event spectrum:

```julia
# Replace the one-hot vector with the actual differential flux:
Φ_AP9 = load_ap9_spectrum(...)         # shape: (30,), same E_bins as training
Φ_norm = Φ_AP9 ./ sum(Φ_AP9)          # normalise to sum=1
S_ext  = Float32.(Φ_norm)

result = solve_inverse(model, best_ps, st, S_ext, norm_stats; cfg=INV_CFG)
```

The surrogate was trained with one-hot spectra (mono-energetic responses). Its
prediction for a composite spectrum `Φ_ext` is an approximation of the linear
superposition of mono-energetic responses — exact in the limit where the geometry
response is linear in the source, which holds for TID (energy deposit) but is
approximate for neutron flux (due to scattering non-linearities). For best
accuracy with realistic spectra, retrain with composite spectrum samples included
in Stage 1.

---

## Verified Smoke Test Results

```
✓  Module load (all 6 sub-modules): OK
✓  Surrogate forward pass (CPU)  shape=(2, 16): OK
✓  Zygote gradient through ResNet (no NaN): OK
✓  GPU forward pass  device=NVIDIA GeForce RTX 5060 Ti: OK
✓  Normstats log10 + Z-score round-trip: OK
✓  Inverse build_input length=33: OK
```

---

## Dependency Summary

```toml
# Key additions to Project.toml
CUDA              # NVIDIA CUDA runtime bindings
LuxCUDA           # Lux.jl CUDA backend
cuDNN             # cuDNN convolution kernels (used by NNlib)
MLDataDevices     # gpu_device() / cpu_device() API
JLD2              # fast HDF5-like checkpoint/data format
MLUtils           # DataLoader for mini-batch iteration
OptimizationOptimJL   # L-BFGS via Optim.jl
OptimizationMOI       # MathOptInterface bridge (for future exact constraints)
```
