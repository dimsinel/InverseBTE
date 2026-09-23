#=
    dataset.jl
    ----------
    Data loading, normalization, and batching pipeline for the
    satellite radiation neural surrogate.

    Expected raw data format (produced by stage1_geant4_sweep.jl):
      Each sweep run produces a JLD2 file in `data/sims/` with:
        :E_MeV          Float64          — incident proton energy [MeV]
        :shield_params  Vector{Float64}  — [t_Al, t_Ta, t_Poly] in mm
        :TID_Gy         Float64          — Total Ionizing Dose [Gy]
        :neutron_flux   Float64          — fast-neutron flux [steps/primary/cm³]
        :n_primaries    Int              — number of simulated primaries

    This module:
      1. Globs all sim JLD2 files and assembles (X, Y) arrays.
      2. Applies log10 normalization (crucial for high dynamic range).
      3. Splits into train/validation sets.
      4. Wraps in MLUtils.DataLoader for minibatch iteration.
=#

module Dataset

using DrWatson
using JLD2
using MLUtils
using Statistics
using Random

export load_and_preprocess, make_dataloaders, NormStats, inverse_transform_y

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

const LOG_EPS = 1e-30   # floor before log10 to avoid -Inf

# Energy range [MeV] — used for normalising the spectrum vector input
const E_MIN_MEV = 10.0
const E_MAX_MEV = 1000.0

# Physical density [g/cm³] of shielding materials — for mass objective
const RHO_AL   = 2.699
const RHO_TA   = 16.65
const RHO_POLY = 0.94

# Normalisation bounds for shield thickness inputs [mm]
const T_AL_MAX   = 20.0
const T_TA_MAX   = 5.0
const T_POLY_MAX = 50.0

# ---------------------------------------------------------------------------
# Normalisation statistics struct
# ---------------------------------------------------------------------------

"""
    NormStats

Statistics for normalising the target variables (in log10 space).
Stored alongside the trained model so inference can invert the transform.

# Fields
- `y_mean` : Vector{Float64} of length 2 — mean of [log10_TID, log10_Φ_n]
- `y_std`  : Vector{Float64} of length 2 — std  of [log10_TID, log10_Φ_n]
"""
struct NormStats
    y_mean :: Vector{Float64}
    y_std  :: Vector{Float64}
end

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

"""
    load_raw_records(sim_dir) → Vector{NamedTuple}

Glob all `*.jld2` files in `sim_dir` and load each record.
Returns a vector of NamedTuples with keys matching the JLD2 schema.
"""
function load_raw_records(sim_dir::String)::Vector{NamedTuple}
    files = filter(f -> endswith(f, ".jld2"), readdir(sim_dir; join=true))
    isempty(files) && error("No JLD2 files found in $sim_dir")

    records = NamedTuple[]
    for f in files
        jldopen(f, "r") do io
            push!(records, (
                E_MeV         = io["E_MeV"],
                shield_params = io["shield_params"],   # [t_Al, t_Ta, t_Poly] mm
                TID_Gy        = io["TID_Gy"],
                neutron_flux  = io["neutron_flux"],
                n_primaries   = io["n_primaries"],
            ))
        end
    end
    return records
end

# ---------------------------------------------------------------------------
# Feature / target construction
# ---------------------------------------------------------------------------

"""
    build_spectrum_vector(E_MeV, E_bins) → Vector{Float32}

For the mono-energetic sweep, construct a one-hot (delta) spectrum vector:
the bin closest to `E_MeV` is 1, all others 0.

For realistic composite spectra (e.g. AP9/IRENE), pass the full flux vector
directly — the surrogate input is identical in structure.
"""
function build_spectrum_vector(E_MeV::Float64, E_bins::Vector{Float64})::Vector{Float32}
    idx = argmin(abs.(E_bins .- E_MeV))
    v = zeros(Float32, length(E_bins))
    v[idx] = 1.0f0
    return v
end

"""
    records_to_arrays(records, E_bins) → (X::Matrix{Float32}, Y::Matrix{Float32})

Convert raw simulation records into normalised feature matrix X and raw
(pre-standardisation) target matrix Y.

Feature vector layout:
  x = [one_hot_spectrum(N_E); t_Al/T_AL_MAX; t_Ta/T_TA_MAX; t_Poly/T_POLY_MAX]
  dim(x) = N_E + 3

Target vector layout (log10 of physical quantities):
  y = [log10(TID_Gy + LOG_EPS); log10(neutron_flux + LOG_EPS)]
  dim(y) = 2

Columns = samples (for compatibility with MLUtils/Lux batch-last convention).
"""
function records_to_arrays(
    records::Vector{<:NamedTuple},
    E_bins::Vector{Float64},
)::Tuple{Matrix{Float32}, Matrix{Float32}}

    N  = length(records)
    NE = length(E_bins)

    X = Matrix{Float32}(undef, NE + 3, N)
    Y = Matrix{Float32}(undef, 2,      N)

    for (i, r) in enumerate(records)
        spec = build_spectrum_vector(r.E_MeV, E_bins)
        tparams = Float32.(
            [r.shield_params[1] / T_AL_MAX,
             r.shield_params[2] / T_TA_MAX,
             r.shield_params[3] / T_POLY_MAX]
        )
        X[:, i] = vcat(spec, tparams)
        Y[:, i] = Float32.(
            [log10(r.TID_Gy       + LOG_EPS),
             log10(r.neutron_flux + LOG_EPS)]
        )
    end
    return X, Y
end

# ---------------------------------------------------------------------------
# Standardise targets
# ---------------------------------------------------------------------------

"""
    standardise_Y(Y) → (Y_norm, NormStats)

Z-score standardise each row (output dimension) of Y.
Returns the normalised matrix and the NormStats needed to invert.
"""
function standardise_Y(Y::Matrix{Float32})::Tuple{Matrix{Float32}, NormStats}
    mu  = mean(Y; dims=2) |> vec .|> Float64
    sig = std(Y;  dims=2) |> vec .|> Float64
    sig .= max.(sig, 1e-8)   # guard zero-std
    Y_norm = (Y .- Float32.(mu)) ./ Float32.(sig)
    return Y_norm, NormStats(mu, sig)
end

"""
    inverse_transform_y(Y_norm, ns::NormStats) → Y_phys

Invert the log10 + Z-score transform to recover physical quantities.

Returns a matrix with rows [TID_Gy, neutron_flux].
"""
function inverse_transform_y(Y_norm::AbstractMatrix, ns::NormStats)
    Y_log = Y_norm .* Float32.(ns.y_std) .+ Float32.(ns.y_mean)
    return 10 .^ Y_log    # back to physical units
end

# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

"""
    train_val_split(X, Y; split=0.8, rng=Random.default_rng())
    → (X_train, Y_train, X_val, Y_val)

Random column-wise split of (X, Y) with `split` fraction for training.
"""
function train_val_split(
    X::Matrix{Float32}, Y::Matrix{Float32};
    split::Float64 = 0.8,
    rng = Random.default_rng(),
)
    N      = size(X, 2)
    perm   = randperm(rng, N)
    n_train = round(Int, split * N)
    tr_idx = perm[1:n_train]
    vl_idx = perm[(n_train+1):end]
    return X[:, tr_idx], Y[:, tr_idx], X[:, vl_idx], Y[:, vl_idx]
end

# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

"""
    make_dataloaders(X_train, Y_train, X_val, Y_val;
                     batchsize=64, shuffle_train=true)
    → (train_loader, val_loader)

Wrap train and validation splits in `MLUtils.DataLoader` objects.
Each mini-batch is `(x_batch, y_batch)` with columns = samples.
"""
function make_dataloaders(
    X_train::Matrix{Float32}, Y_train::Matrix{Float32},
    X_val::Matrix{Float32},   Y_val::Matrix{Float32};
    batchsize::Int  = 64,
    shuffle_train::Bool = true,
)
    train_loader = DataLoader((X_train, Y_train);
        batchsize = batchsize, shuffle = shuffle_train, partial = false)
    val_loader   = DataLoader((X_val, Y_val);
        batchsize = size(X_val, 2), shuffle = false)  # full validation in one batch
    return train_loader, val_loader
end

# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

"""
    load_and_preprocess(sim_dir, E_bins;
                        split=0.8, batchsize=64, seed=42)
    → (train_loader, val_loader, norm_stats, N_features)

One-stop function: load raw Geant4 JLD2 files → build features/targets →
standardise → split → wrap in DataLoaders.

Also saves `data/processed/normstats.jld2` for use during inference.
"""
function load_and_preprocess(
    sim_dir  :: String,
    E_bins   :: Vector{Float64};
    split    :: Float64 = 0.8,
    batchsize:: Int     = 64,
    seed     :: Int     = 42,
    processed_dir :: String = joinpath(dirname(sim_dir), "processed"),
)
    rng = Random.seed!(Random.default_rng(), seed)

    @info "Loading raw simulation records from $sim_dir …"
    records = load_raw_records(sim_dir)
    @info "  Loaded $(length(records)) records."

    X, Y = records_to_arrays(records, E_bins)
    Y_norm, ns = standardise_Y(Y)

    X_tr, Y_tr, X_vl, Y_vl = train_val_split(X, Y_norm; split, rng)
    @info "  Train: $(size(X_tr, 2)) samples | Val: $(size(X_vl, 2)) samples"

    mkpath(processed_dir)
    ns_file = joinpath(processed_dir, "normstats.jld2")
    jldsave(ns_file; y_mean = ns.y_mean, y_std = ns.y_std)
    @info "  NormStats saved → $ns_file"

    train_loader, val_loader = make_dataloaders(X_tr, Y_tr, X_vl, Y_vl; batchsize)
    return train_loader, val_loader, ns, size(X, 1)
end

# ---------------------------------------------------------------------------
# Shielding mass objective (used by inverse optimiser)
# ---------------------------------------------------------------------------

"""
    shield_mass_areal(t_Al_mm, t_Ta_mm, t_Poly_mm) → Float64

Areal mass of the three shielding layers [g/cm²].
Used as the objective in the inverse optimisation problem.
"""
function shield_mass_areal(t_Al_mm, t_Ta_mm, t_Poly_mm)
    return (RHO_AL * t_Al_mm + RHO_TA * t_Ta_mm + RHO_POLY * t_Poly_mm) / 10.0
end

end # module Dataset
