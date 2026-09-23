#=
    stage2_preprocess.jl
    --------------------
    Stage 2: Dataset assembly, normalisation, and preprocessing.

    Loads all raw JLD2 simulation outputs from `data/sims/`,
    builds the feature matrix (normalised spectrum + shielding params) and
    target matrix (log10 TID, log10 neutron flux), applies Z-score
    standardisation on targets, splits into train/val, and saves the
    processed dataset and normalisation statistics.

    Output files (in `data/processed/`):
      - dataset.jld2    : {X_train, Y_train, X_val, Y_val, E_bins}
      - normstats.jld2  : {y_mean, y_std}

    Usage:
      cd InverseBTE
      julia --project scripts/stage2_preprocess.jl
=#

using DrWatson
@quickactivate "InverseBTE"

using JLD2
using Random
using Statistics

include(joinpath(srcdir(), "dataset.jl"))
using .Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const N_E_BINS   = 30      # Must match stage1 sweep
const SPLIT      = 0.80    # train fraction
const BATCHSIZE  = 64
const SEED       = 42

# Log-spaced energy bins [MeV] — must match stage1
const E_BINS_MEV = 10 .^ range(log10(10.0), log10(1000.0), N_E_BINS)

const SIM_DIR       = datadir("sims")
const PROCESSED_DIR = datadir("processed")

# ---------------------------------------------------------------------------
# Load and preprocess
# ---------------------------------------------------------------------------

@info "Stage 2: Data preprocessing"
@info "  Reading from : $SIM_DIR"
@info "  Writing to   : $PROCESSED_DIR"

train_loader, val_loader, norm_stats, N_features = load_and_preprocess(
    SIM_DIR, E_BINS_MEV;
    split    = SPLIT,
    batchsize = BATCHSIZE,
    seed     = SEED,
    processed_dir = PROCESSED_DIR,
)

@info "  N_features = $N_features  ($(N_E_BINS) energy bins + 3 shield params)"
@info "  NormStats  : y_mean=$(round.(norm_stats.y_mean; sigdigits=3))"
@info "              y_std =$(round.(norm_stats.y_std;  sigdigits=3))"

# ---------------------------------------------------------------------------
# Save full processed dataset (for inspection and L-BFGS phase)
# ---------------------------------------------------------------------------

# Collect all train and val data
X_train = reduce(hcat, [x for (x,_) in train_loader])
Y_train = reduce(hcat, [y for (_,y) in train_loader])
X_val   = reduce(hcat, [x for (x,_) in val_loader])
Y_val   = reduce(hcat, [y for (_,y) in val_loader])

dataset_file = joinpath(PROCESSED_DIR, "dataset.jld2")
jldsave(dataset_file;
    X_train   = X_train,
    Y_train   = Y_train,
    X_val     = X_val,
    Y_val     = Y_val,
    E_bins    = E_BINS_MEV,
    n_features = N_features,
)

@info "  Dataset saved → $dataset_file"
@info "  Train: $(size(X_train, 2)) samples, Val: $(size(X_val, 2)) samples"
@info "Stage 2 complete."
