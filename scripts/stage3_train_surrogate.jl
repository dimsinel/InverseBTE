#=
    stage3_train_surrogate.jl
    -------------------------
    Stage 3: Neural surrogate training with GPU acceleration.

    Loads the preprocessed dataset from `data/processed/dataset.jld2`,
    builds the Lux.jl ResNet surrogate, and trains it using:
      Phase 1: Adam with cosine LR annealing  → GPU (RTX 5060 Ti)
      Phase 2: L-BFGS full-batch refinement   → CPU (L-BFGS is CPU-only)

    Checkpoints the best model to `data/processed/surrogate_best.jld2`.
    Saves training history and plots loss curves + parity plots.

    Usage:
      cd InverseBTE
      julia --project scripts/stage3_train_surrogate.jl

    To force CPU training:
      julia --project scripts/stage3_train_surrogate.jl --cpu
=#

using DrWatson
@quickactivate "InverseBTE"

using JLD2
using Lux
using LuxCUDA
using MLDataDevices
using Random
using MLUtils
using ComponentArrays
import CairoMakie: Axis, Figure, lines!, scatter!, axislegend, save

include(joinpath(srcdir(), "dataset.jl"))
include(joinpath(srcdir(), "surrogate.jl"))
include(joinpath(srcdir(), "training.jl"))
using .Dataset
using .Surrogate
using .Training

# ---------------------------------------------------------------------------
# GPU / CPU selection
# ---------------------------------------------------------------------------

FORCE_CPU = "--cpu" in ARGS
use_gpu_flag = FORCE_CPU ? false : nothing   # nothing = auto-detect

if !FORCE_CPU
    @info "CUDA functional: $(CUDA.functional())"
    if CUDA.functional()
        @info "GPU: $(CUDA.name(CUDA.device())), VRAM: $(round(CUDA.totalmem(CUDA.device()) / 1024^3; digits=1)) GiB"
    end
end

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const N_E_BINS   = 30    # must match stage1/stage2

# Larger batchsize to saturate RTX 5060 Ti (16 GB VRAM)
const BATCHSIZE  = FORCE_CPU ? 64 : 512

const TRAIN_CFG = TrainingConfig(
    adam_epochs    = 400,
    lbfgs_iters    = 0,         # 0 = rely on Adam (avoids full-dataset Zygote CPU memory explosion)
    lr_init        = 1e-3,
    lr_min         = 1e-5,
    batchsize      = BATCHSIZE,
    huber_delta    = 1.0f0,
    w_TID          = 1.0f0,
    w_neutron      = 0.5f0,
    checkpoint_dir = datadir("processed"),
    val_every      = 10,
    use_gpu        = use_gpu_flag,
)

const SURROGATE_CFG = SurrogateConfig(
    n_input      = N_E_BINS + 3,   # 30 energy bins + 3 shield params
    hidden_dim   = 128,
    n_res_blocks = 4,
    n_output     = 2,
)

# ---------------------------------------------------------------------------
# Load preprocessed dataset
# ---------------------------------------------------------------------------

@info "Stage 3: Loading preprocessed dataset"
dataset_file   = datadir("processed", "dataset.jld2")
normstats_file = datadir("processed", "normstats.jld2")

!isfile(dataset_file)   && error("Run stage2_preprocess.jl first! Missing: $dataset_file")
!isfile(normstats_file) && error("Missing normstats.jld2. Run stage2 first.")

dataset = jldopen(dataset_file, "r")
X_train = dataset["X_train"]
Y_train = dataset["Y_train"]
X_val   = dataset["X_val"]
Y_val   = dataset["Y_val"]
close(dataset)

normdata   = jldopen(normstats_file, "r")
norm_stats = NormStats(normdata["y_mean"], normdata["y_std"])
close(normdata)

@info "  Train: $(size(X_train, 2)) samples | Val: $(size(X_val, 2)) samples"
@info "  Features: $(size(X_train, 1)) | Outputs: $(size(Y_train, 1))"

# Rebuild DataLoaders (data stays on CPU; training loop streams to GPU)
train_loader, val_loader = make_dataloaders(
    X_train, Y_train, X_val, Y_val; batchsize = BATCHSIZE
)

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

@info "Building surrogate: $SURROGATE_CFG"
model, ps, st = init_surrogate(SURROGATE_CFG; seed = 0)
@info "  Parameters: $(length(ps)) scalars"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

@info "Starting training..."
best_ps, best_st, history = train_surrogate!(
    model, ps, st, train_loader, val_loader;
    cfg        = TRAIN_CFG,
    norm_stats = norm_stats,
)

# ---------------------------------------------------------------------------
# Final validation metrics (always on CPU)
# ---------------------------------------------------------------------------

@info "\nComputing final validation metrics..."
Y_hat_val, _ = Lux.apply(model, X_val, best_ps, best_st)
rrmse = Training.relative_rmse(
    Y_hat_val, Y_val, norm_stats.y_mean, norm_stats.y_std
)
@info "  Relative RMSE: TID = $(round(rrmse[1]; digits=2))% | Neutron flux = $(round(rrmse[2]; digits=2))%"

# ---------------------------------------------------------------------------
# Save training history and plots
# ---------------------------------------------------------------------------

history_file = datadir("processed", "training_history.jld2")
jldsave(history_file; train_loss = history.train_loss, val_loss = history.val_loss)

# Loss curves
fig = Figure(size=(900, 400))
ax  = Axis(fig[1,1],
    xlabel = "Epoch",
    ylabel = "Weighted Huber Loss (log scale)",
    title  = "Surrogate Training Loss",
    yscale = log10,
)
lines!(ax, 1:length(history.train_loss), history.train_loss, label="Train",  color=:royalblue)
val_epochs = TRAIN_CFG.val_every:TRAIN_CFG.val_every:length(history.train_loss)
scatter!(ax, collect(val_epochs), history.val_loss, label="Validation", color=:tomato, markersize=8)
axislegend(ax; position=:rt)
save(plotsdir("surrogate_loss.png"), fig)
@info "Loss plot saved → plots/surrogate_loss.png"

# Parity plot: predicted vs. true in physical space
Y_phys_pred = inverse_transform_y(Y_hat_val, norm_stats)
Y_phys_true = inverse_transform_y(Y_val, norm_stats)

fig2 = Figure(size=(900, 450))
for (k, label) in enumerate(["TID [Gy]", "Neutron flux [steps/pri/cm³]"])
    ax2 = Axis(fig2[1, k],
        xlabel = "Geant4 (true)",
        ylabel = "Surrogate (predicted)",
        title  = "Parity: $label",
        xscale = log10, yscale = log10,
    )
    scatter!(ax2, Y_phys_true[k, :], Y_phys_pred[k, :],
             alpha=0.4, color=:steelblue, markersize=5)
    xlims = extrema(filter(isfinite, Y_phys_true[k, :]))
    lines!(ax2, collect(xlims), collect(xlims), color=:red, linestyle=:dash, label="y=x")
    axislegend(ax2; position=:lt)
end
save(plotsdir("surrogate_parity.png"), fig2)
@info "Parity plot saved → plots/surrogate_parity.png"

@info "Stage 3 complete. Best model → $(datadir("processed", "surrogate_best.jld2"))"
