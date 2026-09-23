#=
    training.jl
    -----------
    Training loop for the radiation transport neural surrogate.

    Two-phase strategy:
      Phase 1 (Adam, cosine LR annealing)  — epochs 1 … adam_epochs
        → runs on GPU if available (batches streamed to device each iteration)
      Phase 2 (L-BFGS full-batch)          — always runs on CPU
        (L-BFGS / OptimizationOptimJL are CPU-only; data is moved back before this phase)

    Loss function: element-wise Huber (smooth-L1) loss in z-standardised
    log10 space, with per-output weights to balance TID vs neutron flux.

    GPU usage:
      - `LuxCUDA` is loaded if available; `gpu_device()` returns a `CUDADevice`.
      - Model parameters/state and each mini-batch are moved to the device
        before forward/backward passes.
      - The best checkpoint is always saved in CPU form.
=#

module Training

using Lux
using LuxCUDA          # registers the CUDA backend for Lux
using MLDataDevices    # gpu_device() / cpu_device()
using Optimisers
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Zygote
using ComponentArrays
using Statistics
using Printf
using JLD2

export TrainingConfig, train_surrogate!, load_checkpoint
export gpu_device, cpu_device          # re-export for convenience in scripts

# ---------------------------------------------------------------------------
# GPU device detection
# ---------------------------------------------------------------------------

"""
    select_device() → device

Return a GPU device if CUDA is functional, otherwise CPU.
Prints a one-line info message so the user knows which device is being used.
"""
function select_device()
    dev = gpu_device()                 # MLDataDevices: CUDADevice or CPUDevice
    if dev isa MLDataDevices.CUDADevice
        @info "GPU training enabled — CUDA device detected ($(CUDA.name(CUDA.device())))"
    else
        @info "No CUDA GPU found — training on CPU."
    end
    return dev
end

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

"""
    TrainingConfig

Hyperparameters for surrogate training.

# Fields
- `adam_epochs`   : epochs for Phase 1 Adam (default 400)
- `lbfgs_iters`   : L-BFGS iterations for Phase 2 (default 200)
- `lr_init`       : initial Adam learning rate (default 1e-3)
- `lr_min`        : minimum LR during cosine decay (default 1e-5)
- `batchsize`     : mini-batch size (default 256 — larger batches saturate GPU better)
- `huber_delta`   : Huber loss delta in log10 units (default 1.0)
- `w_TID`         : loss weight for TID output (default 1.0)
- `w_neutron`     : loss weight for neutron flux output (default 0.5)
- `checkpoint_dir`: directory to save best model checkpoint
- `val_every`     : compute validation loss every N epochs (default 10)
- `use_gpu`       : force GPU on (true) / off (false), or auto-detect (nothing)
"""
Base.@kwdef struct TrainingConfig
    adam_epochs    :: Int     = 400
    lbfgs_iters    :: Int     = 200
    lr_init        :: Float64 = 1e-3
    lr_min         :: Float64 = 1e-5
    batchsize      :: Int     = 256       # larger for GPU efficiency
    huber_delta    :: Float32 = 1.0f0
    w_TID          :: Float32 = 1.0f0
    w_neutron      :: Float32 = 0.5f0
    checkpoint_dir :: String  = joinpath("data", "processed")
    val_every      :: Int     = 10
    use_gpu        :: Union{Bool,Nothing} = nothing   # nothing = auto
end

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

"""
    huber_loss(y_hat, y; δ=1.0f0) → scalar

Smooth-L1 (Huber) loss. For |r| ≤ δ: 0.5r²; for |r| > δ: δ(|r| - 0.5δ).
More robust than MSE for high-energy-tail outliers in radiation data.
Works on both CPU and GPU arrays.
"""
function huber_loss(y_hat::AbstractArray, y::AbstractArray; δ::Float32 = 1.0f0)
    r = y_hat .- y
    return mean(ifelse.(abs.(r) .≤ δ, 0.5f0 .* r.^2, δ .* (abs.(r) .- 0.5f0*δ)))
end

"""
    weighted_huber_loss(y_hat, y, cfg::TrainingConfig) → scalar

Per-output weighted Huber loss.
  L = w_TID · huber(ŷ₁, y₁) + w_neutron · huber(ŷ₂, y₂)
"""
function weighted_huber_loss(
    y_hat :: AbstractMatrix,
    y     :: AbstractMatrix,
    cfg   :: TrainingConfig,
)::Float32
    l_tid = huber_loss(y_hat[1:1, :], y[1:1, :]; δ = cfg.huber_delta)
    l_n   = huber_loss(y_hat[2:2, :], y[2:2, :]; δ = cfg.huber_delta)
    return cfg.w_TID * l_tid + cfg.w_neutron * l_n
end

# ---------------------------------------------------------------------------
# Validation metric
# ---------------------------------------------------------------------------

"""
    relative_rmse(y_hat_norm, y_norm, y_mean, y_std) → Vector{Float64}

Compute relative RMSE in physical space for each output.
Inputs are z-standardised log10 predictions and targets.
Returns [rRMSE_TID, rRMSE_neutron] as percentages.
Automatically moves tensors to CPU before computing.
"""
function relative_rmse(y_hat_norm, y_norm, y_mean, y_std)
    # Ensure on CPU for scalar extraction
    yh = Array(y_hat_norm)
    yt = Array(y_norm)
    # Back to log10 space
    y_hat_log = yh .* Float32.(y_std) .+ Float32.(y_mean)
    y_log     = yt .* Float32.(y_std) .+ Float32.(y_mean)
    # Physical space
    y_hat_phys = 10 .^ y_hat_log
    y_phys     = 10 .^ y_log
    # Relative RMSE per output
    rel_sq = ((y_hat_phys .- y_phys) ./ (y_phys .+ 1f-30)) .^ 2
    return 100.0 .* sqrt.(mean(rel_sq; dims=2)) |> vec .|> Float64
end

# ---------------------------------------------------------------------------
# Cosine LR schedule
# ---------------------------------------------------------------------------

"""
    cosine_lr(epoch, total_epochs, lr_init, lr_min) → Float64

Cosine annealing learning rate schedule.
"""
function cosine_lr(epoch::Int, total_epochs::Int, lr_init::Float64, lr_min::Float64)
    t = (epoch - 1) / max(total_epochs - 1, 1)
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + cos(π * t))
end

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

"""
    train_surrogate!(model, ps, st, train_loader, val_loader;
                     cfg=TrainingConfig(), norm_stats=nothing)
    → (best_ps, best_st, history)

Two-phase GPU-accelerated training:
  Phase 1: Adam with cosine LR annealing on GPU (if available).
           Each mini-batch is moved to the device on-the-fly.
  Phase 2: L-BFGS on the full training set, always on CPU.
           Parameters are moved back from GPU before this phase.

Returns best parameters (by validation loss, always on CPU),
best state (CPU), and a NamedTuple `history` with per-epoch losses.

Checkpoints best model to `cfg.checkpoint_dir/surrogate_best.jld2`.
"""
function train_surrogate!(
    model        :: Lux.AbstractLuxLayer,
    ps           :: ComponentArray,
    st,
    train_loader,
    val_loader;
    cfg          :: TrainingConfig = TrainingConfig(),
    norm_stats   = nothing,
)
    mkpath(cfg.checkpoint_dir)
    ckpt_path = joinpath(cfg.checkpoint_dir, "surrogate_best.jld2")

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    dev = if cfg.use_gpu === true
        gpu_device()
    elseif cfg.use_gpu === false
        cpu_device()
    else
        select_device()       # auto-detect
    end
    cpu_dev = cpu_device()

    # Move model params and state to device
    ps_dev = ps |> dev
    st_dev = st |> dev

    # ------------------------------------------------------------------
    # Pre-collect full training data on CPU (for L-BFGS phase)
    # ------------------------------------------------------------------
    X_train_full_cpu = reduce(hcat, [x for (x, _) in train_loader])
    Y_train_full_cpu = reduce(hcat, [y for (_, y) in train_loader])

    # Pre-collect full validation data on CPU
    X_val_cpu = reduce(hcat, [x for (x, _) in val_loader])
    Y_val_cpu = reduce(hcat, [y for (_, y) in val_loader])

    best_val_loss = Inf32
    best_ps_cpu   = ps |> cpu_dev   # always store CPU copy
    best_st_cpu   = st |> cpu_dev

    history_train = Float32[]
    history_val   = Float32[]

    # ------------------------------------------------------------------
    # Phase 1: Adam with cosine LR annealing (GPU)
    # ------------------------------------------------------------------
    @info "Phase 1: Adam (α₀=$(cfg.lr_init)) for $(cfg.adam_epochs) epochs [device: $(typeof(dev))]"
    opt_state = Optimisers.setup(Adam(cfg.lr_init), ps_dev)

    for epoch in 1:cfg.adam_epochs
        lr = cosine_lr(epoch, cfg.adam_epochs, cfg.lr_init, cfg.lr_min)
        Optimisers.adjust!(opt_state, lr)

        epoch_loss = 0.0f0
        n_batches  = 0

        for (x_batch_cpu, y_batch_cpu) in train_loader
            # Stream batch to device
            x_batch = x_batch_cpu |> dev
            y_batch = y_batch_cpu |> dev

            (loss, st_new), back = Zygote.pullback(ps_dev) do p
                y_hat, st_n = Lux.apply(model, x_batch, p, st_dev)
                weighted_huber_loss(y_hat, y_batch, cfg), st_n
            end
            gs = first(back((one(loss), nothing)))

            opt_state, ps_dev = Optimisers.update!(opt_state, ps_dev, gs)
            st_dev = st_new
            epoch_loss += Float32(loss)
            n_batches  += 1
        end
        epoch_loss /= n_batches
        push!(history_train, epoch_loss)

        # Validation (on CPU to avoid GPU memory pressure during reporting)
        if epoch % cfg.val_every == 0 || epoch == cfg.adam_epochs
            ps_cpu_tmp = ps_dev |> cpu_dev
            st_cpu_tmp = st_dev |> cpu_dev
            y_hat_val, _ = Lux.apply(model, X_val_cpu, ps_cpu_tmp, st_cpu_tmp)
            val_loss = weighted_huber_loss(y_hat_val, Y_val_cpu, cfg)
            push!(history_val, val_loss)

            @printf("  Epoch %4d/%d | train=%.4f | val=%.4f | lr=%.2e\n",
                    epoch, cfg.adam_epochs, epoch_loss, val_loss, lr)

            if val_loss < best_val_loss
                best_val_loss = val_loss
                best_ps_cpu   = ps_cpu_tmp
                best_st_cpu   = st_cpu_tmp
                jldsave(ckpt_path; ps = best_ps_cpu, cfg = cfg)
                @info "    └ Checkpoint saved (val_loss=$(round(Float64(val_loss); sigdigits=4)))"
            end
        end
    end

    # ------------------------------------------------------------------
    # Phase 2: L-BFGS on full training set (always CPU)
    # ------------------------------------------------------------------
    if cfg.lbfgs_iters > 0
        @info "Phase 2: L-BFGS refinement for $(cfg.lbfgs_iters) iterations [CPU]"
        ps_lbfgs = deepcopy(best_ps_cpu)
        st_lbfgs = deepcopy(best_st_cpu)

        function loss_fn_full(p, _extras)
            y_hat, _ = Lux.apply(model, X_train_full_cpu, p, st_lbfgs)
            return weighted_huber_loss(y_hat, Y_train_full_cpu, cfg)
        end

        opt_prob = OptimizationProblem(
            OptimizationFunction(loss_fn_full, AutoZygote()),
            ps_lbfgs,
        )
        try
            opt_sol = solve(opt_prob, LBFGS(); maxiters = cfg.lbfgs_iters, show_trace = false)
            ps_final = opt_sol.u

            # Final validation
            y_hat_final, _ = Lux.apply(model, X_val_cpu, ps_final, st_lbfgs)
            val_loss_final = weighted_huber_loss(y_hat_final, Y_val_cpu, cfg)
            @printf("  L-BFGS done | final val_loss=%.4f\n", val_loss_final)

            if val_loss_final < best_val_loss
                best_val_loss = val_loss_final
                best_ps_cpu   = ps_final
                best_st_cpu   = st_lbfgs
                jldsave(ckpt_path; ps = best_ps_cpu, cfg = cfg)
                @info "  L-BFGS improved checkpoint → saved."
            end
        catch e
            @warn "L-BFGS phase skipped or encountered error: $e"
        end
    end

    history = (
        train_loss = history_train,
        val_loss   = history_val,
        best_val   = best_val_loss,
    )
    return best_ps_cpu, best_st_cpu, history
end

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

"""
    load_checkpoint(path) → (ps, cfg)

Load a saved checkpoint produced by `train_surrogate!`.
Parameters are always stored on CPU; move to GPU in the caller if needed.
"""
function load_checkpoint(path::String)
    data = jldopen(path, "r")
    ps  = data["ps"]
    cfg = data["cfg"]
    close(data)
    return ps, cfg
end

end # module Training
