#=
    surrogate.jl
    ------------
    Differentiable neural surrogate architecture for radiation transport.

    Architecture: Residual MLP in Lux.jl
      Input:  x ∈ ℝ^(N_E + 3)  (log-normalised spectrum + shielding params)
      Output: y ∈ ℝ^2           (z-standardised log10_TID, log10_Φ_n)

    Design choices:
      - GELU activation for smooth gradients through Zygote
      - LayerNorm after each residual block for stable training on
        high-dynamic-range physical data
      - Skip connection (identity path) avoids vanishing gradient in depth
      - Output head is linear (regression)
=#

module Surrogate

using Lux
using Random
using ComponentArrays

export build_surrogate, SurrogateConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

"""
    SurrogateConfig

Hyperparameters for the radiation transport neural surrogate.

# Fields
- `n_input`     : input dimension = N_E + 3
- `hidden_dim`  : width of all hidden layers (default 128)
- `n_res_blocks`: number of residual blocks (default 4)
- `n_output`    : output dimension (default 2: TID + Φ_n)
"""
Base.@kwdef struct SurrogateConfig
    n_input     :: Int = 33          # 30 energy bins + 3 shield params
    hidden_dim  :: Int = 128
    n_res_blocks:: Int = 4
    n_output    :: Int = 2
end

# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

"""
    ResidualBlock(dim)

Single residual block:
  y = LayerNorm(x + Dense(gelu(Dense(x))))

Both Dense layers are `dim → dim`. The skip connection requires the input
and output to share the same dimension.
"""
function ResidualBlock(dim::Int)
    return @compact(
        fc1    = Dense(dim => dim, gelu),
        fc2    = Dense(dim => dim),
        lnorm  = LayerNorm((dim,)),
    ) do x
        z = fc2(fc1(x))
        @return lnorm(x .+ z)
    end
end

# ---------------------------------------------------------------------------
# Full surrogate network
# ---------------------------------------------------------------------------

"""
    build_surrogate(cfg::SurrogateConfig) → Lux.Chain

Build the full ResNet surrogate as a `Lux.Chain`.

Network structure::
  Dense(n_input → hidden_dim, tanh)         # embedding
  ResidualBlock(hidden_dim) × n_res_blocks  # feature extraction
  Dense(hidden_dim → n_output)              # regression head

The tanh in the embedding compresses wide-range normalised inputs.
All remaining activations are GELU (smooth, Zygote-friendly).
"""
function build_surrogate(cfg::SurrogateConfig = SurrogateConfig())::Lux.Chain
    blocks = [ResidualBlock(cfg.hidden_dim) for _ in 1:cfg.n_res_blocks]
    return Chain(
        Dense(cfg.n_input  => cfg.hidden_dim, tanh),   # input embedding
        blocks...,                                      # res blocks
        Dense(cfg.hidden_dim => cfg.n_output),          # linear output head
    )
end

# ---------------------------------------------------------------------------
# Convenience: initialise parameters
# ---------------------------------------------------------------------------

"""
    init_surrogate(cfg=SurrogateConfig(); seed=0)
    → (model, ps, st)

Build surrogate and initialise parameters + state.
Returns `(model, ps, st)` where `ps` is a `ComponentArray` for easy
compatibility with `Optimization.jl`.
"""
function init_surrogate(
    cfg  :: SurrogateConfig = SurrogateConfig();
    seed :: Int = 0,
)
    rng = Random.seed!(Random.default_rng(), seed)
    model = build_surrogate(cfg)
    ps, st = Lux.setup(rng, model)
    ps_ca = ComponentArray(ps)   # flatten to ComponentArray for Optimization.jl
    return model, ps_ca, st
end

export init_surrogate

end # module Surrogate
