#=
    stage1_geant4_sweep.jl
    ----------------------
    Stage 1: Geant4.jl energy × shielding configuration sweep.

    Runs a mono-energetic proton beam for each combination of:
      - Energy bin E_i in `E_BINS_MEV`  (30 log-spaced points, 10–1000 MeV)
      - Shielding config θ_j in `SHIELD_CONFIGS`  (Latin Hypercube sample, N_CONFIGS points)

    Each (E_i, θ_j) run fires `N_EVENTS` protons and records:
      - Total Ionizing Dose (TID) in the Si sensitive volume [Gy]
      - Fast-neutron flux (E_kin > 1 MeV) [steps/primary/cm³]

    Output: one JLD2 file per (E_i, θ_j) pair saved to `data/sims/`.

    Usage:
      cd InverseBTE
      julia --project scripts/stage1_geant4_sweep.jl

    To run a fast smoke-test (5 energies × 5 configs × 1000 events):
      julia --project scripts/stage1_geant4_sweep.jl --smoke-test
=#

using DrWatson
@quickactivate "InverseBTE"

using Geant4
using Geant4.SystemOfUnits
using JLD2
using Random
using Dates

# Include source modules directly (avoids full precompilation on first run)
include(joinpath(srcdir(), "geometry.jl"))
include(joinpath(srcdir(), "scoring.jl"))
using .Geometry
using .Scoring

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

const SMOKE_TEST = "--smoke-test" in ARGS

const N_E_BINS    = SMOKE_TEST ? 5  : 30
const N_CONFIGS   = SMOKE_TEST ? 5  : 2000
const N_EVENTS    = SMOKE_TEST ? 1_000 : 10_000
const N_THREADS   = Sys.CPU_THREADS   # Use all available CPU threads (32 on Ryzen 9 5950X)

# Log-spaced proton energy bins [MeV]
const E_BINS_MEV = 10 .^ range(log10(10.0), log10(1000.0), N_E_BINS)

# Silicon sensitive volume size [cm³]
const SV_VOL_CM3 = 1.0   # 1 cm³ cube (2 × 0.5 cm half-sides)

# ---------------------------------------------------------------------------
# Latin Hypercube sampling of shielding space
# ---------------------------------------------------------------------------

"""
    latin_hypercube_sample(n, bounds; seed=0) -> Matrix{Float64}

Generate `n` samples using stratified (LHS) sampling within `bounds`.
`bounds` is a vector of (lo, hi) tuples.
Returns an `n × ndim` matrix.
"""
function latin_hypercube_sample(
    n::Int, bounds::Vector{Tuple{Float64,Float64}}; seed::Int = 0
)::Matrix{Float64}
    rng = Random.seed!(Random.default_rng(), seed)
    ndim = length(bounds)
    X = Matrix{Float64}(undef, n, ndim)
    for (j, (lo, hi)) in enumerate(bounds)
        perm = randperm(rng, n)
        for i in 1:n
            u = (perm[i] - 1 + rand(rng)) / n   # stratified uniform in [0,1]
            X[i, j] = lo + u * (hi - lo)
        end
    end
    return X
end

# Shielding bounds: [t_Al_mm, t_Ta_mm, t_Poly_mm]
const SHIELD_BOUNDS_VEC = [
    (1.0, 20.0),   # Al  [mm]
    (0.5,  5.0),   # Ta  [mm]
    (5.0, 50.0),   # Poly [mm]
]
const SHIELD_CONFIGS = latin_hypercube_sample(N_CONFIGS, SHIELD_BOUNDS_VEC)

@info "="^60
@info "InverseBTE — Stage 1: Geant4 Energy × Shielding Sweep"
@info "  Smoke-test mode : $SMOKE_TEST"
@info "  Energy bins     : $N_E_BINS  ($(minimum(E_BINS_MEV)) – $(maximum(E_BINS_MEV)) MeV)"
@info "  Shield configs  : $N_CONFIGS"
@info "  Events/run      : $N_EVENTS"
@info "  Total runs      : $(N_E_BINS * N_CONFIGS)"
@info "="^60

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

sim_dir = datadir("sims")
mkpath(sim_dir)

# ---------------------------------------------------------------------------
# Initialize persistent Geant4 application
# ---------------------------------------------------------------------------

init_det = SatelliteDetector(; shield = ShieldParams(10.0, 2.0, 20.0))
sc_mesh  = build_scorers(0.5)   # 0.5 cm half-size = 1 cm³ mesh
fna = FastNeutronAccumulator(; sensitive_vol="SensitiveVol", threshold_MeV=1.0)
step_action(step, app) = fna(step, app)

gun = G4JLGunGenerator(
    particle  = "proton",
    energy    = E_BINS_MEV[1] * MeV,
    direction = G4ThreeVector(0.0, 0.0, 1.0),
    position  = G4ThreeVector(0.0, 0.0, -10.0 * cm),
)

app = G4JLApplication(
    detector          = init_det,
    generator         = gun,
    nthreads          = N_THREADS,
    physics_type      = FTFP_BERT,
    scorers           = G4JLScoringMesh[],
    stepaction_method = step_action,
)

configure(app)
initialize(app)

# Quiet verbose Geant4 UI logs to prevent terminal I/O bottlenecks
ui = G4UImanager!GetUIpointer()
ApplyCommand(ui, "/control/verbose 0")
ApplyCommand(ui, "/run/verbose 0")
ApplyCommand(ui, "/event/verbose 0")
ApplyCommand(ui, "/tracking/verbose 0")

# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

total_runs = N_E_BINS * N_CONFIGS
run_idx    = 0
t_start    = now()

for (ei, E_MeV) in enumerate(E_BINS_MEV)
    for ci in 1:N_CONFIGS
        global run_idx += 1

        t_Al_mm   = SHIELD_CONFIGS[ci, 1]
        t_Ta_mm   = SHIELD_CONFIGS[ci, 2]
        t_Poly_mm = SHIELD_CONFIGS[ci, 3]

        # Output filename — deterministic hash of parameters
        param_hash = string(hash((E_MeV, t_Al_mm, t_Ta_mm, t_Poly_mm)); base=16)[1:8]
        out_file = joinpath(sim_dir, "run_$(param_hash).jld2")

        if isfile(out_file)
            @info "[$(run_idx)/$(total_runs)] Skipping existing: $out_file"
            continue
        end

        if SMOKE_TEST || run_idx % 10 == 0 || run_idx == 1 || run_idx == total_runs
            @info "[$(run_idx)/$(total_runs)] E=$(round(E_MeV; digits=1)) MeV | \
                   Al=$(round(t_Al_mm; digits=1))mm Ta=$(round(t_Ta_mm; digits=2))mm Poly=$(round(t_Poly_mm; digits=1))mm"
        end

        # ---- Build detector & update gun ----
        shield = ShieldParams(t_Al_mm, t_Ta_mm, t_Poly_mm)
        det    = SatelliteDetector(; shield)

        gun_z = -(t_Al_mm + t_Ta_mm + t_Poly_mm) * mm - 5.0 * cm
        gun.data.energy = E_MeV * MeV
        gun.data.position = G4ThreeVector(0.0, 0.0, gun_z)

        # ---- Reinitialize geometry ----
        reinitialize(app, det)

        # ---- Run simulation ----
        beamOn(app, N_EVENTS)

        # ---- Extract results ----
        results = extract_results(sc_mesh, fna, N_EVENTS, SV_VOL_CM3)

        # ---- Save with DrWatson metadata ----
        @tagsave(out_file,
            Dict(
                "E_MeV"          => E_MeV,
                "shield_params"  => [t_Al_mm, t_Ta_mm, t_Poly_mm],
                "TID_Gy"         => results.TID_Gy,
                "neutron_flux"   => results.neutron_flux,
                "n_total_steps"  => results.n_total_steps,
                "n_primaries"    => N_EVENTS,
                "sv_vol_cm3"     => SV_VOL_CM3,
                "physics_list"   => "FTFP_BERT",
            );
            safe = true,
        )

        reset_accumulator!(fna)
    end
end

t_elapsed = now() - t_start
@info "Sweep complete. $(total_runs) runs in $(t_elapsed). Data in: $sim_dir"
