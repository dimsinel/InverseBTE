#=
    scoring.jl
    ----------
    Scoring infrastructure for the satellite radiation surrogate.

    Provides:
      1. `build_scorers(mesh_half_cm)`  → optional G4JLScoringMesh object
      2. `FastNeutronAccumulator`       : a G4JLSteppingAction that accumulates
           both Total Ionizing Dose (TID) and fast-neutron (E_kin > 1 MeV) flux
           directly in the mass geometry (`SensitiveVol`), avoiding parallel-world
           navigation boundaries (`GeomNav0003`).
      3. `extract_results(fna, n_primaries, sv_vol_cm3)` → NamedTuple with scalar TID and Φ_n.
=#

module Scoring

using Geant4
using Geant4.SystemOfUnits

export build_scorers, FastNeutronAccumulator, extract_results, reset_accumulator!

# ---------------------------------------------------------------------------
# 1.  ScoringMesh builders (Optional parallel scorer)
# ---------------------------------------------------------------------------

"""
    build_scorers(mesh_half_cm=0.5) → sc_mesh

Create a scoring mesh centred on the sensitive volume (origin).
"""
function build_scorers(mesh_half_cm::Float64 = 0.5)
    sc_mesh = G4JLScoringMesh(
        "Si_scorer",
        BoxMesh(mesh_half_cm*cm, mesh_half_cm*cm, mesh_half_cm*cm),
        bins = (1, 1, 1),
        quantities = [
            doseDeposit("TID_Gy"),
            nOfStep("n_total", filters = [ParticleFilter("nFilter", "neutron")]),
        ],
    )
    return sc_mesh
end

# ---------------------------------------------------------------------------
# 2.  Direct Mass-Geometry Stepping Action
# ---------------------------------------------------------------------------

"""
    FastNeutronAccumulator

Stepping action state that tracks total energy deposition (TID) and fast-neutron steps
directly inside the physical sensitive volume (`SensitiveVol`).
Bypasses parallel-world scoring meshes to eliminate stuck-track navigation warnings (`GeomNav0003`).
"""
mutable struct FastNeutronAccumulator
    fast_neutron_count  :: Int
    total_neutron_count :: Int
    edep_total_MeV      :: Float64
    sensitive_vol_name  :: String
    threshold_MeV       :: Float64

    FastNeutronAccumulator(; sensitive_vol::String = "SensitiveVol",
                              threshold_MeV::Float64 = 1.0) =
        new(0, 0, 0.0, sensitive_vol, threshold_MeV)
end

function (fna::FastNeutronAccumulator)(step, app)::Nothing
    # Check volume name of pre-step point
    pvol = GetTouchable(GetPreStepPoint(step)) |>
           GetVolume |> GetLogicalVolume |> GetName |> String
    pvol != fna.sensitive_vol_name && return nothing

    # Energy deposit in sensitive volume (for TID)
    edep = GetTotalEnergyDeposit(step) / MeV
    if edep > 0.0
        fna.edep_total_MeV += edep
    end

    # Check particle type for neutrons
    track = GetTrack(step)
    pdef  = GetDefinition(track)
    pname = GetParticleName(pdef) |> String
    if pname == "neutron"
        fna.total_neutron_count += 1
        ekin_MeV = GetKineticEnergy(GetPreStepPoint(step)) / MeV
        if ekin_MeV > fna.threshold_MeV
            fna.fast_neutron_count += 1
        end
    end

    return nothing
end

"""
    reset_accumulator!(fna::FastNeutronAccumulator)

Reset all counters to zero before a new run.
"""
function reset_accumulator!(fna::FastNeutronAccumulator)
    fna.fast_neutron_count  = 0
    fna.total_neutron_count = 0
    fna.edep_total_MeV      = 0.0
    return nothing
end

# ---------------------------------------------------------------------------
# 3.  Result extraction
# ---------------------------------------------------------------------------

"""
    extract_results(fna, n_primaries, sv_vol_cm3) → NamedTuple

Extract TID [Gy] and fast neutron flux [steps/primary/cm³] directly from the accumulator.
"""
function extract_results(fna::FastNeutronAccumulator,
                         n_primaries::Int, sv_vol_cm3::Float64 = 1.0)
    # Silicon mass [kg]: density = 2.33 g/cm³ -> 2.33e-3 kg per cm³
    mass_kg  = 2.33e-3 * sv_vol_cm3
    MeV_to_J = 1.602176634e-13
    TID_Gy   = (fna.edep_total_MeV * MeV_to_J) / (mass_kg * Float64(n_primaries))

    neutron_flux  = Float64(fna.fast_neutron_count) / (Float64(n_primaries) * sv_vol_cm3)
    n_total_steps = fna.total_neutron_count

    return (
        TID_Gy        = TID_Gy,
        neutron_flux  = neutron_flux,
        n_total_steps = n_total_steps,
    )
end

# Overload for backwards compatibility if sc_mesh is passed as first argument
function extract_results(sc_mesh, fna::FastNeutronAccumulator,
                         n_primaries::Int, sv_vol_cm3::Float64 = 1.0)
    return extract_results(fna, n_primaries, sv_vol_cm3)
end

end # module Scoring

