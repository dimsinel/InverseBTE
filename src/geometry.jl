#=
    geometry.jl
    -----------
    Parametric satellite shielding geometry for Geant4.jl.

    Geometry: A cubic spacecraft body composed of three concentric material
    shells (Aluminium → Tantalum → Polyethylene) surrounding a 1 cm³ silicon
    sensitive volume at the centre. All layers are implemented as nested G4Box
    volumes (Boolean subtraction approximation of spherical shells).

    Coordinate system: Origin at centre of sensitive volume.
    Beam direction: +Z (protons incident from -Z face).
=#

module Geometry

using Geant4
using Geant4.SystemOfUnits

export ShieldParams, shield_bounds
export SatelliteDetector

# ---------------------------------------------------------------------------
# Parameter struct
# ---------------------------------------------------------------------------

"""
    ShieldParams

Thickness (cm) of each shielding layer around the sensitive volume.

# Fields
- `t_Al`   : Aluminium outer shell thickness [cm]
- `t_Ta`   : Tantalum middle shell thickness [cm]
- `t_Poly` : Polyethylene inner shell thickness [cm]

# Physical ranges (for optimisation bounds)
```
julia> ShieldParams.BOUNDS
(Al=(1mm, 20mm), Ta=(0.5mm, 5mm), Poly=(5mm, 50mm))
```
"""
struct ShieldParams
    t_Al   :: Float64   # mm
    t_Ta   :: Float64   # mm
    t_Poly :: Float64   # mm
end

const SHIELD_BOUNDS = (
    Al   = (1.0, 20.0),   # mm
    Ta   = (0.5,  5.0),   # mm
    Poly = (5.0, 50.0),   # mm
)

"""Clamp a `ShieldParams` to physical bounds."""
function clamp_params(p::ShieldParams)::ShieldParams
    ShieldParams(
        clamp(p.t_Al,   SHIELD_BOUNDS.Al...),
        clamp(p.t_Ta,   SHIELD_BOUNDS.Ta...),
        clamp(p.t_Poly, SHIELD_BOUNDS.Poly...),
    )
end

"""
    shield_bounds() → NamedTuple

Return the shielding parameter bounds as a NamedTuple of (min, max) tuples [mm].
"""
shield_bounds() = SHIELD_BOUNDS

# ---------------------------------------------------------------------------
# Detector struct
# ---------------------------------------------------------------------------

"""
    SatelliteDetector <: G4JLDetector

Parametric nested-box spacecraft shielding detector.

# Constructor keyword arguments
- `shield`       : `ShieldParams` (default: nominal 10 mm Al / 2 mm Ta / 20 mm Poly)
- `sensitive_half` : half-side of the cubic Si sensitive volume [mm] (default: 5 mm → 1 cm³)
- `world_margin`   : extra clearance around outermost shell [cm] (default: 5 cm)
"""
mutable struct SatelliteDetector <: G4JLDetector
    shield         :: ShieldParams
    sensitive_half :: Float64      # mm  (half-side of Si cube)
    world_margin   :: Float64      # cm

    SatelliteDetector(;
        shield         = ShieldParams(10.0, 2.0, 20.0),
        sensitive_half = 5.0,
        world_margin   = 5.0,
    ) = new(shield, sensitive_half, world_margin)
end

# Register the constructor method that Geant4.jl dispatches to
Geant4.getConstructor(::SatelliteDetector)::Function = construct_satellite

# ---------------------------------------------------------------------------
# Geometry construction
# ---------------------------------------------------------------------------

"""
    construct_satellite(det::SatelliteDetector)::CxxPtr{G4VPhysicalVolume}

Build the nested-box shielding geometry and return the world physical volume.

Layer layout (half-sizes in Z, same in X and Y):

  [World air box]
    └── [Al outer box]           half = Si_half + t_Poly + t_Ta + t_Al
          └── [Ta middle box]    half = Si_half + t_Poly + t_Ta
                └── [Poly box]   half = Si_half + t_Poly
                      └── [Si sensitive volume]  half = Si_half
"""
function construct_satellite(det::SatelliteDetector)::CxxPtr{G4VPhysicalVolume}
    nist = G4NistManager!Instance()

    # --- Materials ---
    mat_air  = FindOrBuildMaterial(nist, "G4_AIR")
    mat_al   = FindOrBuildMaterial(nist, "G4_Al")
    mat_ta   = FindOrBuildMaterial(nist, "G4_Ta")
    mat_poly = FindOrBuildMaterial(nist, "G4_POLYETHYLENE")
    mat_si   = FindOrBuildMaterial(nist, "G4_Si")

    p  = det.shield
    sh = det.sensitive_half * mm

    # Layer half-sizes (same in X, Y, Z → cubic)
    h_si   = sh
    h_poly = sh + p.t_Poly * mm
    h_ta   = h_poly + p.t_Ta * mm
    h_al   = h_ta + p.t_Al * mm
    h_world = h_al + det.world_margin * cm

    checkOverlaps = false

    # --- World ---
    sol_world = G4Box("World", h_world, h_world, h_world)
    log_world = G4LogicalVolume(sol_world, mat_air, "World")
    phys_world = G4PVPlacement(
        nothing, G4ThreeVector(), log_world, "World", nothing, false, 0, checkOverlaps
    )

    # --- Aluminium outer shell ---
    sol_al = G4Box("ShieldAl", h_al, h_al, h_al)
    log_al = G4LogicalVolume(sol_al, mat_al, "ShieldAl")
    G4PVPlacement(
        nothing, G4ThreeVector(), log_al, "ShieldAl", log_world, false, 0, checkOverlaps
    )

    # --- Tantalum middle shell (daughter of Al) ---
    sol_ta = G4Box("ShieldTa", h_ta, h_ta, h_ta)
    log_ta = G4LogicalVolume(sol_ta, mat_ta, "ShieldTa")
    G4PVPlacement(
        nothing, G4ThreeVector(), log_ta, "ShieldTa", log_al, false, 0, checkOverlaps
    )

    # --- Polyethylene inner shell (daughter of Ta) ---
    sol_poly = G4Box("ShieldPoly", h_poly, h_poly, h_poly)
    log_poly = G4LogicalVolume(sol_poly, mat_poly, "ShieldPoly")
    G4PVPlacement(
        nothing, G4ThreeVector(), log_poly, "ShieldPoly", log_ta, false, 0, checkOverlaps
    )

    # --- Silicon sensitive volume (daughter of Poly) ---
    sol_si = G4Box("SensitiveVol", h_si, h_si, h_si)
    log_si = G4LogicalVolume(sol_si, mat_si, "SensitiveVol")
    G4PVPlacement(
        nothing, G4ThreeVector(), log_si, "SensitiveVol", log_poly, false, 0, checkOverlaps
    )

    # --- Visualisation ---
    SetVisAttributes(log_world, G4VisAttributes!GetInvisible())
    SetVisAttributes(log_al,   G4VisAttributes(G4Colour(0.75, 0.75, 0.75, 0.4)))  # grey Al
    SetVisAttributes(log_ta,   G4VisAttributes(G4Colour(0.55, 0.0,  0.55, 0.6)))  # purple Ta
    SetVisAttributes(log_poly, G4VisAttributes(G4Colour(0.0,  0.8,  0.2,  0.5)))  # green Poly
    SetVisAttributes(log_si,   G4VisAttributes(G4Colour(0.0,  0.5,  1.0,  0.9)))  # blue Si

    return phys_world
end

end # module Geometry
