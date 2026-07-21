# SND Design 4 — MTC + SiTarget + SiPad (SiWCalo) with 1 cm gap + copper plates

Date: 2026-07-21
Status: Approved (design), pending implementation plan

## Goal

Add a new `SND_design` value (`4`) that reuses Design 3's three subdetectors
(MTC, SiliconTarget, SiWCalo) but with two geometry changes:

1. Shift the SiTarget + SiWCalo block downstream toward the MTC so the
   **SiWCalo ↔ MTC gap is exactly 1 cm** (currently `muShield.Zgap[-2]`,
   ≈ 15 cm for `TRY_2026`).
2. Add **copper plates on the top and bottom of the SiWCalo**, spanning the
   full gap between the 5th and 6th muon-shield magnets (`Magn5`/`Magn6`), built
   inside the `ShipMuonShield` class and controlled through `SetSNDSpace`.

Designs 1–3 must remain behaviourally unchanged (byte-for-byte).

## Verified current-state facts

- Auto-placement order (upstream → downstream): **SiTarget → SiWCalo → gap →
  MTC**. SiWCalo's downstream face neighbours MTC's upstream face.
- MTC upstream face is at `muShield.Entrance[-2]`
  (`MTC.z = Entrance[-2] + mtc_len/2`).
- Current SiWCalo downstream face is at `Entrance[-2] - Zgap[-2]`, so the
  **current SiWCalo↔MTC gap equals `muShield.Zgap[-2]`**. Setting it to 1 cm is
  a one-term substitution in `configure_snd_siliconTarget`.
- The two SND "middle magnets" that already carry the SND hole are
  `Magn5 = Magn[nMagnets-3]` and `Magn6 = Magn[nMagnets-2]`
  (`nMagnets = 8` for `TRY_2026`, indices include `MagnAbsorb = 0`).
- In `ShipMuonShield::ConstructGeometry`, per-magnet global centres `Z[]` and
  half-lengths `Z_relf[]` are in scope after the magnet loop, so the Magn5→Magn6
  z-gap is directly computable there.
- The `copper` medium is already defined in `geometry/media.geo`.

## Design decisions (confirmed with user)

| Topic | Decision |
|-------|----------|
| Location of the 1 cm gap | Between **SiWCalo and MTC** faces |
| What moves | The **SiTarget + SiWCalo block together**, preserving internal spacing |
| Copper plate z-extent | **Full Magn5–Magn6 gap** |
| Copper plate API | **Folded into `SetSNDSpace`** |
| Copper plate x-width | = SiWCalo width (54 cm), **from YAML** |
| Copper plate y-thickness | **10 cm per plate, from YAML** |
| 1 cm gap source | **YAML value** (`mtcGap`), not hardcoded |
| Copper plate x-centre | **x = 0** (SiWCalo centre) |

## Implementation approach

Design 4 is layered on Design 3 via flags on `ship_geo`, rather than duplicating
the `configure_snd_*` functions. Minimal, additive, and leaves Designs 1–3
untouched.

### Part 1 — 1 cm SiWCalo↔MTC gap

- `geometry/SiWCalo_config.yaml`: add `mtcGap: 1.0` (cm).
- `python/shipDet_conf.py`, `configure_snd_siliconTarget`: replace the hardcoded
  `ship_geo.muShield.Zgap[-2]` term in the auto `zPosition` with
  `getattr(ship_geo, "snd_mtc_gap", ship_geo.muShield.Zgap[-2])`.
  - Design 3 (and any other) keeps `Zgap[-2]` via the default.
  - Design 4 sets `ship_geo.snd_mtc_gap = <mtcGap>*u.cm` before calling.
  - SiWCalo placement and the magnet SND voids automatically follow, since they
    key off each subdetector's resolved `z_pos`.

### Part 2 — Copper plates in `ShipMuonShield`

- `geometry/SiWCalo_config.yaml`: add
  ```yaml
  copperPlates:
    enabled: true
    width: 54      # x (cm), defaults to SiWCalo width
    thickness: 10  # y per plate (cm)
  ```
- `passive/ShipMuonShield.h/.cxx`: extend the setter
  ```cpp
  void SetSNDSpace(Bool_t hole, Bool_t fillIron, const SNDDimensions& dims,
                   Bool_t copperPlates = false,
                   Double_t plateWidth = 0., Double_t plateThickness = 0.);
  ```
  storing `fCopperPlates`, `fCopperPlateWidth`, `fCopperPlateThickness` as
  members. In `ConstructGeometry`, after the magnet loop, when
  `fCopperPlates` is set:
  - z-range = `[Z[n-3] + Z_relf[n-3], Z[n-2] - Z_relf[n-2]]`
    (n = `nMagnets`); z-centre and half-length from that range.
  - Build two `copper` `TGeoBBox` plates:
    - half-x = `plateWidth/2` (27 cm), centred at x = 0.
    - half-y = `plateThickness/2` (5 cm).
    - y-centres flush above/below SiWCalo:
      `±(snd_dimensions["SiWCalo"]["dy"]/2 + plateThickness/2)`.
  - Add both plates to the `MuonShieldArea` assembly (`tShield`).

### Part 3 — Wiring Design 4

- `macro/run_simScript.py`:
  - `available_snd_designs = [1, 2, 3, 4]`.
  - Update `--SND_design` help text:
    `4: MTC + SiliconTarget + SiWCalo (1 cm gap) + Cu plates`.
- `python/shipDet_conf.py` dispatch (`for design in ship_geo.SND_design`):
  - Add `elif design == 4:` running the same three `configure_snd_*` as Design 3,
    but first setting `ship_geo.snd_mtc_gap` (from `mtcGap`) and the copper-plate
    parameters (from the SiWCalo `copperPlates` block).
  - Extend the two existing membership checks that currently gate on `3` /
    `(2, 3)` to also include `4`:
    - SiWCalo-length offset in `configure_snd_siliconTarget`.
    - `SetSNDSpace` invocation guard and the `SiWCalo` entry in `snd_dimensions`.
  - Pass the copper-plate args into the existing `SetSNDSpace` call for Design 4.

## Open verification items (during implementation)

- Confirm SiWCalo is placed at x-centre 0 (needed for plate x-centring). If it
  carries an x-offset, mirror it for the plates.
- Run `python/experimental/check_overlaps.py` after the ~14 cm downstream shift:
  the SiTarget upstream face must still clear `Magn5`, and the copper plates must
  not overlap the magnet iron or the SiWCalo.
- Confirm the `copper` medium name resolves via `gGeoManager->GetMedium` /
  `ShipGeo::InitMedium` in the muon-shield context.

## Out of scope

- No changes to Designs 1, 2, 3.
- No digitisation/reconstruction changes (copper plates are passive; no
  sensitive volume).
- No retuning of magnet parameters or `shield_db`.
