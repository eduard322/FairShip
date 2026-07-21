# SND Design 4 (1 cm gap + copper plates) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SND_design 4` — Design 3's MTC + SiliconTarget + SiWCalo but with a 1 cm SiWCalo↔MTC gap and copper plates above/below the SiWCalo filling the Magn5–Magn6 gap.

**Architecture:** Design 4 is layered on Design 3 via flags on `ship_geo`. Placement change is a one-term substitution in the SiliconTarget auto-placement. Copper plates are passive `TGeoBBox` volumes built inside `ShipMuonShield::ConstructGeometry`, driven by parameters folded into the existing `SetSNDSpace` setter.

**Tech Stack:** C++20 (ROOT TGeo, FairRoot), Python 3.12 (`shipDet_conf.py`, `run_simScript.py`), YAML geometry configs, CMake+Ninja build, aliBuild/alienv environment.

## Global Constraints

- Designs 1, 2, 3 must remain behaviourally unchanged — all new behaviour gated behind `design == 4` or `getattr(..., default)` that reproduces old values.
- Units: all lengths in cm; use `u.cm` on the Python side where the surrounding code does.
- No new sensitive volumes — copper plates are passive (no `*Point`/digitisation).
- C++ edits must keep `clang-format` clean; Python edits must pass `ruff`.
- New files need SPDX headers (none created here — only edits).
- The build/run environment must be loaded first:
  `eval $(alienv load FairShip/latest --no-refresh)` (or `alienv enter FairShip/latest`).
- Incremental C++ build: `ninja -C build ShipPassive`.
- `$FAIRSHIP` points at the repo root.

---

## Task 1: YAML config — `mtcGap` and `copperPlates` block

**Files:**
- Modify: `geometry/SiWCalo_config.yaml`

**Interfaces:**
- Produces: YAML keys `SiWCalo.mtcGap` (float, cm) and
  `SiWCalo.copperPlates.{enabled: bool, width: float, thickness: float}` (cm),
  consumed by `python/shipDet_conf.py` Design-4 dispatch (Task 4).

- [ ] **Step 1: Add the keys to the SiWCalo config**

Append inside the `SiWCalo:` mapping in `geometry/SiWCalo_config.yaml`:

```yaml
  mtcGap: 1.0 # Design 4 only: gap (cm) between SiWCalo downstream face and MTC upstream face
  copperPlates: # Design 4 only: passive Cu plates above/below SiWCalo filling the Magn5-Magn6 gap
    enabled: true
    width: 54 # full x width (cm), matches SiWCalo targetWidth
    thickness: 10 # y thickness (cm) per plate
```

- [ ] **Step 2: Verify the YAML parses and keys are present**

Run:
```bash
python -c "import yaml; d=yaml.safe_load(open('geometry/SiWCalo_config.yaml'))['SiWCalo']; print(d['mtcGap'], d['copperPlates'])"
```
Expected: `1.0 {'enabled': True, 'width': 54, 'thickness': 10}`

- [ ] **Step 3: Commit**

```bash
git add geometry/SiWCalo_config.yaml
git commit -m "feat(SND): add Design-4 mtcGap and copperPlates config to SiWCalo"
```

---

## Task 2: C++ — extend `SetSNDSpace` signature and store copper-plate params

**Files:**
- Modify: `passive/ShipMuonShield.h` (declaration + members, near lines 34, 51)
- Modify: `passive/ShipMuonShield.cxx:56-74` (`SetSNDSpace` body)

**Interfaces:**
- Produces: new setter overload
  `void SetSNDSpace(Bool_t hole, Bool_t fillIron, const SNDDimensions& dims, Bool_t copperPlates, Double_t plateWidth, Double_t plateThickness)`
  with defaults `copperPlates=false, plateWidth=0., plateThickness=0.`;
  members `fCopperPlates`, `fCopperPlateWidth`, `fCopperPlateThickness`
  consumed by `ConstructGeometry` (Task 3).

- [ ] **Step 1: Update the declaration in `ShipMuonShield.h`**

Replace the existing declaration (currently at line 34):

```cpp
  void SetSNDSpace(Bool_t hole, Bool_t fillIron,
                   const SNDDimensions& snd_dimensions);
```

with:

```cpp
  void SetSNDSpace(Bool_t hole, Bool_t fillIron,
                   const SNDDimensions& snd_dimensions,
                   Bool_t copperPlates = false, Double_t plateWidth = 0.,
                   Double_t plateThickness = 0.);
```

- [ ] **Step 2: Add member variables in `ShipMuonShield.h`**

After the existing line `Double_t snd_hole_dx = 0., snd_hole_dy = 0.;` (around line 51) add:

```cpp
  Bool_t fCopperPlates{false};
  Double_t fCopperPlateWidth{0.}, fCopperPlateThickness{0.};
```

- [ ] **Step 3: Update the definition in `ShipMuonShield.cxx`**

Replace the signature and opening of `SetSNDSpace` (lines 56-61):

```cpp
void ShipMuonShield::SetSNDSpace(Bool_t hole, Bool_t fillIron,
                                 const SNDDimensions& dimensions) {
  snd_hole = hole;
  fill_iron = fillIron;
  snd_dimensions = dimensions;
```

with:

```cpp
void ShipMuonShield::SetSNDSpace(Bool_t hole, Bool_t fillIron,
                                 const SNDDimensions& dimensions,
                                 Bool_t copperPlates, Double_t plateWidth,
                                 Double_t plateThickness) {
  snd_hole = hole;
  fill_iron = fillIron;
  snd_dimensions = dimensions;
  fCopperPlates = copperPlates;
  fCopperPlateWidth = plateWidth;
  fCopperPlateThickness = plateThickness;
```

- [ ] **Step 4: Compile the passive library**

Run:
```bash
ninja -C build ShipPassive
```
Expected: build succeeds (no compile errors in `ShipMuonShield`).

- [ ] **Step 5: Commit**

```bash
git add passive/ShipMuonShield.h passive/ShipMuonShield.cxx
git commit -m "feat(SND): extend ShipMuonShield::SetSNDSpace with copper-plate params"
```

---

## Task 3: C++ — build copper plates in `ConstructGeometry`

**Files:**
- Modify: `passive/ShipMuonShield.cxx` — inside `ConstructGeometry`, after the
  magnet loop and before/after `top->AddNode(tShield, 1);` (around line 449).

**Interfaces:**
- Consumes: `fCopperPlates`, `fCopperPlateWidth`, `fCopperPlateThickness`,
  `snd_dimensions`, and local vectors `Z`, `Z_relf`, `nMagnets` (in scope after
  `Initialize`).
- Produces: two `copper` `TGeoBBox` nodes added to `tShield`.

- [ ] **Step 1: Add the copper-plate construction block**

Insert immediately after `top->AddNode(tShield, 1);` (line ~449, before the
`// Absorber` section):

```cpp
  // --- SND copper plates (Design 4): passive Cu slabs above and below the
  //     SiWCalo, spanning the full gap between the two SND middle magnets
  //     (Magn[nMagnets-3] = "Magn5" and Magn[nMagnets-2] = "Magn6"). ---
  if (fCopperPlates && snd_dimensions.count("SiWCalo")) {
    ShipGeo::InitMedium("copper");
    TGeoMedium* copper = gGeoManager->GetMedium("copper");

    // z-span = the physical gap between the two middle magnets.
    Double_t z5_down = Z[nMagnets - 3] + Z_relf[nMagnets - 3];
    Double_t z6_up = Z[nMagnets - 2] - Z_relf[nMagnets - 2];
    Double_t plate_zc = 0.5 * (z5_down + z6_up);
    Double_t plate_hz = 0.5 * (z6_up - z5_down);

    Double_t plate_hx = fCopperPlateWidth / 2.;
    Double_t plate_hy = fCopperPlateThickness / 2.;
    // Flush against the SiWCalo top/bottom faces.
    Double_t siwcalo_hy = snd_dimensions.at("SiWCalo").at("dy") / 2.;
    Double_t plate_yc = siwcalo_hy + plate_hy;

    TGeoVolume* cuPlate = gGeoManager->MakeBox("SND_CuPlate", copper, plate_hx,
                                               plate_hy, plate_hz);
    cuPlate->SetLineColor(kOrange + 7);
    tShield->AddNode(cuPlate, 1,
                     new TGeoTranslation(0., plate_yc, plate_zc));   // top
    tShield->AddNode(cuPlate, 2,
                     new TGeoTranslation(0., -plate_yc, plate_zc));  // bottom
  }

```

- [ ] **Step 2: Confirm `ShipGeo::InitMedium` is available in this translation unit**

Run:
```bash
grep -n "ShipGeo::InitMedium\|#include" passive/ShipMuonShield.cxx | grep -i "initmedium\|ShipGeo" | head
```
Expected: at least one existing `ShipGeo::InitMedium("iron")` call (line ~414) — same header already in scope, so no new include is needed. If the grep shows only the new usage, add `#include "ShipGeoConfig.h"` (or the header that declares `ShipGeo::InitMedium`, matching the `iron` usage).

- [ ] **Step 3: Compile the passive library**

Run:
```bash
ninja -C build ShipPassive
```
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add passive/ShipMuonShield.cxx
git commit -m "feat(SND): build copper plates across Magn5-Magn6 gap in ConstructGeometry"
```

---

## Task 4: Python — generalise the SiWCalo↔MTC gap and wire Design 4

**Files:**
- Modify: `python/shipDet_conf.py:166` (SiWCalo-length gate) and `:176-181`
  (SiliconTarget auto zPosition)
- Modify: `python/shipDet_conf.py:353-388` (design dispatch loop)
- Modify: `python/shipDet_conf.py:395-421` (`SetSNDSpace` guard + call)

**Interfaces:**
- Consumes: `ship_geo.snd_mtc_gap` (float cm, optional),
  `ship_geo.snd_copper_plates` (dict `{enabled, width, thickness}`, optional),
  YAML from Task 1.
- Consumes: `SetSNDSpace(..., copperPlates, plateWidth, plateThickness)` from
  Task 2.

- [ ] **Step 1: Generalise the SiWCalo-length gate to include Design 4**

In `configure_snd_siliconTarget`, change line 166 from:

```python
        if 3 in getattr(ship_geo, "SND_design", []):
```
to:
```python
        if any(x in getattr(ship_geo, "SND_design", []) for x in (3, 4)):
```

- [ ] **Step 2: Substitute the MTC-gap term in the auto zPosition**

In the same function, replace the `zPosition` assignment (lines 176-181):

```python
        ship_geo.SiliconTarget_geo.zPosition = (
            ship_geo.muShield.Entrance[-2]
            - ship_geo.muShield.Zgap[-2]
            - SiWCalo_total_length
            - ship_geo.SiliconTarget_geo.SiliconTarget_total_length / 2
        )
```

with:

```python
        mtc_gap = getattr(ship_geo, "snd_mtc_gap", ship_geo.muShield.Zgap[-2])
        ship_geo.SiliconTarget_geo.zPosition = (
            ship_geo.muShield.Entrance[-2]
            - mtc_gap
            - SiWCalo_total_length
            - ship_geo.SiliconTarget_geo.SiliconTarget_total_length / 2
        )
```

- [ ] **Step 3: Add the Design-4 dispatch branch**

In the `for design in ship_geo.SND_design:` loop, after the `elif design == 3:`
block (ends line ~379) and before the `else:` (line ~381), insert:

```python
            elif design == 4:
                # SND design 4 -- design 3 layout, but SiWCalo<->MTC gap = mtcGap
                # (default 1 cm) and copper plates across the Magn5-Magn6 gap.
                siwcalo_yaml = os.path.join(os.environ["FAIRSHIP"], "geometry", "SiWCalo_config.yaml")
                with open(siwcalo_yaml) as siwcalo_file:
                    siwcalo_cfg = yaml.safe_load(siwcalo_file)["SiWCalo"]
                ship_geo.snd_mtc_gap = siwcalo_cfg["mtcGap"] * u.cm
                ship_geo.snd_copper_plates = siwcalo_cfg["copperPlates"]
                detector_configs = {
                    "MTC_config": configure_snd_mtc,
                    "SiliconTarget_config": configure_snd_siliconTarget,
                    "SiWCalo_config": configure_snd_SiWCalo,
                }
                for config in detector_configs:
                    detector_configs[config](
                        os.path.join(os.environ["FAIRSHIP"], "geometry", config + ".yaml"), ship_geo
                    )
```

- [ ] **Step 4: Extend the `SetSNDSpace` guard and SiWCalo-dims gate to Design 4**

Change line 395 from:
```python
        if any(x in getattr(ship_geo, "SND_design", []) for x in (2, 3)):
```
to:
```python
        if any(x in getattr(ship_geo, "SND_design", []) for x in (2, 3, 4)):
```

Change line 410 from:
```python
            if 3 in ship_geo.SND_design:
```
to:
```python
            if any(x in ship_geo.SND_design for x in (3, 4)):
```

- [ ] **Step 5: Pass copper-plate args into the `SetSNDSpace` call**

Replace the call (lines 417-421):

```python
            MuonShield.SetSNDSpace(
                hole=True,
                fillIron=True,  # False: legacy full-length hole
                snd_dimensions=snd_dimensions,
            )
```

with:

```python
            cu = getattr(ship_geo, "snd_copper_plates", {"enabled": False, "width": 0.0, "thickness": 0.0})
            MuonShield.SetSNDSpace(
                hole=True,
                fillIron=True,  # False: legacy full-length hole
                snd_dimensions=snd_dimensions,
                copperPlates=bool(cu["enabled"]),
                plateWidth=cu["width"] * u.cm,
                plateThickness=cu["thickness"] * u.cm,
            )
```

- [ ] **Step 6: Lint the Python file**

Run:
```bash
ruff check python/shipDet_conf.py && ruff format --check python/shipDet_conf.py
```
Expected: no errors (run `ruff format python/shipDet_conf.py` if formatting differs, then re-check).

- [ ] **Step 7: Commit**

```bash
git add python/shipDet_conf.py
git commit -m "feat(SND): wire SND_design 4 (1cm MTC gap + copper plates)"
```

---

## Task 5: Python — register Design 4 in `run_simScript.py`

**Files:**
- Modify: `macro/run_simScript.py:371` (help text) and `:396`
  (`available_snd_designs`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `--SND_design 4` accepted by the CLI and forwarded to
  `geometry_config` / `shipDet_conf`.

- [ ] **Step 1: Extend the list of available designs**

Change line 396 from:
```python
available_snd_designs = [1, 2, 3]  # Extend this list as new designs are added
```
to:
```python
available_snd_designs = [1, 2, 3, 4]  # Extend this list as new designs are added
```

- [ ] **Step 2: Update the `--SND_design` help text**

Change the `help=` string (line 372) from:
```python
    help="Choose SND design(s) among [1,2,...] or 'all' to enable all. 1: EmulsionTarget, 2: MTC + SiliconTarget,  3: MTC + SiliconTarget + SiW Pixels",
```
to:
```python
    help="Choose SND design(s) among [1,2,...] or 'all' to enable all. 1: EmulsionTarget, 2: MTC + SiliconTarget, 3: MTC + SiliconTarget + SiW Pixels, 4: same as 3 with 1cm SiWCalo-MTC gap + copper plates",
```

- [ ] **Step 3: Verify the CLI accepts design 4**

Run:
```bash
python macro/run_simScript.py --help 2>&1 | grep -A2 "SND_design"
```
Expected: help text mentions `4: same as 3 ...`.

- [ ] **Step 4: Lint**

Run:
```bash
ruff check macro/run_simScript.py
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add macro/run_simScript.py
git commit -m "feat(SND): register SND_design 4 in run_simScript CLI"
```

---

## Task 6: Integration — build geometry with Design 4 and verify

**Files:**
- None modified. Verification only.

**Interfaces:**
- Consumes: everything from Tasks 1-5.

- [ ] **Step 1: Full rebuild of the affected library**

Run:
```bash
ninja -C build ShipPassive
```
Expected: success.

- [ ] **Step 2: Generate a Design-4 geometry**

Run:
```bash
cd "$FAIRSHIP" && python macro/run_simScript.py --SND --SND_design 4 -n 1 --tag snd4-check 2>&1 | tee /tmp/edursov/snd4.log | grep -iE "SiliconTarget zPosition|SiWCalo zPosition|Cu|copper|overlap"
```
Expected: prints resolved `SiliconTarget zPosition` and `SiWCalo zPosition`; no fatal errors. Simulation produces `geo_snd4-check.root`.

- [ ] **Step 3: Verify the 1 cm SiWCalo↔MTC gap and copper plates via the geometry**

Run this inspection snippet:
```bash
python - <<'PY'
import ROOT
ROOT.gGeoManager = None
f = ROOT.TFile.Open("geo_snd4-check.root")
# geo file stores the TGeoManager
gm = None
for k in f.GetListOfKeys():
    o = k.ReadObj()
    if isinstance(o, ROOT.TGeoManager):
        gm = o; break
assert gm, "no TGeoManager in geo file"
# find SiWCalo and MTC z-extents and copper plates
def zrange(namefrag):
    zmin, zmax = 1e9, -1e9
    it = ROOT.TGeoIterator(gm.GetTopVolume())
    node = it.Next()
    while node:
        if namefrag.lower() in node.GetName().lower():
            m = it.GetCurrentMatrix()
            t = m.GetTranslation()
            box = node.GetVolume().GetShape()
            try:
                dz = box.GetDZ()
            except Exception:
                dz = 0
            zmin = min(zmin, t[2]-dz); zmax = max(zmax, t[2]+dz)
        node = it.Next()
    return zmin, zmax
print("SiWCalo z:", zrange("SiWCalo"))
print("MTC z:", zrange("MTC"))
print("CuPlate z:", zrange("SND_CuPlate"))
PY
```
Expected: `SiWCalo` downstream edge and `MTC` upstream edge differ by ≈ 1 cm; `SND_CuPlate` prints a non-empty z-range (plates exist). Note: exact node-name matching may need adjustment to the actual assembly names printed — treat this as a guided check, not a rigid assertion.

- [ ] **Step 4: Run the overlap check**

Run:
```bash
python python/experimental/check_overlaps.py -g geo_snd4-check.root 2>&1 | tail -20
```
Expected: no illegal overlaps involving `SND_CuPlate`, `SiWCalo`, `SiliconTarget`, or the SND middle magnets. If overlaps appear, adjust plate `thickness`/`width` in `SiWCalo_config.yaml` or investigate the SiTarget→Magn5 clearance (see spec "Open verification items").

- [ ] **Step 5: Confirm Designs 1-3 are unchanged**

Run:
```bash
cd "$FAIRSHIP" && python macro/run_simScript.py --SND --SND_design 3 -n 1 --tag snd3-check 2>&1 | grep -iE "SiliconTarget zPosition|SiWCalo zPosition"
```
Expected: `SiliconTarget zPosition` matches the pre-change value (gap term = `Zgap[-2]`, not 1 cm), confirming Design 3 is untouched.

- [ ] **Step 6: Commit any config tweaks from overlap resolution (if needed)**

```bash
git add -A
git commit -m "test(SND): verify SND_design 4 geometry (gap + copper plates, overlaps clean)"
```

---

## Self-Review notes

- **Spec coverage:** Part 1 (1 cm gap) → Tasks 1,4. Part 2 (copper plates) →
  Tasks 1,2,3,4. Part 3 (wiring) → Tasks 4,5. Verification items → Task 6.
- **No unit-test harness:** FairShip geometry has no pytest; verification is
  build + geometry-macro + overlap check, matching repo conventions
  (`check_overlaps.py`, sim→geo chain).
- **Type consistency:** setter name `SetSNDSpace` and params
  `copperPlates/plateWidth/plateThickness` used identically in Tasks 2, 4;
  `ship_geo.snd_mtc_gap`, `ship_geo.snd_copper_plates` set in Task 4 and
  consumed in Task 4 (same file); member names `fCopperPlates*` consistent
  between Tasks 2 and 3.
- **Commit-per-task** so any single task can be reverted independently.
